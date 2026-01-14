from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

from scipy.stats import chi2, ncx2
from numpy.linalg import inv


# ----------------------------
# I/O: Happy loader
# ----------------------------
HAPPY_COLS = [
    "id", "mag_r", "u_g", "g_r", "r_i", "i_z",
    "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5",
]

def load_happy(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    df = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None,
        names=HAPPY_COLS, engine="python"
    )
    if df.shape[1] != len(HAPPY_COLS):
        raise ValueError(f"Bad columns in {path.name}: got {df.shape[1]}")
    return df


# ----------------------------
# transforms
# ----------------------------
def tf(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)

def itf(z: np.ndarray) -> np.ndarray:
    return np.expm1(z)


# ----------------------------
# Hoeffding lambda (fast quantile form)
# ----------------------------
def find_lambda_fast(alpha: float, delta: float,
                     z_cal: np.ndarray, mu_cal: np.ndarray, var_cal: np.ndarray) -> float:
    n_cal = len(z_cal)
    threshold = max(0.0, alpha - np.sqrt(np.log(1.0 / delta) / (2.0 * n_cal)))
    sd = np.sqrt(np.maximum(var_cal, 1e-12))
    r = np.abs(z_cal - mu_cal) / sd

    q = 1.0 - threshold
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(r[~np.isnan(r)], q))


# ----------------------------
# P-spline (1D) utilities for GY
# ----------------------------
def second_diff_penalty(m: int) -> np.ndarray:
    I = np.eye(m)
    D2 = np.diff(I, n=2, axis=0)  # (m-2) x m
    return D2

def fit_pspline_1d(x: np.ndarray, y: np.ndarray,
                   n_knots: int = 12, degree: int = 3,
                   lam_grid: np.ndarray | None = None):
    """
    Penalized B-spline regression (P-spline) with 2nd-difference penalty.
    Returns dict with:
      - transformer
      - beta
      - lam
      - BtB, A, D2, B
      - fitted, df_effective
    """
    x = x.reshape(-1, 1)
    n = x.shape[0]

    # B-spline basis
    trans = SplineTransformer(
        n_knots=n_knots, degree=degree,
        include_bias=True  # to mimic spline with intercept
    )
    B = trans.fit_transform(x)  # n x m
    m = B.shape[1]

    BtB = B.T @ B
    D2 = second_diff_penalty(m)
    P = D2.T @ D2

    if lam_grid is None:
        lam_grid = np.logspace(-6, 6, 60)

    best = None
    y = y.reshape(-1, 1)

    for lam in lam_grid:
        A = inv(BtB + lam * P)              # m x m
        beta = A @ (B.T @ y)                # m x 1
        fitted = (B @ beta).ravel()
        resid = y.ravel() - fitted

        # effective df = trace(S) where S = B A B^T -> trace(A BtB)
        df_eff = float(np.trace(A @ BtB))

        # GCV (basic)
        denom = max(1e-8, (n - df_eff) ** 2)
        gcv = (n * np.sum(resid ** 2)) / denom

        if (best is None) or (gcv < best["gcv"]):
            best = dict(
                gcv=float(gcv), lam=float(lam),
                A=A, beta=beta, fitted=fitted, df_eff=df_eff,
                trans=trans, B=B, BtB=BtB, D2=D2, P=P
            )

    return best

def pspline_predict(fit, x_new: np.ndarray) -> np.ndarray:
    x_new = x_new.reshape(-1, 1)
    B_new = fit["trans"].transform(x_new)
    return (B_new @ fit["beta"]).ravel()

def compute_norm_fast(B: np.ndarray, A: np.ndarray, BtB: np.ndarray) -> np.ndarray:
    # ||S_{:,j}|| where S = B A B^T, computed as in your R helper
    M = A @ BtB @ A
    return np.sqrt(np.sum((B @ M) * B, axis=1))

def find_k_factor(nu: float, norm_lx_h: np.ndarray, P: float, gamma: float) -> np.ndarray:
    # sqrt( nu * chi2_{P, df=1, ncp=norm^2} / chi2_{1-gamma, df=nu} )
    denom = chi2.ppf(1.0 - gamma, df=nu)
    num = ncx2.ppf(P, df=1, nc=norm_lx_h ** 2)
    return np.sqrt(nu * num / denom)


# ----------------------------
# Ours (multivariate) : ML mean + ML var + Hoeffding lambda
# ----------------------------
def run_ours(dfA: pd.DataFrame, dfB: pd.DataFrame,
             feature_cols: list[str], y_col: str,
             alpha: float, delta: float,
             n_sample: int, seed: int):
    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    X = dfA_s[feature_cols].to_numpy()
    y = dfA_s[y_col].to_numpy()

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=0.5, random_state=seed, shuffle=True
    )
    z_tr, z_cal = tf(y_tr), tf(y_cal)

    mean_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", GradientBoostingRegressor(random_state=seed)),
    ])
    var_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", GradientBoostingRegressor(random_state=seed + 1)),
    ])

    mean_pipe.fit(X_tr, z_tr)
    mu_tr = mean_pipe.predict(X_tr)
    res2_tr = (z_tr - mu_tr) ** 2

    var_pipe.fit(X_tr, res2_tr)

    mu_cal = mean_pipe.predict(X_cal)
    var_cal = np.maximum(var_pipe.predict(X_cal), 1e-6)

    lam = find_lambda_fast(alpha, delta, z_cal, mu_cal, var_cal)

    # test on Happy B
    X_test = dfB[feature_cols].to_numpy()
    y_test = dfB[y_col].to_numpy()

    mu_test = mean_pipe.predict(X_test)
    var_test = np.maximum(var_pipe.predict(X_test), 1e-6)

    lower = np.maximum(itf(mu_test - lam * np.sqrt(var_test)), 0.0)
    upper = itf(mu_test + lam * np.sqrt(var_test))

    content = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mean_width = float(np.mean(upper - lower))

    return dict(method="Ours(multivar)", lambda_=lam, content=content, mean_width=mean_width)



# ----------------------------
# GY (1D baseline): x = mag_r only
# ----------------------------
def run_gy_1d(dfA: pd.DataFrame, dfB: pd.DataFrame,
             x_col: str, y_col: str,
             alpha: float, delta: float,
             n_sample: int, seed: int):
    """
    Python port of your GY pipeline on 1D x:
      - Work on z=log1p(y)
      - Fit mean spline on z
      - Fit variance spline on residual^2
      - Standardize: z / sqrt(varhat)
      - Fit spline on standardized response
      - Compute k-factor via ||l_x^T H|| using B, A, BtB
      - Form TI on standardized, map back, invert transform
    """
    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    x = dfA_s[x_col].to_numpy()
    y = dfA_s[y_col].to_numpy()
    z = tf(y)

    # 1) mean fit on z
    mean_fit = fit_pspline_1d(x, z, n_knots=12, degree=3)
    mu = mean_fit["fitted"]
    res2 = (z - mu) ** 2

    # 2) variance fit on res2
    var_fit = fit_pspline_1d(x, res2, n_knots=12, degree=3)
    var_hat = np.maximum(var_fit["fitted"], 1e-6)

    # 3) standardized response
    t = z / np.sqrt(var_hat)

    # 4) spline on standardized response (this provides B, A, BtB for k-factor)
    t_fit = fit_pspline_1d(x, t, n_knots=12, degree=3)

    B = t_fit["B"]
    BtB = t_fit["BtB"]
    A = t_fit["A"]
    t_hat = t_fit["fitted"]

    # residual variance on standardized scale
    n = len(x)
    nu = n - 1
    est_var = float(np.sum((t - t_hat) ** 2) / max(1.0, (n - 1)))

    # k-factor
    norm_vals = compute_norm_fast(B, A, BtB)
    k = find_k_factor(nu=nu, norm_lx_h=norm_vals, P=1.0 - alpha, gamma=delta)

    # TI on standardized scale
    upper_t = t_hat + np.sqrt(est_var) * k
    lower_t = t_hat - np.sqrt(est_var) * k

    # map back to z-scale
    upper_z = upper_t * np.sqrt(var_hat)
    lower_z = lower_t * np.sqrt(var_hat)

    upper = itf(upper_z)
    lower = np.maximum(itf(lower_z), 0.0)

    # test on Happy B: predict using the learned fits
    x_test = dfB[x_col].to_numpy()
    y_test = dfB[y_col].to_numpy()

    # For prediction, we recompute the pipeline on test x:
    # mean(z|x), var(res^2|x), then standardized spline prediction t_hat(x),
    # BUT k(x) needs norm(x) too. We'll approximate by evaluating basis at x_test.
    # (This is consistent with our pspline construction.)

    # mean & var at x_test
    mu_test = pspline_predict(mean_fit, x_test)
    var_test = np.maximum(pspline_predict(var_fit, x_test), 1e-6)

    # standardized spline at x_test
    # Build B_test from t_fit transformer
    x_test_ = x_test.reshape(-1, 1)
    B_test = t_fit["trans"].transform(x_test_)  # n_test x m
    beta_t = t_fit["beta"]
    t_hat_test = (B_test @ beta_t).ravel()

    # k(x_test): need norm at x_test
    # use the same M = A BtB A and compute sqrt(b^T M b) per row
    M = A @ BtB @ A
    norm_test = np.sqrt(np.sum((B_test @ M) * B_test, axis=1))
    k_test = find_k_factor(nu=nu, norm_lx_h=norm_test, P=1.0 - alpha, gamma=delta)

    upper_t_test = t_hat_test + np.sqrt(est_var) * k_test
    lower_t_test = t_hat_test - np.sqrt(est_var) * k_test

    upper_z_test = upper_t_test * np.sqrt(var_test)
    lower_z_test = lower_t_test * np.sqrt(var_test)

    upper_test = itf(upper_z_test)
    lower_test = np.maximum(itf(lower_z_test), 0.0)

    content = float(np.mean((y_test >= lower_test) & (y_test <= upper_test)))
    mean_width = float(np.mean(upper_test - lower_test))

    return dict(method="GY(1D mag_r)", lambda_=np.nan, content=content, mean_width=mean_width)

def main():
    root = Path(__file__).resolve().parents[1]
    dfA = load_happy(root / "data" / "Happy" / "happy_A")
    dfB = load_happy(root / "data" / "Happy" / "happy_B")

    alpha = 0.10
    delta = 0.05
    n_sample = 5000
    seed = 123

    # ---- Ours: multivariate features ----
    feature_cols = ["mag_r", "u_g", "g_r", "r_i", "i_z", "feat1", "feat2", "feat3", "feat4", "feat5"]
    y_col = "z_spec"

    res_ours = run_ours(dfA, dfB, feature_cols, y_col, alpha, delta, n_sample, seed)
    res_ours['method'] = "Ours(multivar)"

    # ---- Ours: 1D baseline using mag_r ----
    res_ours_1d = run_ours(
        dfA, dfB,
        feature_cols=["mag_r"],
        y_col=y_col,
        alpha=alpha, delta=delta,
        n_sample=n_sample, seed=seed
    )
    res_ours_1d['method'] = "Ours(1D mag_r)"

    # ---- GY: 1D baseline using mag_r ----
    res_gy = run_gy_1d(dfA, dfB, x_col="mag_r", y_col=y_col, alpha=alpha, delta=delta, n_sample=n_sample, seed=seed)
    res_gy['method'] = "GY(1D mag_r)"

    out = pd.DataFrame([res_ours, res_ours_1d, res_gy])
    print(out.to_string(index=False))


def run_many_seeds(dfA, dfB, seeds=range(1, 51), n_sample=5000, alpha=0.10, delta=0.05):
    rows = []
    for seed in seeds:
        # Ours(1D)
        r1 = run_ours(
            dfA, dfB,
            feature_cols=["mag_r"],
            y_col="z_spec",
            alpha=alpha, delta=delta,
            n_sample=n_sample, seed=seed
        )
        r1["method"] = "Ours(1D mag_r)"
        r1["seed"] = seed

        # GY(1D)
        r2 = run_gy_1d(
            dfA, dfB,
            x_col="mag_r", y_col="z_spec",
            alpha=alpha, delta=delta,
            n_sample=n_sample, seed=seed
        )
        r2["method"] = "GY(1D mag_r)"
        r2["seed"] = seed

        rows.extend([r1, r2])

    return pd.DataFrame(rows)

def main():
    root = Path(__file__).resolve().parents[1]
    dfA = load_happy(root / "data" / "Happy" / "happy_A")
    dfB = load_happy(root / "data" / "Happy" / "happy_B")

    df_res = run_many_seeds(dfA, dfB)

    print(
        df_res
        .groupby("method")[["content", "mean_width"]]
        .agg(["mean", "std", "median"])
        .round(4)
    )

    df_res.to_csv("results_happy_1d.csv", index=False)


if __name__ == "__main__":
    main()