from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

    r_ok = r[np.isfinite(r)]
    if r_ok.size == 0:
        return float("nan")
    return float(np.quantile(r_ok, q))


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
      - trans, beta, A, B, BtB
      - fitted
    """
    from sklearn.preprocessing import SplineTransformer

    x = x.reshape(-1, 1)
    n = x.shape[0]

    trans = SplineTransformer(
        n_knots=n_knots, degree=degree,
        include_bias=True
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
        A = inv(BtB + lam * P)
        beta = A @ (B.T @ y)
        fitted = (B @ beta).ravel()
        resid = y.ravel() - fitted

        df_eff = float(np.trace(A @ BtB))
        denom = max(1e-8, (n - df_eff) ** 2)
        gcv = (n * np.sum(resid ** 2)) / denom

        if (best is None) or (gcv < best["gcv"]):
            best = dict(
                gcv=float(gcv), lam=float(lam),
                A=A, beta=beta, fitted=fitted,
                trans=trans, B=B, BtB=BtB
            )

    return best

def pspline_predict(fit, x_new: np.ndarray) -> np.ndarray:
    x_new = x_new.reshape(-1, 1)
    B_new = fit["trans"].transform(x_new)
    return (B_new @ fit["beta"]).ravel()

def find_k_factor(nu: float, norm_lx_h: np.ndarray, P: float, gamma: float) -> np.ndarray:
    # sqrt( nu * chi2_{P, df=1, ncp=norm^2} / chi2_{1-gamma, df=nu} )
    denom = chi2.ppf(1.0 - gamma, df=nu)
    num = ncx2.ppf(P, df=1, nc=norm_lx_h ** 2)
    return np.sqrt(nu * num / denom)


def _norm_from_B_A_BtB(B: np.ndarray, A: np.ndarray, BtB: np.ndarray) -> np.ndarray:
    """
    norm(x) = sqrt( b(x)^T M b(x) ),  M = A BtB A
    for each row b(x) in B.
    """
    M = A @ BtB @ A
    return np.sqrt(np.sum((B @ M) * B, axis=1))


def _k_on_new_x(x_new: np.ndarray, fit, nu: float, P: float, gamma: float) -> np.ndarray:
    B_new = fit["trans"].transform(x_new.reshape(-1, 1))
    A = fit["A"]
    BtB = fit["BtB"]
    M = A @ BtB @ A
    norm_new = np.sqrt(np.sum((B_new @ M) * B_new, axis=1))
    return find_k_factor(nu=nu, norm_lx_h=norm_new, P=P, gamma=gamma)


# ============================================================
# Ours (1D) : ML mean + ML var + Hoeffding lambda
# ============================================================
def run_ours(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    alpha: float, delta: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,
):
    """
    Split dfA_s into train/cal.
    Optional parametric bootstrap augmentation is applied ONLY to training.
    """
    rng = np.random.default_rng(seed)

    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    X = dfA_s[[x_col]].to_numpy()
    y = dfA_s[y_col].to_numpy()
    z = tf(y)

    X_tr, X_cal, z_tr, z_cal = train_test_split(
        X, z, test_size=0.5, random_state=seed, shuffle=True
    )

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

    # initial fit on training
    mean_pipe.fit(X_tr, z_tr)
    mu_tr = mean_pipe.predict(X_tr)
    res2_tr = (z_tr - mu_tr) ** 2
    var_pipe.fit(X_tr, res2_tr)

    # bootstrap augmentation on training only
    if bootstrap_mult and bootstrap_mult > 0:
        n_syn = int(round(bootstrap_mult * X_tr.shape[0]))
        idx = rng.integers(0, X_tr.shape[0], size=n_syn)
        X_syn = X_tr[idx]

        mu_syn = mean_pipe.predict(X_syn)
        var_syn = np.maximum(var_pipe.predict(X_syn), 1e-12)
        z_syn = mu_syn + rng.normal(size=n_syn) * np.sqrt(var_syn)

        X_tr_aug = np.vstack([X_tr, X_syn])
        z_tr_aug = np.concatenate([z_tr, z_syn])

        mean_pipe.fit(X_tr_aug, z_tr_aug)
        mu_tr_aug = mean_pipe.predict(X_tr_aug)
        res2_tr_aug = (z_tr_aug - mu_tr_aug) ** 2
        var_pipe.fit(X_tr_aug, res2_tr_aug)

    # calibration lambda
    mu_cal = mean_pipe.predict(X_cal)
    var_cal = np.maximum(var_pipe.predict(X_cal), 1e-6)
    lam = find_lambda_fast(alpha, delta, z_cal, mu_cal, var_cal)

    # test on dfB
    X_test = dfB[[x_col]].to_numpy()
    y_test = dfB[y_col].to_numpy()

    mu_test = mean_pipe.predict(X_test)
    var_test = np.maximum(var_pipe.predict(X_test), 1e-6)

    z_lo = mu_test - lam * np.sqrt(var_test)
    z_hi = mu_test + lam * np.sqrt(var_test)

    lower = np.maximum(itf(z_lo), 0.0)
    upper = itf(z_hi)

    content = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mean_width = float(np.mean(upper - lower))

    return dict(method="HCTI", lambda_=lam, content=content, mean_width=mean_width, bootstrap_mult=bootstrap_mult)


# ============================================================
# GY-hetero (1D): your original heteroscedastic pipeline
# ============================================================
def run_gy_hetero(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    alpha: float, delta: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,
):
    rng = np.random.default_rng(seed)

    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    x = dfA_s[x_col].to_numpy()
    y = dfA_s[y_col].to_numpy()
    z = tf(y)

    # --- bootstrap augmentation (optional) on (x, z) using hetero generator ---
    if bootstrap_mult and bootstrap_mult > 0:
        # generator fits on original sample
        mean_fit0 = fit_pspline_1d(x, z, n_knots=12, degree=3)
        mu0 = mean_fit0["fitted"]
        res2_0 = (z - mu0) ** 2
        var_fit0 = fit_pspline_1d(x, res2_0, n_knots=12, degree=3)

        n_syn = int(round(bootstrap_mult * len(x)))
        idx = rng.integers(0, len(x), size=n_syn)
        x_syn = x[idx]

        mu_syn = pspline_predict(mean_fit0, x_syn)
        var_syn = np.maximum(pspline_predict(var_fit0, x_syn), 1e-12)
        z_syn = mu_syn + rng.normal(size=n_syn) * np.sqrt(var_syn)
        y_syn = np.maximum(itf(z_syn), 0.0)

        x = np.concatenate([x, x_syn])
        y = np.concatenate([y, y_syn])
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

    # 4) spline on standardized response
    t_fit = fit_pspline_1d(x, t, n_knots=12, degree=3)

    B = t_fit["B"]
    BtB = t_fit["BtB"]
    A = t_fit["A"]
    t_hat = t_fit["fitted"]

    n = len(x)
    nu = n - 1
    est_var = float(np.sum((t - t_hat) ** 2) / max(1.0, (n - 1)))

    # k(x) on training x (not necessary for prediction but kept for consistency)
    norm_train = _norm_from_B_A_BtB(B, A, BtB)
    _ = find_k_factor(nu=nu, norm_lx_h=norm_train, P=1.0 - alpha, gamma=delta)

    # ---- test ----
    x_test = dfB[x_col].to_numpy()
    y_test = dfB[y_col].to_numpy()

    var_test = np.maximum(pspline_predict(var_fit, x_test), 1e-6)

    # t_hat(x_test)
    B_test = t_fit["trans"].transform(x_test.reshape(-1, 1))
    t_hat_test = (B_test @ t_fit["beta"]).ravel()

    # k(x_test)
    k_test = _k_on_new_x(x_test, t_fit, nu=nu, P=1.0 - alpha, gamma=delta)

    upper_t = t_hat_test + np.sqrt(est_var) * k_test
    lower_t = t_hat_test - np.sqrt(est_var) * k_test

    upper_z = upper_t * np.sqrt(var_test)
    lower_z = lower_t * np.sqrt(var_test)

    upper = itf(upper_z)
    lower = np.maximum(itf(lower_z), 0.0)

    content = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mean_width = float(np.mean(upper - lower))

    return dict(method="Parametric TI-hetero", lambda_=np.nan, content=content, mean_width=mean_width, bootstrap_mult=bootstrap_mult)


# ============================================================
# GY-homo (1D): homoscedastic version (no variance spline / no standardization)
# ============================================================
def run_gy_homo(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    alpha: float, delta: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,
):
    rng = np.random.default_rng(seed)

    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    x = dfA_s[x_col].to_numpy()
    y = dfA_s[y_col].to_numpy()
    z = tf(y)

    # --- bootstrap augmentation (optional) using homo generator ---
    if bootstrap_mult and bootstrap_mult > 0:
        mean_fit0 = fit_pspline_1d(x, z, n_knots=12, degree=3)
        z_hat0 = mean_fit0["fitted"]
        n0 = len(x)
        nu0 = n0 - 1
        s2_0 = float(np.sum((z - z_hat0) ** 2) / max(1.0, nu0))

        n_syn = int(round(bootstrap_mult * n0))
        idx = rng.integers(0, n0, size=n_syn)
        x_syn = x[idx]

        mu_syn = pspline_predict(mean_fit0, x_syn)
        z_syn = mu_syn + rng.normal(size=n_syn) * np.sqrt(max(s2_0, 1e-12))
        y_syn = np.maximum(itf(z_syn), 0.0)

        x = np.concatenate([x, x_syn])
        y = np.concatenate([y, y_syn])
        z = tf(y)

    # 1) mean fit on z
    mean_fit = fit_pspline_1d(x, z, n_knots=12, degree=3)
    z_hat = mean_fit["fitted"]

    n = len(x)
    nu = n - 1
    s2 = float(np.sum((z - z_hat) ** 2) / max(1.0, nu))  # homoscedastic variance estimate

    # k(x) depends on leverage norm from mean spline fit
    norm_train = _norm_from_B_A_BtB(mean_fit["B"], mean_fit["A"], mean_fit["BtB"])
    _ = find_k_factor(nu=nu, norm_lx_h=norm_train, P=1.0 - alpha, gamma=delta)

    # ---- test ----
    x_test = dfB[x_col].to_numpy()
    y_test = dfB[y_col].to_numpy()

    z_hat_test = pspline_predict(mean_fit, x_test)
    k_test = _k_on_new_x(x_test, mean_fit, nu=nu, P=1.0 - alpha, gamma=delta)

    upper_z = z_hat_test + np.sqrt(max(s2, 1e-12)) * k_test
    lower_z = z_hat_test - np.sqrt(max(s2, 1e-12)) * k_test

    upper = itf(upper_z)
    lower = np.maximum(itf(lower_z), 0.0)

    content = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mean_width = float(np.mean(upper - lower))

    return dict(method="Parametric TI-homo", lambda_=np.nan, content=content, mean_width=mean_width, bootstrap_mult=bootstrap_mult)


# ============================================================
# Run many seeds (3-way)
# ============================================================
def run_many_seeds_3way(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    seeds=range(1, 51),
    n_sample: int = 5000,
    alpha: float = 0.10,
    delta: float = 0.05,
    bootstrap_mult: float = 0.0,
    x_col: str = "mag_r",
    y_col: str = "z_spec",
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rows.append({**run_ours(dfA, dfB, x_col, y_col, alpha, delta, n_sample, seed, bootstrap_mult),
                     "seed": seed})
        rows.append({**run_gy_homo(dfA, dfB, x_col, y_col, alpha, delta, n_sample, seed, bootstrap_mult),
                     "seed": seed})
        rows.append({**run_gy_hetero(dfA, dfB, x_col, y_col, alpha, delta, n_sample, seed, bootstrap_mult),
                     "seed": seed})
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    dfA = load_happy(root / "data" / "Happy" / "happy_A")
    dfB = load_happy(root / "data" / "Happy" / "happy_B")

    alpha = 0.10
    delta = 0.05
    n_sample = 5000
    seeds = range(1, 51)

    # no bootstrap
    df0 = run_many_seeds_3way(
        dfA, dfB,
        seeds=seeds, n_sample=n_sample, alpha=alpha, delta=delta,
        bootstrap_mult=0.0
    )
    df0["setting"] = "no_bootstrap"

    # bootstrap
    df1 = run_many_seeds_3way(
        dfA, dfB,
        seeds=seeds, n_sample=n_sample, alpha=alpha, delta=delta,
        bootstrap_mult=1.0
    )
    df1["setting"] = "bootstrap_mult=1.0"

    df_all = pd.concat([df0, df1], ignore_index=True)

    summary = (
        df_all
        .groupby(["setting", "method"])[["content", "mean_width"]]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    print(summary)

    out_csv = "results_happy_3way.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Optional: quick sanity check for inf
    bad = df_all[~np.isfinite(df_all["mean_width"])]
    if len(bad) > 0:
        print("\n[WARN] Non-finite mean_width detected (showing first 20 rows):")
        print(bad[["setting", "method", "seed", "content", "mean_width"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
