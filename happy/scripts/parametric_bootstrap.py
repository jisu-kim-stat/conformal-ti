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
      - transformer
      - beta
      - lam
      - BtB, A, D2, B
      - fitted, df_effective
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
                A=A, beta=beta, fitted=fitted, df_eff=df_eff,
                trans=trans, B=B, BtB=BtB, D2=D2, P=P
            )

    return best

def pspline_predict(fit, x_new: np.ndarray) -> np.ndarray:
    x_new = x_new.reshape(-1, 1)
    B_new = fit["trans"].transform(x_new)
    return (B_new @ fit["beta"]).ravel()

def compute_norm_fast(B: np.ndarray, A: np.ndarray, BtB: np.ndarray) -> np.ndarray:
    M = A @ BtB @ A
    return np.sqrt(np.sum((B @ M) * B, axis=1))

def find_k_factor(nu: float, norm_lx_h: np.ndarray, P: float, gamma: float) -> np.ndarray:
    denom = chi2.ppf(1.0 - gamma, df=nu)
    num = ncx2.ppf(P, df=1, nc=norm_lx_h ** 2)
    return np.sqrt(nu * num / denom)


# ============================================================
# Parametric bootstrap augmentation (core)
# ============================================================
def bootstrap_augment_z_normal(
    X_base: np.ndarray,
    mu_fn,
    var_fn,
    n_syn: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (X_syn, z_syn) under:
        z | X ~ Normal(mu(X), var(X))
    where X_syn is resampled from X_base (empirical covariate distribution).
    """
    n0 = X_base.shape[0]
    idx = rng.integers(0, n0, size=n_syn)
    X_syn = X_base[idx]

    mu = mu_fn(X_syn)
    var = np.maximum(var_fn(X_syn), 1e-12)
    z_syn = mu + rng.normal(size=n_syn) * np.sqrt(var)
    return X_syn, z_syn


# ============================================================
# Ours (multivariate) with optional bootstrap augmentation
# ============================================================
def run_ours(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    feature_cols: list[str], y_col: str,
    alpha: float, delta: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,   # e.g. 1.0 means add n_tr synthetic points
):
    """
    If bootstrap_mult>0:
      - split dfA_s into train/cal (50/50)
      - fit mean/var on train
      - generate synthetic points ONLY for the training part
      - refit mean/var on (train + synthetic), keep calibration untouched
      - compute lambda on calibration
      - evaluate on dfB
    """
    rng = np.random.default_rng(seed)

    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    X = dfA_s[feature_cols].to_numpy()
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

    # ---- 1) initial fit on training ----
    mean_pipe.fit(X_tr, z_tr)
    mu_tr = mean_pipe.predict(X_tr)
    res2_tr = (z_tr - mu_tr) ** 2
    var_pipe.fit(X_tr, res2_tr)

    # ---- 2) bootstrap augmentation on training only ----
    if bootstrap_mult and bootstrap_mult > 0:
        n_syn = int(round(bootstrap_mult * X_tr.shape[0]))

        def mu_fn(Xin):  # uses current mean_pipe
            return mean_pipe.predict(Xin)

        def var_fn(Xin):  # uses current var_pipe
            return np.maximum(var_pipe.predict(Xin), 1e-12)

        X_syn, z_syn = bootstrap_augment_z_normal(
            X_base=X_tr, mu_fn=mu_fn, var_fn=var_fn, n_syn=n_syn, rng=rng
        )

        # refit on augmented training
        X_tr_aug = np.vstack([X_tr, X_syn])
        z_tr_aug = np.concatenate([z_tr, z_syn])

        mean_pipe.fit(X_tr_aug, z_tr_aug)
        mu_tr_aug = mean_pipe.predict(X_tr_aug)
        res2_tr_aug = (z_tr_aug - mu_tr_aug) ** 2
        var_pipe.fit(X_tr_aug, res2_tr_aug)

    # ---- 3) calibration for lambda (NOT augmented) ----
    mu_cal = mean_pipe.predict(X_cal)
    var_cal = np.maximum(var_pipe.predict(X_cal), 1e-6)
    lam = find_lambda_fast(alpha, delta, z_cal, mu_cal, var_cal)

    # ---- 4) test on Happy B ----
    X_test = dfB[feature_cols].to_numpy()
    y_test = dfB[y_col].to_numpy()

    mu_test = mean_pipe.predict(X_test)
    var_test = np.maximum(var_pipe.predict(X_test), 1e-6)

    lower = np.maximum(itf(mu_test - lam * np.sqrt(var_test)), 0.0)
    upper = itf(mu_test + lam * np.sqrt(var_test))

    content = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mean_width = float(np.mean(upper - lower))

    return dict(method="Ours", lambda_=lam, content=content, mean_width=mean_width, bootstrap_mult=bootstrap_mult)


# ============================================================
# GY (1D) with optional parametric bootstrap augmentation
# ============================================================
def run_gy_1d_from_arrays(
    x: np.ndarray, y: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    alpha: float, delta: float,
):
    """
    Same as your run_gy_1d but uses provided arrays directly
    (no internal sampling).
    """
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

    norm_vals = compute_norm_fast(B, A, BtB)
    k = find_k_factor(nu=nu, norm_lx_h=norm_vals, P=1.0 - alpha, gamma=delta)

    # predictions on test
    mu_test = pspline_predict(mean_fit, x_test)
    var_test = np.maximum(pspline_predict(var_fit, x_test), 1e-6)

    B_test = t_fit["trans"].transform(x_test.reshape(-1, 1))
    t_hat_test = (B_test @ t_fit["beta"]).ravel()

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

    return dict(method="GY(1D)", lambda_=np.nan, content=content, mean_width=mean_width)


def run_gy_1d(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    alpha: float, delta: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,   # e.g. 1.0 => add n_sample synthetic points
):
    """
    If bootstrap_mult>0:
      - sample dfA_s of size n_sample
      - fit (mean_fit, var_fit) on dfA_s
      - generate synthetic samples by:
          x_syn ~ empirical x (resample x from dfA_s)
          z_syn | x_syn ~ Normal(mu_hat(x_syn), var_hat(x_syn))
          y_syn = expm1(z_syn) (clipped to >=0)
      - train GY on augmented (x_aug, y_aug)
      - evaluate on dfB (unchanged)
    """
    rng = np.random.default_rng(seed)

    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    x = dfA_s[x_col].to_numpy()
    y = dfA_s[y_col].to_numpy()

    # baseline fit to drive bootstrap generator (on z scale)
    z = tf(y)
    mean_fit = fit_pspline_1d(x, z, n_knots=12, degree=3)
    mu_hat = mean_fit["fitted"]
    res2 = (z - mu_hat) ** 2
    var_fit = fit_pspline_1d(x, res2, n_knots=12, degree=3)
    var_hat = np.maximum(var_fit["fitted"], 1e-6)

    if bootstrap_mult and bootstrap_mult > 0:
        n_syn = int(round(bootstrap_mult * len(x)))
        idx = rng.integers(0, len(x), size=n_syn)
        x_syn = x[idx]

        mu_syn = pspline_predict(mean_fit, x_syn)
        var_syn = np.maximum(pspline_predict(var_fit, x_syn), 1e-12)
        z_syn = mu_syn + rng.normal(size=n_syn) * np.sqrt(var_syn)
        y_syn = np.maximum(itf(z_syn), 0.0)

        x_aug = np.concatenate([x, x_syn])
        y_aug = np.concatenate([y, y_syn])
    else:
        x_aug, y_aug = x, y

    x_test = dfB[x_col].to_numpy()
    y_test = dfB[y_col].to_numpy()

    out = run_gy_1d_from_arrays(
        x=x_aug, y=y_aug,
        x_test=x_test, y_test=y_test,
        alpha=alpha, delta=delta
    )
    out["bootstrap_mult"] = bootstrap_mult
    return out


# ============================================================
# Experiments: compare with/without bootstrap augmentation
# ============================================================
def run_many_seeds_bootstrap(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    seeds=range(1, 51),
    n_sample: int = 5000,
    alpha: float = 0.10,
    delta: float = 0.05,
    bootstrap_mult: float = 0.0,   # set 0.5, 1.0, 2.0, ...
):
    rows = []
    for seed in seeds:
        # Ours(1D mag_r)
        r1 = run_ours(
            dfA, dfB,
            feature_cols=["mag_r"],
            y_col="z_spec",
            alpha=alpha, delta=delta,
            n_sample=n_sample, seed=seed,
            bootstrap_mult=bootstrap_mult,
        )
        r1["method"] = "Ours(1D mag_r)"
        r1["seed"] = seed

        # GY(1D mag_r)
        r2 = run_gy_1d(
            dfA, dfB,
            x_col="mag_r", y_col="z_spec",
            alpha=alpha, delta=delta,
            n_sample=n_sample, seed=seed,
            bootstrap_mult=bootstrap_mult,
        )
        r2["method"] = "GY(1D mag_r)"
        r2["seed"] = seed

        rows.extend([r1, r2])

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    dfA = load_happy(root / "data" / "Happy" / "happy_A")
    dfB = load_happy(root / "data" / "Happy" / "happy_B")

    alpha = 0.10
    delta = 0.05
    n_sample = 5000
    seeds = range(1, 51)

    # 1) no augmentation
    df0 = run_many_seeds_bootstrap(
        dfA, dfB, seeds=seeds,
        n_sample=n_sample, alpha=alpha, delta=delta,
        bootstrap_mult=0.0
    )
    df0["setting"] = "no_bootstrap"

    # 2) with augmentation (example: add as many synthetic points as training size for Ours;
    #    add as many synthetic points as n_sample for GY)
    #    Try 0.5, 1.0, 2.0 etc.
    df1 = run_many_seeds_bootstrap(
        dfA, dfB, seeds=seeds,
        n_sample=n_sample, alpha=alpha, delta=delta,
        bootstrap_mult=1.0
    )
    df1["setting"] = "bootstrap_mult=1.0"

    df_all = pd.concat([df0, df1], ignore_index=True)

    # summary
    summary = (
        df_all
        .groupby(["setting", "method"])[["content", "mean_width"]]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    print(summary)

    out_csv = f"results_happy_1d_bootstrap_mult1.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
