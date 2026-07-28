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


# ============================================================
# I/O: Happy loader
# ============================================================
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


# ============================================================
# Transform: run methods on z = log(1 + y), report intervals on y-scale
# ============================================================
def tf(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))


def itf(z: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(z, dtype=float))


# ============================================================
# Common PAC calibration helpers
# ============================================================
def pac_lambda(pac_alpha: float, n_cal: int) -> float:
    return float(np.sqrt(np.log(2.0 / pac_alpha) / (2.0 * n_cal)))


def adjusted_quantile(scores: np.ndarray, content_level: float, pac_alpha: float) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), float("nan"), float("nan")
    lam = pac_lambda(pac_alpha=pac_alpha, n_cal=scores.size)
    q_level = content_level + lam
    if q_level >= 1.0:
        # The formal guarantee assumes content_level + lambda < 1.
        # For numerical runs, use the maximum calibration score.
        q_level_used = 1.0
    else:
        q_level_used = q_level
    return float(np.quantile(scores, q_level_used)), float(lam), float(q_level_used)


def evaluate_interval(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    finite_width = width[np.isfinite(width)]
    return dict(
        content=float(np.mean((y >= lower) & (y <= upper))),
        mean_width=float(np.mean(finite_width)) if finite_width.size else float("inf"),
        median_width=float(np.median(finite_width)) if finite_width.size else float("inf"),
        q90_width=float(np.quantile(finite_width, 0.90)) if finite_width.size else float("inf"),
    )


def make_mean_var_pipes(seed: int):
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
    return mean_pipe, var_pipe


def fit_mean_var(X_tr: np.ndarray, z_tr: np.ndarray, seed: int, bootstrap_mult: float = 0.0):
    rng = np.random.default_rng(seed)
    mean_pipe, var_pipe = make_mean_var_pipes(seed)

    mean_pipe.fit(X_tr, z_tr)
    mu_tr = mean_pipe.predict(X_tr)
    res2_tr = (z_tr - mu_tr) ** 2
    eps = max(1e-10, 1e-3 * float(np.median(res2_tr[np.isfinite(res2_tr)]))) if np.any(np.isfinite(res2_tr)) else 1e-6
    var_pipe.fit(X_tr, np.log(res2_tr + eps))

    if bootstrap_mult and bootstrap_mult > 0:
        n_syn = int(round(bootstrap_mult * X_tr.shape[0]))
        idx = rng.integers(0, X_tr.shape[0], size=n_syn)
        X_syn = X_tr[idx]
        mu_syn = mean_pipe.predict(X_syn)
        var_syn = np.maximum(np.exp(var_pipe.predict(X_syn)), eps)
        z_syn = mu_syn + rng.normal(size=n_syn) * np.sqrt(var_syn)
        X_aug = np.vstack([X_tr, X_syn])
        z_aug = np.concatenate([z_tr, z_syn])

        mean_pipe, var_pipe = make_mean_var_pipes(seed)
        mean_pipe.fit(X_aug, z_aug)
        mu_aug = mean_pipe.predict(X_aug)
        res2_aug = (z_aug - mu_aug) ** 2
        eps = max(1e-10, 1e-3 * float(np.median(res2_aug[np.isfinite(res2_aug)]))) if np.any(np.isfinite(res2_aug)) else eps
        var_pipe.fit(X_aug, np.log(res2_aug + eps))

    return mean_pipe, var_pipe, eps


def predict_mean_var(mean_pipe, var_pipe, eps: float, X: np.ndarray):
    mu = mean_pipe.predict(X)
    var = np.maximum(np.exp(var_pipe.predict(X)), eps)
    sd = np.sqrt(var)
    return mu, var, sd


def prepare_happy_split(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_sample: int,
    seed: int,
):
    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    X = dfA_s[[x_col]].to_numpy()
    y = dfA_s[y_col].to_numpy(dtype=float)
    z = tf(y)
    X_tr, X_cal, z_tr, z_cal = train_test_split(
        X, z, test_size=0.5, random_state=seed, shuffle=True
    )
    X_te = dfB[[x_col]].to_numpy()
    y_te = dfB[y_col].to_numpy(dtype=float)
    return X_tr, X_cal, z_tr, z_cal, X_te, y_te


# ============================================================
# PAC-calibrated methods
# ============================================================
def run_srti(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    content_level: float, pac_alpha: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,
):
    X_tr, X_cal, z_tr, z_cal, X_te, y_te = prepare_happy_split(dfA, dfB, x_col, y_col, n_sample, seed)
    mean_pipe, var_pipe, eps = fit_mean_var(X_tr, z_tr, seed, bootstrap_mult)
    mu_cal, _, sd_cal = predict_mean_var(mean_pipe, var_pipe, eps, X_cal)
    scores = np.abs(z_cal - mu_cal) / sd_cal
    qhat, lam, q_level = adjusted_quantile(scores, content_level, pac_alpha)

    mu_te, _, sd_te = predict_mean_var(mean_pipe, var_pipe, eps, X_te)
    lower_z = mu_te - qhat * sd_te
    upper_z = mu_te + qhat * sd_te
    lower = np.maximum(itf(lower_z), 0.0)
    upper = itf(upper_z)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="SR-TI", lambda_=lam, qhat=qhat, q_level=q_level, bootstrap_mult=bootstrap_mult)
    return out


def run_asrti(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    content_level: float, pac_alpha: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,
):
    X_tr, X_cal, z_tr, z_cal, X_te, y_te = prepare_happy_split(dfA, dfB, x_col, y_col, n_sample, seed)
    mean_pipe, var_pipe, eps = fit_mean_var(X_tr, z_tr, seed, bootstrap_mult)

    mu_tr, _, sd_tr = predict_mean_var(mean_pipe, var_pipe, eps, X_tr)
    zstd_tr = (z_tr - mu_tr) / sd_tr
    tau = (1.0 - content_level) / 2.0
    a_minus = abs(float(np.quantile(zstd_tr[np.isfinite(zstd_tr)], tau)))
    a_plus = float(np.quantile(zstd_tr[np.isfinite(zstd_tr)], 1.0 - tau))
    a_minus = max(a_minus, 1e-8)
    a_plus = max(a_plus, 1e-8)

    mu_cal, _, sd_cal = predict_mean_var(mean_pipe, var_pipe, eps, X_cal)
    zstd_cal = (z_cal - mu_cal) / sd_cal
    scores = np.maximum(-zstd_cal / a_minus, zstd_cal / a_plus)
    qhat, lam, q_level = adjusted_quantile(scores, content_level, pac_alpha)

    mu_te, _, sd_te = predict_mean_var(mean_pipe, var_pipe, eps, X_te)
    lower_z = mu_te - qhat * a_minus * sd_te
    upper_z = mu_te + qhat * a_plus * sd_te
    lower = np.maximum(itf(lower_z), 0.0)
    upper = itf(upper_z)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="ASR-TI", lambda_=lam, qhat=qhat, q_level=q_level,
               a_minus=a_minus, a_plus=a_plus, bootstrap_mult=bootstrap_mult)
    return out


def run_cqr_ti(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    content_level: float, pac_alpha: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,
):
    X_tr, X_cal, z_tr, z_cal, X_te, y_te = prepare_happy_split(dfA, dfB, x_col, y_col, n_sample, seed)

    # Optional synthetic augmentation using the same mean/variance generator as SR-TI.
    if bootstrap_mult and bootstrap_mult > 0:
        rng = np.random.default_rng(seed)
        mean_pipe, var_pipe, eps = fit_mean_var(X_tr, z_tr, seed, bootstrap_mult=0.0)
        n_syn = int(round(bootstrap_mult * X_tr.shape[0]))
        idx = rng.integers(0, X_tr.shape[0], size=n_syn)
        X_syn = X_tr[idx]
        mu_syn, _, sd_syn = predict_mean_var(mean_pipe, var_pipe, eps, X_syn)
        z_syn = mu_syn + rng.normal(size=n_syn) * sd_syn
        X_tr = np.vstack([X_tr, X_syn])
        z_tr = np.concatenate([z_tr, z_syn])

    alpha_lo = (1.0 - content_level) / 2.0
    alpha_hi = (1.0 + content_level) / 2.0
    qlo = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", GradientBoostingRegressor(loss="quantile", alpha=alpha_lo, random_state=seed)),
    ])
    qhi = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", GradientBoostingRegressor(loss="quantile", alpha=alpha_hi, random_state=seed + 1)),
    ])
    qlo.fit(X_tr, z_tr)
    qhi.fit(X_tr, z_tr)

    lo_cal = qlo.predict(X_cal)
    hi_cal = qhi.predict(X_cal)
    scores = np.maximum(lo_cal - z_cal, z_cal - hi_cal)
    qhat, lam, q_level = adjusted_quantile(scores, content_level, pac_alpha)

    lo_te = qlo.predict(X_te) - qhat
    hi_te = qhi.predict(X_te) + qhat
    lower = np.maximum(itf(lo_te), 0.0)
    upper = itf(hi_te)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="CQR-TI", lambda_=lam, qhat=qhat, q_level=q_level, bootstrap_mult=bootstrap_mult)
    return out


# ============================================================
# Parametric TI benchmark: stabilized heteroscedastic P-spline TI
# ============================================================
def second_diff_penalty(m: int) -> np.ndarray:
    I = np.eye(m)
    return np.diff(I, n=2, axis=0)


def fit_pspline_1d(x: np.ndarray, y: np.ndarray, n_knots: int = 12, degree: int = 3,
                   lam_grid: np.ndarray | None = None):
    from sklearn.preprocessing import SplineTransformer

    x = x.reshape(-1, 1)
    n = x.shape[0]
    trans = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=True)
    B = trans.fit_transform(x)
    m = B.shape[1]
    BtB = B.T @ B
    D2 = second_diff_penalty(m)
    P = D2.T @ D2
    if lam_grid is None:
        lam_grid = np.logspace(-6, 6, 60)
    best = None
    y2 = y.reshape(-1, 1)
    for lam in lam_grid:
        A = inv(BtB + lam * P)
        beta = A @ (B.T @ y2)
        fitted = (B @ beta).ravel()
        resid = y2.ravel() - fitted
        df_eff = float(np.trace(A @ BtB))
        denom = max(1e-8, (n - df_eff) ** 2)
        gcv = (n * np.sum(resid ** 2)) / denom
        if best is None or gcv < best["gcv"]:
            best = dict(gcv=float(gcv), lam=float(lam), trans=trans, B=B, BtB=BtB,
                        A=A, beta=beta, fitted=fitted)
    return best


def pspline_predict(fit, x_new: np.ndarray) -> np.ndarray:
    B_new = fit["trans"].transform(x_new.reshape(-1, 1))
    return (B_new @ fit["beta"]).ravel()


def find_k_factor(nu: float, norm_lx_h: np.ndarray, P: float, gamma: float) -> np.ndarray:
    denom = chi2.ppf(1.0 - gamma, df=nu)
    num = ncx2.ppf(P, df=1, nc=norm_lx_h ** 2)
    return np.sqrt(nu * num / denom)


def k_on_new_x(x_new: np.ndarray, fit, nu: float, P: float, gamma: float) -> np.ndarray:
    B_new = fit["trans"].transform(x_new.reshape(-1, 1))
    A = fit["A"]
    BtB = fit["BtB"]
    M = A @ BtB @ A
    norm_new = np.sqrt(np.sum((B_new @ M) * B_new, axis=1))
    return find_k_factor(nu=nu, norm_lx_h=norm_new, P=P, gamma=gamma)


def run_parametric_ti(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    x_col: str, y_col: str,
    content_level: float, pac_alpha: float,
    n_sample: int, seed: int,
    bootstrap_mult: float = 0.0,
):
    rng = np.random.default_rng(seed)
    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    x = dfA_s[x_col].to_numpy(dtype=float)
    y = dfA_s[y_col].to_numpy(dtype=float)
    z = tf(y)

    if bootstrap_mult and bootstrap_mult > 0:
        mean_fit0 = fit_pspline_1d(x, z, n_knots=12, degree=3)
        mu0 = mean_fit0["fitted"]
        res2_0 = (z - mu0) ** 2
        var_fit0 = fit_pspline_1d(x, np.log(res2_0 + 1e-8), n_knots=12, degree=3)
        n_syn = int(round(bootstrap_mult * len(x)))
        idx = rng.integers(0, len(x), size=n_syn)
        x_syn = x[idx]
        mu_syn = pspline_predict(mean_fit0, x_syn)
        var_syn = np.maximum(np.exp(pspline_predict(var_fit0, x_syn)), 1e-8)
        z_syn = mu_syn + rng.normal(size=n_syn) * np.sqrt(var_syn)
        y_syn = np.maximum(itf(z_syn), 0.0)
        x = np.concatenate([x, x_syn])
        y = np.concatenate([y, y_syn])
        z = tf(y)

    mean_fit = fit_pspline_1d(x, z, n_knots=12, degree=3)
    mu = mean_fit["fitted"]
    res2 = (z - mu) ** 2
    floor = max(float(np.quantile(res2[np.isfinite(res2)], 0.05)), 1e-10)
    var_fit = fit_pspline_1d(x, np.log(res2 + 1e-8), n_knots=12, degree=3)
    var_hat = np.maximum(np.exp(var_fit["fitted"]), floor)
    t = (z - mu) / np.sqrt(var_hat)

    t_fit = fit_pspline_1d(x, t, n_knots=12, degree=3)
    t_hat = t_fit["fitted"]
    n = len(x)
    nu = max(1.0, n - 1.0)
    est_var = max(float(np.sum((t - t_hat) ** 2) / nu), 1e-12)

    x_te = dfB[x_col].to_numpy(dtype=float)
    y_te = dfB[y_col].to_numpy(dtype=float)
    mu_te = pspline_predict(mean_fit, x_te)
    var_te = np.maximum(np.exp(pspline_predict(var_fit, x_te)), floor)
    t_hat_te = pspline_predict(t_fit, x_te)
    k_te = k_on_new_x(x_te, t_fit, nu=nu, P=content_level, gamma=pac_alpha)

    upper_t = t_hat_te + np.sqrt(est_var) * k_te
    lower_t = t_hat_te - np.sqrt(est_var) * k_te
    upper_z = mu_te + upper_t * np.sqrt(var_te)
    lower_z = mu_te + lower_t * np.sqrt(var_te)
    upper = itf(upper_z)
    lower = np.maximum(itf(lower_z), 0.0)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="Parametric TI", lambda_=np.nan, qhat=np.nan, q_level=np.nan,
               bootstrap_mult=bootstrap_mult)
    return out


# ============================================================
# Runner
# ============================================================
def run_many_seeds_4way(
    dfA: pd.DataFrame, dfB: pd.DataFrame,
    seeds=range(1, 51),
    n_sample: int = 5000,
    content_level: float = 0.90,
    pac_alpha: float = 0.05,
    bootstrap_mult: float = 0.0,
    x_col: str = "mag_r",
    y_col: str = "z_spec",
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for fn in [run_parametric_ti, run_srti, run_asrti, run_cqr_ti]:
            r = fn(dfA, dfB, x_col, y_col, content_level, pac_alpha, n_sample, seed, bootstrap_mult)
            r.update(
            dataset="redshift",
            seed=seed,
            n_train=n_sample // 2,
            n_cal=n_sample - n_sample // 2,
            n_test=len(dfB),
            content_level=content_level,
            pac_alpha=pac_alpha,
            )   
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root containing data/real/redshift/happy_A and happy_B.",
    )
    ap.add_argument("--happy_a", type=str, default=None)
    ap.add_argument("--happy_b", type=str, default=None)
    ap.add_argument("--x_col", type=str, default="mag_r")
    ap.add_argument("--y_col", type=str, default="z_spec")
    ap.add_argument("--n_sample", type=int, default=5000)
    ap.add_argument("--content_level", type=float, default=0.90)
    ap.add_argument("--pac_alpha", type=float, default=0.05)
    ap.add_argument("--seed_from", type=int, default=1)
    ap.add_argument("--seed_to", type=int, default=50)
    ap.add_argument("--bootstrap_mult", type=float, default=0.0)
    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to results/real/redshift/results_redshift_4methods.csv.",
    )
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).resolve()
    else:
        # scripts/real/redshift/real_redshift_4methods.py 기준으로 ti_project/
        root = Path(__file__).resolve().parents[3]

    if args.happy_a is not None and args.happy_b is not None:
        path_a = Path(args.happy_a).resolve()
        path_b = Path(args.happy_b).resolve()
    else:
        path_a = root / "data" / "real" / "redshift" / "happy_A"
        path_b = root / "data" / "real" / "redshift" / "happy_B"

    if args.out_csv is None:
        out_csv = root / "results" / "real" / "redshift" / "results_redshift_4methods.csv"
    else:
        out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dfA = load_happy(path_a)
    dfB = load_happy(path_b)
    seeds = range(args.seed_from, args.seed_to + 1)
    df_all = run_many_seeds_4way(
        dfA, dfB, seeds=seeds, n_sample=args.n_sample,
        content_level=args.content_level, pac_alpha=args.pac_alpha,
        bootstrap_mult=args.bootstrap_mult, x_col=args.x_col, y_col=args.y_col,
    )

    summary = (df_all.groupby("method")[["content", "mean_width", "median_width", "q90_width"]]
               .agg(["mean", "std", "median"]).round(4))
    print("\n=== Happy summary ===")
    print(summary)
    
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
