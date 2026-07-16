from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

from scipy.stats import chi2, ncx2
from numpy.linalg import inv


DEFAULT_TSA_URL = "https://raw.githubusercontent.com/hunj/tsa-passenger-throughput/main/output.csv"


# ============================================================
# Data and transformations
# ============================================================
def load_tsa(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url, header=None, names=["date", "throughput"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")
    df = df.dropna(subset=["date", "throughput"]).sort_values("date").reset_index(drop=True)
    return df


def make_transform(tf_mode: str, scale_c: float):
    tf_mode = tf_mode.lower()
    if tf_mode not in ("asinh", "log_scaled"):
        raise ValueError(f"Unknown tf_mode={tf_mode}")

    def tf_fn(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if tf_mode == "log_scaled":
            return np.log1p(y / scale_c)
        return np.arcsinh(y / scale_c)

    def itf_fn(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if tf_mode == "log_scaled":
            return scale_c * np.expm1(z)
        return scale_c * np.sinh(z)

    return tf_fn, itf_fn


def make_time_features(dates: pd.Series, d0: pd.Timestamp) -> np.ndarray:
    t = (dates - d0).dt.days.to_numpy()
    dow = dates.dt.dayofweek.to_numpy()
    month = dates.dt.month.to_numpy()
    sin7 = np.sin(2 * np.pi * t / 7.0)
    cos7 = np.cos(2 * np.pi * t / 7.0)
    sin365 = np.sin(2 * np.pi * t / 365.25)
    cos365 = np.cos(2 * np.pi * t / 365.25)
    return np.column_stack([t, dow, month, sin7, cos7, sin365, cos365])


# ============================================================
# Split helpers
# ============================================================
@dataclass(frozen=True)
class TSASplit:
    df_train: pd.DataFrame
    df_cal: pd.DataFrame
    df_test: pd.DataFrame
    split_mode: str


def split_random(df: pd.DataFrame, seed: int, target_sizes: tuple[int, int, int] = (365, 365, 360)):
    rng = np.random.default_rng(seed)
    n_tr, n_cal, n_te = target_sizes
    n = len(df)
    if n_tr + n_cal + n_te > n:
        raise ValueError("target_sizes exceed dataset size")
    idx = np.arange(n)
    rng.shuffle(idx)
    tr_idx = idx[:n_tr]
    cal_idx = idx[n_tr:n_tr + n_cal]
    te_idx = idx[n_tr + n_cal:n_tr + n_cal + n_te]
    return (
        df.iloc[tr_idx].reset_index(drop=True),
        df.iloc[cal_idx].reset_index(drop=True),
        df.iloc[te_idx].reset_index(drop=True),
    )


def split_time(df: pd.DataFrame, a_end: str, b_start: str, train_frac_A: float, seed: int):
    a_end = pd.to_datetime(a_end)
    b_start = pd.to_datetime(b_start)
    dfA = df[df["date"] <= a_end].copy()
    dfB = df[df["date"] >= b_start].copy()
    rng = np.random.default_rng(seed)
    idxA = np.arange(len(dfA))
    rng.shuffle(idxA)
    nA = len(dfA)
    n_tr = int(np.floor(train_frac_A * nA))
    n_tr = max(10, min(n_tr, nA - 10))
    tr_idx = idxA[:n_tr]
    cal_idx = idxA[n_tr:]
    return dfA.iloc[tr_idx].reset_index(drop=True), dfA.iloc[cal_idx].reset_index(drop=True), dfB.reset_index(drop=True)


# ============================================================
# PAC calibration helpers
# ============================================================
def pac_lambda(pac_alpha: float, n_cal: int) -> float:
    return float(np.sqrt(np.log(2.0 / pac_alpha) / (2.0 * n_cal)))


def adjusted_quantile(scores: np.ndarray, content_level: float, pac_alpha: float) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), float("nan"), float("nan")
    lam = pac_lambda(pac_alpha, len(scores))
    q_level = content_level + lam
    q_level_used = 1.0 if q_level >= 1.0 else q_level
    return float(np.quantile(scores, q_level_used)), float(lam), float(q_level_used)


def evaluate_interval(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    finite = width[np.isfinite(width)]
    return dict(
        content=float(np.mean((y >= lower) & (y <= upper))),
        mean_width=float(np.mean(finite)) if finite.size else float("inf"),
        median_width=float(np.median(finite)) if finite.size else float("inf"),
        q90_width=float(np.quantile(finite, 0.90)) if finite.size else float("inf"),
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


def fit_mean_var(X_tr: np.ndarray, z_tr: np.ndarray, seed: int):
    mean_pipe, var_pipe = make_mean_var_pipes(seed)
    mean_pipe.fit(X_tr, z_tr)
    mu_tr = mean_pipe.predict(X_tr)
    res2_tr = (z_tr - mu_tr) ** 2
    finite = res2_tr[np.isfinite(res2_tr)]
    eps = max(1e-10, 1e-3 * float(np.median(finite))) if finite.size else 1e-6
    var_pipe.fit(X_tr, np.log(res2_tr + eps))
    return mean_pipe, var_pipe, eps


def predict_mean_var(mean_pipe, var_pipe, eps: float, X: np.ndarray):
    mu = mean_pipe.predict(X)
    var = np.maximum(np.exp(var_pipe.predict(X)), eps)
    return mu, var, np.sqrt(var)


def prepare_tsa_arrays(df_train: pd.DataFrame, df_cal: pd.DataFrame, df_test: pd.DataFrame, tf_fn):
    d0 = df_train["date"].min()
    X_tr = make_time_features(df_train["date"], d0)
    X_cal = make_time_features(df_cal["date"], d0)
    X_te = make_time_features(df_test["date"], d0)
    y_tr = df_train["throughput"].to_numpy(dtype=float)
    y_cal = df_cal["throughput"].to_numpy(dtype=float)
    y_te = df_test["throughput"].to_numpy(dtype=float)
    z_tr = tf_fn(y_tr)
    z_cal = tf_fn(y_cal)
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te, z_tr, z_cal


# ============================================================
# PAC-calibrated methods
# ============================================================
def run_hcti(df_train: pd.DataFrame, df_cal: pd.DataFrame, df_test: pd.DataFrame,
             content_level: float, pac_alpha: float, seed: int, tf_fn, itf_fn):
    X_tr, X_cal, X_te, _, _, y_te, z_tr, z_cal = prepare_tsa_arrays(df_train, df_cal, df_test, tf_fn)
    mean_pipe, var_pipe, eps = fit_mean_var(X_tr, z_tr, seed)
    mu_cal, _, sd_cal = predict_mean_var(mean_pipe, var_pipe, eps, X_cal)
    scores = np.abs(z_cal - mu_cal) / sd_cal
    qhat, lam, q_level = adjusted_quantile(scores, content_level, pac_alpha)
    mu_te, _, sd_te = predict_mean_var(mean_pipe, var_pipe, eps, X_te)
    lower = np.maximum(itf_fn(mu_te - qhat * sd_te), 0.0)
    upper = itf_fn(mu_te + qhat * sd_te)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="HCTI", lambda_=lam, qhat=qhat, q_level=q_level, lower=lower, upper=upper)
    return out


def run_hcti_asym(df_train: pd.DataFrame, df_cal: pd.DataFrame, df_test: pd.DataFrame,
                  content_level: float, pac_alpha: float, seed: int, tf_fn, itf_fn):
    X_tr, X_cal, X_te, _, _, y_te, z_tr, z_cal = prepare_tsa_arrays(df_train, df_cal, df_test, tf_fn)
    mean_pipe, var_pipe, eps = fit_mean_var(X_tr, z_tr, seed)
    mu_tr, _, sd_tr = predict_mean_var(mean_pipe, var_pipe, eps, X_tr)
    zstd_tr = (z_tr - mu_tr) / sd_tr
    zstd_tr = zstd_tr[np.isfinite(zstd_tr)]
    tau = (1.0 - content_level) / 2.0
    a_minus = max(abs(float(np.quantile(zstd_tr, tau))), 1e-8)
    a_plus = max(float(np.quantile(zstd_tr, 1.0 - tau)), 1e-8)

    mu_cal, _, sd_cal = predict_mean_var(mean_pipe, var_pipe, eps, X_cal)
    zstd_cal = (z_cal - mu_cal) / sd_cal
    scores = np.maximum(-zstd_cal / a_minus, zstd_cal / a_plus)
    qhat, lam, q_level = adjusted_quantile(scores, content_level, pac_alpha)

    mu_te, _, sd_te = predict_mean_var(mean_pipe, var_pipe, eps, X_te)
    lower = np.maximum(itf_fn(mu_te - qhat * a_minus * sd_te), 0.0)
    upper = itf_fn(mu_te + qhat * a_plus * sd_te)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="HCTI-asym", lambda_=lam, qhat=qhat, q_level=q_level,
               a_minus=a_minus, a_plus=a_plus, lower=lower, upper=upper)
    return out


def run_cqr_ti(df_train: pd.DataFrame, df_cal: pd.DataFrame, df_test: pd.DataFrame,
               content_level: float, pac_alpha: float, seed: int, tf_fn, itf_fn):
    X_tr, X_cal, X_te, _, _, y_te, z_tr, z_cal = prepare_tsa_arrays(df_train, df_cal, df_test, tf_fn)
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
    lower = np.maximum(itf_fn(qlo.predict(X_te) - qhat), 0.0)
    upper = itf_fn(qhi.predict(X_te) + qhat)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="CQR-TI", lambda_=lam, qhat=qhat, q_level=q_level, lower=lower, upper=upper)
    return out


# ============================================================
# Parametric TI benchmark: stabilized heteroscedastic P-spline TI
# ============================================================
def second_diff_penalty(m: int) -> np.ndarray:
    I = np.eye(m)
    return np.diff(I, n=2, axis=0)


def fit_pspline_1d(x: np.ndarray, y: np.ndarray, n_knots: int = 12, degree: int = 3,
                   lam_grid: np.ndarray | None = None):
    x = x.reshape(-1, 1)
    trans = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=True)
    B = trans.fit_transform(x)
    n, m = B.shape
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
    return find_k_factor(nu, norm_new, P, gamma)


def run_parametric_ti(df_train: pd.DataFrame, df_cal: pd.DataFrame, df_test: pd.DataFrame,
                      content_level: float, pac_alpha: float, seed: int, tf_fn, itf_fn):
    # df_cal is not used by the parametric benchmark, but kept for a common interface.
    d0 = df_train["date"].min()
    x_tr = (df_train["date"] - d0).dt.days.to_numpy(dtype=float)
    x_te = (df_test["date"] - d0).dt.days.to_numpy(dtype=float)
    y_tr = df_train["throughput"].to_numpy(dtype=float)
    y_te = df_test["throughput"].to_numpy(dtype=float)
    z_tr = tf_fn(y_tr)

    mean_fit = fit_pspline_1d(x_tr, z_tr, n_knots=12, degree=3)
    mu_tr = mean_fit["fitted"]
    res2 = (z_tr - mu_tr) ** 2
    floor = max(float(np.quantile(res2[np.isfinite(res2)], 0.05)), 1e-10)
    var_fit = fit_pspline_1d(x_tr, np.log(res2 + 1e-8), n_knots=12, degree=3)
    var_hat = np.maximum(np.exp(var_fit["fitted"]), floor)
    t = (z_tr - mu_tr) / np.sqrt(var_hat)

    t_fit = fit_pspline_1d(x_tr, t, n_knots=12, degree=3)
    t_hat = t_fit["fitted"]
    nu = max(1.0, len(x_tr) - 1.0)
    est_var = max(float(np.sum((t - t_hat) ** 2) / nu), 1e-12)

    mu_te = pspline_predict(mean_fit, x_te)
    var_te = np.maximum(np.exp(pspline_predict(var_fit, x_te)), floor)
    t_hat_te = pspline_predict(t_fit, x_te)
    k_te = k_on_new_x(x_te, t_fit, nu=nu, P=content_level, gamma=pac_alpha)
    upper_t = t_hat_te + np.sqrt(est_var) * k_te
    lower_t = t_hat_te - np.sqrt(est_var) * k_te
    upper_z = mu_te + upper_t * np.sqrt(var_te)
    lower_z = mu_te + lower_t * np.sqrt(var_te)
    upper = itf_fn(upper_z)
    lower = np.maximum(itf_fn(lower_z), 0.0)
    out = evaluate_interval(y_te, lower, upper)
    out.update(method="Parametric TI", lambda_=np.nan, qhat=np.nan, q_level=np.nan, lower=lower, upper=upper)
    return out


# ============================================================
# Runner
# ============================================================
def run_many_seeds(
    df: pd.DataFrame,
    seeds=range(1, 51),
    content_level: float = 0.90,
    pac_alpha: float = 0.05,
    tf_mode: str = "asinh",
    scale_c: float = 1e6,
    split_mode: str = "random",
    target_sizes: tuple[int, int, int] = (365, 365, 360),
    a_end: str = "2020-12-31",
    b_start: str = "2021-01-01",
    train_frac_A: float = 0.5,
):
    rows = []
    interval_rows = []
    tf_fn, itf_fn = make_transform(tf_mode=tf_mode, scale_c=scale_c)

    for seed in seeds:
        if split_mode == "random":
            df_tr, df_cal, df_te = split_random(df, seed=seed, target_sizes=target_sizes)
        else:
            df_tr, df_cal, df_te = split_time(df, a_end=a_end, b_start=b_start,
                                              train_frac_A=train_frac_A, seed=seed)

        for fn in [run_parametric_ti, run_hcti, run_hcti_asym, run_cqr_ti]:
            r = fn(df_tr, df_cal, df_te, content_level, pac_alpha, seed, tf_fn, itf_fn)
            lower = np.asarray(r.pop("lower"), dtype=float)
            upper = np.asarray(r.pop("upper"), dtype=float)
            r.update(dataset="TSA", seed=seed, split_mode=split_mode, tf_mode=tf_mode, scale_c=scale_c,
                     n_train=len(df_tr), n_cal=len(df_cal), n_test=len(df_te),
                     content_level=content_level, pac_alpha=pac_alpha)
            rows.append(r)

            tmp = pd.DataFrame({
                "dataset": "TSA",
                "seed": seed,
                "method": r["method"],
                "split_mode": split_mode,
                "tf_mode": tf_mode,
                "scale_c": scale_c,
                "date": df_te["date"].astype(str).to_numpy(),
                "y": df_te["throughput"].to_numpy(dtype=float),
                "lower": lower,
                "upper": upper,
            })
            interval_rows.append(tmp)

    return pd.DataFrame(rows), pd.concat(interval_rows, ignore_index=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DEFAULT_TSA_URL)
    ap.add_argument("--content_level", type=float, default=0.90)
    ap.add_argument("--pac_alpha", type=float, default=0.05)
    ap.add_argument("--seed_from", type=int, default=1)
    ap.add_argument("--seed_to", type=int, default=50)
    ap.add_argument("--tf_mode", type=str, default="asinh", choices=["asinh", "log_scaled"])
    ap.add_argument("--scale_c", type=float, default=1e6)
    ap.add_argument("--split_mode", type=str, default="random", choices=["random", "time"])
    ap.add_argument("--a_end", type=str, default="2020-12-31")
    ap.add_argument("--b_start", type=str, default="2021-01-01")
    ap.add_argument("--train_frac_A", type=float, default=0.5)
    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to results/real/tsa/results_tsa_4methods.csv.",
    )
    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root. Defaults to three levels above this script.",
    )
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).resolve()
    else:
        # scripts/real/tsa/real_tsa_4methods.py 기준 ti_project/
        root = Path(__file__).resolve().parents[3]

    if args.out_csv is None:
        out_csv = root / "results" / "real" / "tsa" / "results_tsa_4methods.csv"
    else:
        out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = load_tsa(args.data)
    seeds = range(args.seed_from, args.seed_to + 1)
    df_res, df_intervals = run_many_seeds(
        df, seeds=seeds, content_level=args.content_level, pac_alpha=args.pac_alpha,
        tf_mode=args.tf_mode, scale_c=args.scale_c, split_mode=args.split_mode,
        target_sizes=(365, 365, 360), a_end=args.a_end, b_start=args.b_start,
        train_frac_A=args.train_frac_A,
    )
    summary = (df_res.groupby("method")[["content", "mean_width", "median_width", "q90_width"]]
               .agg(["mean", "std", "median"]).round(4))
    print("\n=== TSA summary ===")
    print(summary)

    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved summary: {out_csv}")

    interval_csv = out_csv.with_name(out_csv.stem + "_intervals.csv")
    df_intervals.to_csv(interval_csv, index=False)
    print(f"Saved intervals: {interval_csv}")


if __name__ == "__main__":
    main()
