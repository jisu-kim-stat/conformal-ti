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

# =========================
# Utils
# =========================
from pathlib import Path
import pandas as pd
import numpy as np


def save_intervals_long_csv(
    df_test: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    out_path: str | Path,
    method: str,
    seed: int,
    split_mode: str,
    x_col: str = "day_idx",
    y_col: str = "throughput",
    date_col: str = "date",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "seed": np.full(len(df_test), seed, dtype=int),
        "method": np.full(len(df_test), method, dtype=object),
        "split_mode": np.full(len(df_test), split_mode, dtype=object),
        "x": df_test[x_col].to_numpy() if x_col in df_test.columns else np.arange(len(df_test)),
        "y": df_test[y_col].to_numpy(),
        "lower": np.asarray(lower, dtype=float),
        "upper": np.asarray(upper, dtype=float),
    }
    if date_col in df_test.columns:
        out["date"] = df_test[date_col].astype(str).to_numpy()

    df_out = pd.DataFrame(out)
    df_out.to_csv(out_path, index=False)


# =========================
# Data loading
# =========================
DEFAULT_TSA_URL = "https://raw.githubusercontent.com/hunj/tsa-passenger-throughput/main/output.csv"

def load_tsa(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url, header=None, names=["date", "throughput"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")
    df = df.dropna(subset=["date", "throughput"]).sort_values("date").reset_index(drop=True)
    return df



# ----------------------------
# transforms (choose one)
# ----------------------------
def make_transform(tf_mode: str, scale_c: float):
    tf_mode = tf_mode.lower()
    if tf_mode not in ("asinh", "log_scaled"):
        raise ValueError(f"Unknown tf_mode={tf_mode}")

    def tf_fn(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if tf_mode == "log_scaled":
            return np.log1p(y / scale_c)
        else:  # asinh
            return np.arcsinh(y / scale_c)

    def itf_fn(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if tf_mode == "log_scaled":
            return scale_c * np.expm1(z)
        else:  # asinh
            return scale_c * np.sinh(z)

    return tf_fn, itf_fn

tf_fn, itf_fn = make_transform(tf_mode="asinh", scale_c=1e6)


# =========================
# Hoeffding lambda (fast quantile form)
# =========================
def find_lambda_fast(alpha: float, delta: float,
                     z_cal: np.ndarray, mu_cal: np.ndarray, var_cal: np.ndarray) -> float:
    n_cal = len(z_cal)
    threshold = max(0.0, alpha - np.sqrt(np.log(1.0 / delta) / (2.0 * n_cal)))

    sd = np.sqrt(np.maximum(var_cal, 1e-12))
    r = np.abs(z_cal - mu_cal) / sd
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")

    q = 1.0 - threshold
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(r, q))

def make_time_features(dates: pd.Series, d0: pd.Timestamp) -> np.ndarray:
    """
    Time-series friendly features:
      - t (days since d0)
      - day-of-week, month
      - weekly / yearly cyclic terms
    """
    t = (dates - d0).dt.days.to_numpy()
    dow = dates.dt.dayofweek.to_numpy()
    month = dates.dt.month.to_numpy()

    sin7 = np.sin(2 * np.pi * t / 7.0)
    cos7 = np.cos(2 * np.pi * t / 7.0)
    sin365 = np.sin(2 * np.pi * t / 365.25)
    cos365 = np.cos(2 * np.pi * t / 365.25)

    X = np.column_stack([t, dow, month, sin7, cos7, sin365, cos365])
    return X

# =========================
# ---DEBUG : Scale이 의미가 있는지 확인
# =========================
def q(x, qs=(0, 0.01, 0.5, 0.99, 1.0)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return np.quantile(x, qs)

def debug_scale_invariance(
    df: pd.DataFrame,
    seed: int = 1,
    split_mode: str = "random",
    target_sizes=(365, 365, 360),
    alpha: float = 0.10,
    delta: float = 0.05,
    tf_mode: str = "asinh",
    scale_list=(1e5, 1e6, 1e7),
    a_end="2020-12-31",
    b_start="2021-01-01",
    train_frac_A=0.5,
):
    # --- fixed split (important) ---
    if split_mode == "random":
        df_tr, df_cal, df_te = split_random(df, seed=seed, target_sizes=target_sizes)
    else:
        df_tr, df_cal, df_te = split_time(df, a_end=a_end, b_start=b_start,
                                          train_frac_A=train_frac_A, seed=seed)

    print(f"\n[Split] mode={split_mode} seed={seed} |train|={len(df_tr)} |cal|={len(df_cal)} |test|={len(df_te)}")

    # --- run Ours only (hetero) with diagnostics ---
    for c in scale_list:
        tf_fn, itf_fn = make_transform(tf_mode=tf_mode, scale_c=float(c))

        # ===== replicate core of run_ours_1d but expose internals =====
        d0 = df_tr["date"].min()
        X_tr  = make_time_features(df_tr["date"], d0)
        X_cal = make_time_features(df_cal["date"], d0)
        X_te  = make_time_features(df_te["date"], d0)

        y_tr  = df_tr["throughput"].to_numpy()
        y_cal = df_cal["throughput"].to_numpy()
        y_te  = df_te["throughput"].to_numpy()

        z_tr  = tf_fn(y_tr)
        z_cal = tf_fn(y_cal)

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
        finite = np.isfinite(res2_tr)
        # 상대 eps : train residual scale에 비례
        eps = 1e-3 * np.median(res2_tr[finite]) if np.any(finite) else 1e-6
        eps = max(eps, 1e-10)

        logv_tr = np.log(res2_tr + eps)
        var_pipe.fit(X_tr, logv_tr)

        mu_cal = mean_pipe.predict(X_cal)
        logv_cal = var_pipe.predict(X_cal)
        var_cal = np.maximum(np.exp(logv_cal), eps)

        lam = find_lambda_fast(alpha, delta, z_cal, mu_cal, var_cal)

        mu_te = mean_pipe.predict(X_te)
        logv_te = var_pipe.predict(X_te)
        var_te = np.maximum(np.exp(logv_te), eps)

        z_margin = lam * np.sqrt(var_te)
        z_lo = mu_te - z_margin
        z_hi = mu_te + z_margin

        lower = np.maximum(itf_fn(z_lo), 0.0)
        upper = itf_fn(z_hi)

        content = float(np.mean((y_te >= lower) & (y_te <= upper)))
        mean_width = float(np.mean(upper - lower))

        # ===== print diagnostics =====
        print(f"\n[c={c:.1e}] tf={tf_mode}")
        print("  lam:", lam)
        print("  z_tr  q:", q(z_tr))
        print("  var_cal q:", q(var_cal))
        print("  var_te  q:", q(var_te))
        print("  z_margin q:", q(z_margin))
        print("  width q:", q(upper - lower))
        print("  content:", content, " mean_width:", mean_width)

# =========================
# P-spline utilities (GY)
# =========================
def second_diff_penalty(m: int) -> np.ndarray:
    I = np.eye(m)
    D2 = np.diff(I, n=2, axis=0)
    return D2

def fit_pspline_1d(x: np.ndarray, y: np.ndarray,
                   n_knots: int = 12, degree: int = 3,
                   lam_grid: np.ndarray | None = None):
    """
    Penalized B-spline regression (P-spline) with 2nd-difference penalty.
    Uses basic GCV over lam_grid.
    """
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
                trans=trans, B=B, BtB=BtB, A=A, beta=beta,
                fitted=fitted
            )
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


# =========================
# Split definition (time-respecting)
# =========================
@dataclass(frozen=True)
class TSASplit:
    df_train: pd.DataFrame
    df_cal: pd.DataFrame
    df_test: pd.DataFrame
    split_mode: str

def make_time_split(df: pd.DataFrame,
                    a_end: str = "2020-12-31",
                    b_start: str = "2021-01-01",
                    train_frac_within_A: float = 0.5) -> TSASplit:
    a_end = pd.to_datetime(a_end)
    b_start = pd.to_datetime(b_start)

    dfA = df[df["date"] <= a_end].copy().sort_values("date").reset_index(drop=True)
    dfB = df[df["date"] >= b_start].copy().sort_values("date").reset_index(drop=True)

    if len(dfA) < 50 or len(dfB) < 50:
        raise ValueError(f"Too small split: |A|={len(dfA)} |B|={len(dfB)}")

    nA = len(dfA)
    n_tr = int(np.floor(train_frac_within_A * nA))
    n_tr = max(10, min(n_tr, nA - 10))

    df_train = dfA.iloc[:n_tr].copy().reset_index(drop=True)
    df_cal   = dfA.iloc[n_tr:].copy().reset_index(drop=True)
    df_test  = dfB.copy().reset_index(drop=True)

    return TSASplit(df_train=df_train, df_cal=df_cal, df_test=df_test)


# =========================
# Split Method
# =========================
def split_time(df: pd.DataFrame, a_end: str, b_start: str, train_frac_A: float, seed: int):
    a_end = pd.to_datetime(a_end)
    b_start = pd.to_datetime(b_start)

    dfA = df[df["date"] <= a_end].copy()
    dfB = df[df["date"] >= b_start].copy()

    # A를 섞어서 train/cal로 (기존 너희 코드와 동일한 철학)
    rng = np.random.default_rng(seed)
    idxA = np.arange(len(dfA))
    rng.shuffle(idxA)

    nA = len(dfA)
    n_tr = int(np.floor(train_frac_A * nA))
    tr_idx = idxA[:n_tr]
    cal_idx = idxA[n_tr:]

    df_tr = dfA.iloc[tr_idx].reset_index(drop=True)
    df_cal = dfA.iloc[cal_idx].reset_index(drop=True)
    df_te = dfB.reset_index(drop=True)

    return df_tr, df_cal, df_te


def split_random(df: pd.DataFrame, seed: int,
                 test_frac: float | None = None,
                 cal_frac: float | None = None,
                 target_sizes: tuple[int, int, int] | None = None):
    """
    Random i.i.d split.
    - If target_sizes is provided, uses exact (n_train, n_cal, n_test).
    - Else uses fractions:
        test_frac for test,
        cal_frac among remaining (default 0.5).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)

    if target_sizes is not None:
        n_tr, n_cal, n_te = target_sizes
        assert n_tr + n_cal + n_te <= n, "target_sizes exceed dataset size"
        tr_idx = idx[:n_tr]
        cal_idx = idx[n_tr:n_tr + n_cal]
        te_idx = idx[n_tr + n_cal:n_tr + n_cal + n_te]
    else:
        if test_frac is None:
            test_frac = 0.25
        if cal_frac is None:
            cal_frac = 0.5  # among remaining after test

        n_te = int(np.floor(test_frac * n))
        rem = n - n_te
        n_cal = int(np.floor(cal_frac * rem))
        n_tr = rem - n_cal

        te_idx = idx[:n_te]
        cal_idx = idx[n_te:n_te + n_cal]
        tr_idx = idx[n_te + n_cal:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_cal = df.iloc[cal_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    return df_tr, df_cal, df_te

# =========================
# Methods
# =========================

def run_ours_1d(df_train: pd.DataFrame, df_cal: pd.DataFrame, df_test: pd.DataFrame,
               alpha: float, delta: float, seed: int, debug: bool = False,
               tf_fn = tf_fn, itf_fn = itf_fn):
    d0 = df_train["date"].min()

    X_tr  = make_time_features(df_train["date"], d0)
    X_cal = make_time_features(df_cal["date"], d0)
    X_te  = make_time_features(df_test["date"], d0)

    y_tr  = df_train["throughput"].to_numpy()
    y_cal = df_cal["throughput"].to_numpy()
    y_te  = df_test["throughput"].to_numpy()

    z_tr = tf_fn(y_tr)
    z_cal = tf_fn(y_cal)

    mean_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", GradientBoostingRegressor(random_state=seed)),
    ])

    # log-variance regression for stability
    var_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", GradientBoostingRegressor(random_state=seed + 1)),
    ])

    # mean fit
    mean_pipe.fit(X_tr, z_tr)
    mu_tr = mean_pipe.predict(X_tr)

    # log-variance fit
    eps = 1e-6
    res2_tr = (z_tr - mu_tr) ** 2
    logv_tr = np.log(res2_tr + eps)

    var_pipe.fit(X_tr, logv_tr)

    mu_cal = mean_pipe.predict(X_cal)
    logv_cal = var_pipe.predict(X_cal)
    var_cal = np.maximum(np.exp(logv_cal), eps)

    lam = find_lambda_fast(alpha, delta, z_cal, mu_cal, var_cal)

    mu_te = mean_pipe.predict(X_te)
    logv_te = var_pipe.predict(X_te)
    var_te = np.maximum(np.exp(logv_te), eps)

    z_margin = lam * np.sqrt(var_te)
    z_lo = mu_te - z_margin
    z_hi = mu_te + z_margin

    lower = np.maximum(itf_fn(z_lo), 0.0)
    upper = itf_fn(z_hi)

    content = float(np.mean((y_te >= lower) & (y_te <= upper)))
    mean_width = float(np.mean(upper - lower))

    if debug:
        sd_cal = np.sqrt(var_cal)
        r = np.abs(z_cal - mu_cal) / sd_cal
        rq = np.quantile(r[np.isfinite(r)], [0.5, 0.9, 0.95, 0.99])

        print("[Ours debug]")
        print("  lam:", lam)
        print("  var_cal min/max:", var_cal.min(), var_cal.max())
        print("  var_te  min/max:", var_te.min(), var_te.max())
        print("  z_margin median:", float(np.median(z_margin)))
        print("  r quantiles:", rq)


    return dict(method="Ours", lambda_=lam, content=content, 
                mean_width=mean_width, lower=lower, upper=upper)

def run_ours_homo_1d(df_train: pd.DataFrame,
                     df_cal: pd.DataFrame,
                     df_test: pd.DataFrame,
                     alpha: float,
                     delta: float,
                     seed: int,
                     tf_fn = tf_fn,
                     itf_fn = itf_fn,
                     scale_c: float = 1e6,
                     robust_sigma: bool = False):
    """
    Ours-homo:
      - fit mean model (GBR) on train
      - estimate ONE global sigma on train residuals (in z-space)
      - lambda from calibration standardized residuals using Hoeffding-adjusted quantile
      - interval: z in [mu - lam*sigma, mu + lam*sigma]
    """

    d0 = df_train["date"].min()
    x_tr = (df_train["date"] - d0).dt.days.to_numpy()
    x_cal = (df_cal["date"] - d0).dt.days.to_numpy()
    x_te = (df_test["date"] - d0).dt.days.to_numpy()

    y_tr = df_train["throughput"].to_numpy()
    y_cal = df_cal["throughput"].to_numpy()
    y_te = df_test["throughput"].to_numpy()

    z_tr = tf_fn(y_tr)
    z_cal = tf_fn(y_cal)

    # mean model (1D feature)
    X_tr = x_tr.reshape(-1, 1)
    X_cal = x_cal.reshape(-1, 1)
    X_te = x_te.reshape(-1, 1)

    mean_model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", GradientBoostingRegressor(random_state=seed)),
    ])
    mean_model.fit(X_tr, z_tr)

    mu_tr = mean_model.predict(X_tr)
    mu_cal = mean_model.predict(X_cal)

    # global sigma estimate (z-space)
    resid_tr = z_tr - mu_tr

    if robust_sigma:
        # robust: MAD -> sigma approx (for normal), but still fine as scale proxy
        mad = np.median(np.abs(resid_tr - np.median(resid_tr)))
        sigma = float(max(1e-8, 1.4826 * mad))
    else:
        sigma = float(np.sqrt(max(1e-12, np.mean(resid_tr ** 2))))

    var_cal = np.full_like(mu_cal, sigma**2, dtype=float)

    # lambda from calibration (still uses your Hoeffding quantile routine)
    lam = find_lambda_fast(alpha, delta, z_cal, mu_cal, var_cal)

    # test interval
    mu_te = mean_model.predict(X_te)

    lower_z = mu_te - lam * sigma
    upper_z = mu_te + lam * sigma

    lower = np.maximum(itf_fn(lower_z), 0.0)
    upper = itf_fn(upper_z)

    content = float(np.mean((y_te >= lower) & (y_te <= upper)))
    mean_width = float(np.mean(upper - lower))

    return dict(
        method="Ours-homo",
        lambda_=lam,
        sigma=sigma,
        content=content,
        mean_width=mean_width,
        lower = lower,
        upper = upper,
    )

def run_ours_pspline_hetero(
    df_train: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    alpha: float,
    delta: float,
    tf_fn,
    itf_fn,
    n_knots: int = 12,
    degree: int = 3,
    eps: float = 1e-8,
    floor_q: float = 0.05,       # variance floor quantile (0.05~0.10 추천)
    z_clip: float | None = None, # overflow 디버깅용 (필요 없으면 None)
    debug: bool = False,
):
    """
    Ours-PSpline-hetero (모델 통제용):
      - mean: P-spline on z (train)
      - var:  P-spline on log(res2 + eps) (train)  -> var = exp(pred)
      - calibration: Hoeffding-adjusted quantile로 lambda 추정
          r_i = |z_i - mu(x_i)| / sqrt(var(x_i))
      - test: z-interval = mu(x) ± lambda * sqrt(var(x))
      - back-transform to y via itf_fn

    비교 포인트:
      - GY-hetero와 동일한 base learner(P-spline)를 쓰되,
        interval rule만 ours(Hoeffding quantile) vs GY(k-factor)로 비교 가능.
    """

    # ---- x (1D time index) ----
    d0 = df_train["date"].min()
    x_tr  = (df_train["date"] - d0).dt.days.to_numpy(dtype=float)
    x_cal = (df_cal["date"]   - d0).dt.days.to_numpy(dtype=float)
    x_te  = (df_test["date"]  - d0).dt.days.to_numpy(dtype=float)

    # ---- y -> z ----
    y_tr  = df_train["throughput"].to_numpy(dtype=float)
    y_cal = df_cal["throughput"].to_numpy(dtype=float)
    y_te  = df_test["throughput"].to_numpy(dtype=float)

    z_tr  = tf_fn(y_tr)
    z_cal = tf_fn(y_cal)

    # ---- 1) mean on z (train) ----
    mean_fit = fit_pspline_1d(x_tr, z_tr, n_knots=n_knots, degree=degree)
    mu_tr = mean_fit["fitted"]

    # ---- 2) variance on log(res2 + eps) (train) ----
    res = z_tr - mu_tr
    res2 = res**2

    # 데이터 기반 floor (너무 작은 값이면 표준화가 폭주)
    finite_res2 = res2[np.isfinite(res2)]
    floor = float(np.quantile(finite_res2, floor_q)) if finite_res2.size else 1e-6
    floor = max(floor, 1e-10)

    log_res2 = np.log(res2 + eps)
    var_fit = fit_pspline_1d(x_tr, log_res2, n_knots=n_knots, degree=degree)

    # ---- calibration predictions ----
    mu_cal = pspline_predict(mean_fit, x_cal)
    logv_cal = pspline_predict(var_fit, x_cal)
    var_cal = np.exp(logv_cal)
    var_cal = np.maximum(var_cal, floor)

    # ---- lambda (Hoeffding quantile) ----
    lam = find_lambda_fast(alpha, delta, z_cal=z_cal, mu_cal=mu_cal, var_cal=var_cal)

    # ---- test predictions ----
    mu_te = pspline_predict(mean_fit, x_te)
    logv_te = pspline_predict(var_fit, x_te)
    var_te = np.exp(logv_te)
    var_te = np.maximum(var_te, floor)

    half = lam * np.sqrt(var_te)
    lower_z = mu_te - half
    upper_z = mu_te + half

    if z_clip is not None:
        lower_z = np.clip(lower_z, -z_clip, z_clip)
        upper_z = np.clip(upper_z, -z_clip, z_clip)

    lower = np.maximum(itf_fn(lower_z), 0.0)
    upper = itf_fn(upper_z)

    content = float(np.mean((y_te >= lower) & (y_te <= upper)))
    w = upper - lower
    finite = np.isfinite(w)
    mean_width = float(np.mean(w[finite]) if np.any(finite) else float("inf"))

    if debug:
        def q(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            return None if x.size == 0 else np.quantile(x, [0, 0.01, 0.5, 0.99, 1.0])

        print("[Ours-PS hetero debug]")
        print("  floor:", floor)
        print("  lambda:", lam)
        print("  var_cal q:", q(var_cal))
        print("  var_te  q:", q(var_te))
        print("  half    q:", q(half))
        print("  upper_z q:", q(upper_z))
        print("  upper   q:", q(upper))

    return dict(
        method="Ours-PS-hetero",
        lambda_=float(lam),
        content=content,
        mean_width=mean_width,
        lower=lower,
        upper=upper,
    )

def _quantiles(x, qs=(0, 0.01, 0.5, 0.99, 1.0)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return np.quantile(x, qs)

def run_gy_homo(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    alpha: float,
    delta: float,
    tf_fn=None,
    itf_fn=None,
    n_knots: int = 12,
    degree: int = 3,
    z_clip: float | None = None,  # ex) 10.0 (optional)
    debug: bool = False,
):
    """
    GY-homo:
      - Fit P-spline mean on transformed z
      - Use k-factor with nu=n-1, s2 from residuals
    Stability tweaks:
      - optional z_clip before itf to avoid overflow in expm1/sinh
    """
    if tf_fn is None or itf_fn is None:
        raise ValueError("tf_fn and itf_fn must be provided")

    d0 = df_train["date"].min()
    x_tr = (df_train["date"] - d0).dt.days.to_numpy(dtype=float)
    x_te = (df_test["date"] - d0).dt.days.to_numpy(dtype=float)

    y_tr = df_train["throughput"].to_numpy(dtype=float)
    y_te = df_test["throughput"].to_numpy(dtype=float)

    z_tr = tf_fn(y_tr)

    mean_fit = fit_pspline_1d(x_tr, z_tr, n_knots=n_knots, degree=degree)
    z_hat_tr = mean_fit["fitted"]

    n = len(x_tr)
    nu = max(1.0, n - 1.0)
    s2 = float(np.sum((z_tr - z_hat_tr) ** 2) / nu)
    s2 = max(s2, 1e-12)

    z_hat_te = pspline_predict(mean_fit, x_te)
    k_te = k_on_new_x(x_te, mean_fit, nu=nu, P=1.0 - alpha, gamma=delta)

    half = np.sqrt(s2) * k_te
    upper_z = z_hat_te + half
    lower_z = z_hat_te - half

    if z_clip is not None:
        upper_z = np.clip(upper_z, -z_clip, z_clip)
        lower_z = np.clip(lower_z, -z_clip, z_clip)

    upper = itf_fn(upper_z)
    lower = np.maximum(itf_fn(lower_z), 0.0)

    content = float(np.mean((y_te >= lower) & (y_te <= upper)))
    mean_width = float(np.mean(upper - lower))

    if debug:
        print("[GY-homo debug]")
        print("  z_tr q:", _quantiles(z_tr))
        print("  s2:", s2)
        print("  k_te q:", _quantiles(k_te))
        print("  upper_z q:", _quantiles(upper_z))
        print("  upper  q:", _quantiles(upper))

    return dict(
        method="GY-homo",
        lambda_=np.nan,
        content=content,
        mean_width=mean_width,
        lower=lower,
        upper=upper,
    )


def run_gy_hetero(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    alpha: float,
    delta: float,
    tf_fn=None,
    itf_fn=None,
    n_knots: int = 12,
    degree: int = 3,
    eps: float = 1e-8,
    floor_q: float = 0.05,         # data-driven floor quantile for var (recommended 0.05~0.10)
    z_clip: float | None = None,    # ex) 10.0 (optional)
    debug: bool = False,
):
    """
    GY-hetero (stabilized):
      1) Fit mean P-spline on z
      2) Fit log-variance P-spline on log(res2 + eps)  <-- key stabilization
      3) Standardize residuals: t = (z - mu)/sqrt(var_hat)
      4) Fit P-spline on t, compute est_var
      5) Test: build intervals in t-space, map back via sqrt(var_te), then itf

    Additional stabilization:
      - variance floor = quantile(res2, floor_q) (not 1e-6)
      - optional z_clip before itf
    """
    if tf_fn is None or itf_fn is None:
        raise ValueError("tf_fn and itf_fn must be provided")

    d0 = df_train["date"].min()
    x_tr = (df_train["date"] - d0).dt.days.to_numpy(dtype=float)
    x_te = (df_test["date"] - d0).dt.days.to_numpy(dtype=float)

    y_tr = df_train["throughput"].to_numpy(dtype=float)
    y_te = df_test["throughput"].to_numpy(dtype=float)

    z_tr = tf_fn(y_tr)

    # 1) mean on z
    mean_fit = fit_pspline_1d(x_tr, z_tr, n_knots=n_knots, degree=degree)
    mu_tr = mean_fit["fitted"]
    res = z_tr - mu_tr
    res2 = res ** 2

    # --- variance floor (z-scale dependent, 반드시 데이터 기반으로) ---
    # 너무 작으면 표준화가 폭주한다.
    floor = float(np.quantile(res2[np.isfinite(res2)], floor_q)) if np.any(np.isfinite(res2)) else 1e-6
    floor = max(floor, 1e-10)

    # 2) variance on log(res2 + eps)
    log_res2 = np.log(res2 + eps)
    var_fit = fit_pspline_1d(x_tr, log_res2, n_knots=n_knots, degree=degree)
    logv_hat_tr = var_fit["fitted"]
    var_hat = np.exp(logv_hat_tr)
    var_hat = np.maximum(var_hat, floor)

    # 3) standardized residuals
    t = res / np.sqrt(var_hat)

    # 4) spline on standardized residuals
    t_fit = fit_pspline_1d(x_tr, t, n_knots=n_knots, degree=degree)
    t_hat_tr = t_fit["fitted"]

    n = len(x_tr)
    nu = max(1.0, n - 1.0)
    est_var = float(np.sum((t - t_hat_tr) ** 2) / nu)
    est_var = max(est_var, 1e-12)

    # test variance: predict log-var then exp
    logv_te = pspline_predict(var_fit, x_te)
    var_te = np.exp(logv_te)
    var_te = np.maximum(var_te, floor)

    # test t-mean
    t_hat_te = pspline_predict(t_fit, x_te)
    k_te = k_on_new_x(x_te, t_fit, nu=nu, P=1.0 - alpha, gamma=delta)

    half_t = np.sqrt(est_var) * k_te
    upper_t = t_hat_te + half_t
    lower_t = t_hat_te - half_t

    upper_z = mu_te = pspline_predict(mean_fit, x_te)  # mean on test z
    # NOTE: interval in z = mu_te + t * sqrt(var_te)
    upper_z = mu_te + upper_t * np.sqrt(var_te)
    lower_z = mu_te + lower_t * np.sqrt(var_te)

    if z_clip is not None:
        upper_z = np.clip(upper_z, -z_clip, z_clip)
        lower_z = np.clip(lower_z, -z_clip, z_clip)

    upper = itf_fn(upper_z)
    lower = np.maximum(itf_fn(lower_z), 0.0)

    content = float(np.mean((y_te >= lower) & (y_te <= upper)))
    w = upper - lower
    finite = np.isfinite(w)
    mean_width = float(np.mean(w[finite]) if np.any(finite) else float("inf"))

    if debug:
        print("[GY-hetero debug]")
        print("  z_tr q:", _quantiles(z_tr))
        print("  mu_tr q:", _quantiles(mu_tr))
        print("  res2 q:", _quantiles(res2))
        print("  floor:", floor)
        print("  var_hat q:", _quantiles(var_hat))
        print("  t q:", _quantiles(t))
        print("  est_var:", est_var)
        print("  var_te q:", _quantiles(var_te))
        print("  k_te q:", _quantiles(k_te))
        print("  upper_z q:", _quantiles(upper_z))
        print("  upper  q:", _quantiles(upper))

    return dict(
        method="GY-hetero",
        lambda_=np.nan,
        content=content,
        mean_width=mean_width,
        lower=lower,
        upper=upper,
    )

# =========================
# Runner
# =========================

def run_many_seeds(df: pd.DataFrame,
                   seeds=range(1, 51),
                   alpha: float = 0.10,
                   delta: float = 0.05,
                   tf_mode: str = "asinh",
                   scale_c: float = 1e6,
                   split_mode: str = "random",
                   target_sizes: tuple[int,int,int] = (365,365,360),
                   a_end: str = "2020-12-31",
                   b_start: str = "2021-01-01",
                   train_frac_A: float = 0.5) -> pd.DataFrame:
        rows = []
        interval_rows = []

        tf_fn, itf_fn = make_transform(tf_mode=tf_mode, scale_c=scale_c)

        for seed in seeds:
            if split_mode == "random":
                df_tr, df_cal, df_te = split_random(df, seed=seed, target_sizes=target_sizes)
            else:
                df_tr, df_cal, df_te = split_time(df, a_end=a_end, b_start=b_start,
                                                train_frac_A=train_frac_A, seed=seed)

            split = TSASplit(df_tr, df_cal, df_te, split_mode=split_mode)

            r_ours = run_ours_1d(split.df_train, split.df_cal, split.df_test,
                                alpha, delta, seed, tf_fn=tf_fn, itf_fn=itf_fn)

            r_ours_homo = run_ours_homo_1d(split.df_train, split.df_cal, split.df_test,
                                        alpha, delta, seed, tf_fn=tf_fn, itf_fn=itf_fn)
            
            r_ours_ps = run_ours_pspline_hetero(
            df_train=split.df_train,
            df_cal=split.df_cal,
            df_test=split.df_test,
            alpha=alpha,
            delta=delta,
            tf_fn=tf_fn,
            itf_fn=itf_fn)

            r_gy_h = run_gy_homo(split.df_train, split.df_test, alpha, delta,
                                tf_fn=tf_fn, itf_fn=itf_fn)

            r_gy_he = run_gy_hetero(split.df_train, split.df_test, alpha, delta,
                                    tf_fn=tf_fn, itf_fn=itf_fn)

            results = [r_ours, r_ours_homo, r_ours_ps, r_gy_h, r_gy_he]
            for r in results:
                r["seed"] = seed
                r["split_mode"] = split_mode
                r["tf_mode"] = tf_mode
                r["scale_c"] = scale_c
                r["n_train"] = len(df_tr)
                r["n_cal"] = len(df_cal)
                r["n_test"] = len(df_te)

            rows.extend(results)

            for r in results:
                lo = np.asarray(r["lower"], dtype=float)
                hi = np.asarray(r["upper"], dtype=float)
                tmp = pd.DataFrame({
                    "seed": seed,
                    "method": r["method"],
                    "split_mode": split_mode,
                    "tf_mode": tf_mode,
                    "scale_c": scale_c,
                    "date": df_te["date"].astype(str).to_numpy() if "date" in df_te.columns else None,
                    "y": df_te["throughput"].to_numpy(),
                    "lower": lo,
                    "upper": hi,
                })
                if "date" in tmp.columns and tmp["date"].isna().all():
                    tmp = tmp.drop(columns=["date"])

                interval_rows.append(tmp)

        df_res = pd.DataFrame(rows)
        df_intervals = pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame()
        return df_res, df_intervals

def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default=DEFAULT_TSA_URL,
                    help="CSV path or URL (two columns: date, throughput; no header).")

    # time split options
    ap.add_argument("--a_end", type=str, default="2020-12-31")
    ap.add_argument("--b_start", type=str, default="2021-01-01")
    ap.add_argument("--train_frac_A", type=float, default=0.5)

    # CP params
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.05)

    # seeds
    ap.add_argument("--seed_from", type=int, default=1)
    ap.add_argument("--seed_to", type=int, default=50)

    # transform
    ap.add_argument("--tf_mode", type=str, default="asinh",
                    choices=["asinh", "log_scaled"],
                    help="Transformation mode.")
    ap.add_argument("--scale_c", type=float, default=1e6,
                    help="Scaling constant for transformation.")

    # split mode
    ap.add_argument("--split_mode", type=str, default="random",
                    choices=["time", "random"],
                    help="time: A->train/cal, B->test; random: i.i.d. random split")

    ap.add_argument("--test_frac", type=float, default=None,
                    help="(random only) test fraction. If None, uses target_sizes.")
    ap.add_argument("--cal_frac", type=float, default=None,
                    help="(random only) calibration fraction among remaining. Default 0.5.")

    ap.add_argument("--out_csv", type=str, default="results_tsa_3way.csv")
    args = ap.parse_args()

    # -------------------------
    # Load data
    # -------------------------
    df = load_tsa(args.data)

    # -------------------------
    # Run
    # -------------------------
    seeds = range(args.seed_from, args.seed_to + 1)
    # Run
    df_res, df_intervals = run_many_seeds(
        df,
        seeds=seeds,
        alpha=args.alpha,
        delta=args.delta,
        tf_mode=args.tf_mode,
        scale_c=args.scale_c,
        split_mode=args.split_mode,
        target_sizes=(365,365,360),
        a_end=args.a_end,
        b_start=args.b_start,
        train_frac_A=args.train_frac_A
    )

    # Summary print
    summary = (
        df_res.groupby("method")[["content", "mean_width"]]
        .agg(["mean", "std", "median"])
    )
    print("\n=== Summary ===")
    print(summary.round(4))

    # Save summary metrics
    df_res.to_csv(args.out_csv, index=False)
    print(f"\nSaved summary: {args.out_csv}")

    # Save intervals (long format)
    interval_csv = Path(args.out_csv).with_name(Path(args.out_csv).stem + "_intervals.csv")
    df_intervals.to_csv(interval_csv, index=False)
    print(f"Saved intervals: {interval_csv}")


    # -------------------------
    # Diagnostics
    # -------------------------
    bad = df_res[~np.isfinite(df_res["mean_width"])]
    if len(bad) > 0:
        print("\n[WARN] Non-finite mean_width detected (first 20 rows):")
        print(bad[["method", "seed", "content", "mean_width"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
