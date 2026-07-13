from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from scipy.stats import chi2, ncx2
from patsy import dmatrix


# ----------------------------
# Happy loader
# ----------------------------
HAPPY_COLS = [
    "id", "mag_r", "u_g", "g_r", "r_i", "i_z",
    "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5",
]

def load_happy(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(
        path, sep=r"\s+", comment="#", header=None,
        names=HAPPY_COLS, engine="python"
    )

# ----------------------------
# transforms
# ----------------------------
def tf(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)

def itf(z: np.ndarray) -> np.ndarray:
    return np.expm1(z)

# ----------------------------
# Hoeffding lambda (fast)
# ----------------------------
def find_lambda_fast(alpha: float, delta: float,
                     z_cal: np.ndarray, mu_cal: np.ndarray, var_cal: np.ndarray) -> float:
    n_cal = len(z_cal)
    threshold = max(0.0, alpha - np.sqrt(np.log(1.0 / delta) / (2.0 * n_cal)))

    sd = np.sqrt(np.maximum(var_cal, 1e-12))
    r = np.abs(z_cal - mu_cal) / sd

    # smallest lambda such that P(r >= lambda) <= threshold
    q = 1.0 - threshold  # want quantile at 1-threshold
    q = min(max(q, 0.0), 1.0)
    r = r[~np.isnan(r)]
    return float(np.quantile(r, q))

# ----------------------------
# Penalized B-spline with 2nd-diff penalty (GY spline surrogate)
# ----------------------------
def make_bspline_basis(x: np.ndarray, df: int) -> np.ndarray:
    # x: shape (n,)
    # include_intercept=True makes it closer to R's bs(..., intercept=TRUE)
    B = dmatrix(
        f"bs(x, df={df}, degree=3, include_intercept=True) - 1",
        {"x": x},
        return_type="dataframe"
    )
    return np.asarray(B)

def second_diff_matrix(p: int, order: int = 2) -> np.ndarray:
    # D: (p-order) x p second-difference operator
    D = np.eye(p)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D

def fit_penalized_spline_gcv(x: np.ndarray, y: np.ndarray, df: int,
                             lam_grid: np.ndarray | None = None) -> dict:
    """
    Fit beta = argmin ||y - B beta||^2 + lam ||D beta||^2.
    Choose lam by GCV over lam_grid.
    Returns dict with beta, lam, B, S (smoother matrix), df_eff, yhat, rss.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = len(x)

    B = make_bspline_basis(x, df=df)  # n x p
    p = B.shape[1]
    D = second_diff_matrix(p, order=2)
    DtD = D.T @ D
    BtB = B.T @ B
    Bty = B.T @ y

    if lam_grid is None:
        lam_grid = np.logspace(-8, 4, 80)

    best = None

    I = np.eye(p)
    for lam in lam_grid:
        M = BtB + lam * DtD
        # solve (BtB + lam DtD) beta = B^T y
        try:
            beta = np.linalg.solve(M, Bty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(M, Bty, rcond=None)[0]

        yhat = B @ beta
        resid = y - yhat
        rss = float(resid.T @ resid)

        # S = B (BtB + lam DtD)^{-1} B^T
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        S = B @ Minv @ B.T
        df_eff = float(np.trace(S))

        denom = (n - df_eff)
        if denom <= 1e-8:
            continue
        gcv = (n * rss) / (denom ** 2)

        if (best is None) or (gcv < best["gcv"]):
            best = {
                "beta": beta, "lam": float(lam), "B": B, "S": S,
                "Minv": Minv, "BtB": BtB,
                "df_eff": df_eff, "yhat": yhat, "rss": rss, "gcv": float(gcv)
            }

    if best is None:
        raise RuntimeError("GCV failed to pick a lambda. Try expanding lam_grid or adjusting df.")
    return best

def predict_penalized_spline(x_train: np.ndarray, fit: dict, x_new: np.ndarray, df: int) -> np.ndarray:
    # Using the same basis spec; patsy will create consistent basis if df matches.
    # In practice, for stable prediction you want to build basis on new x with same knots.
    # Here we accept minor approximation; sufficient for visualization & comparison.
    B_new = make_bspline_basis(np.asarray(x_new).ravel(), df=df)
    return B_new @ fit["beta"]

def compute_norm_lx_h(B: np.ndarray, Minv: np.ndarray, BtB: np.ndarray) -> np.ndarray:
    # matches your R:
    # A = ginv(BtB + lam D'D)
    # M = A %*% BtB %*% A
    # norm_j = sqrt( b_j^T M b_j )
    M = Minv @ BtB @ Minv
    return np.sqrt(np.sum((B @ M) * B, axis=1))

def find_k_factor(nu: float, norm_lx_h: np.ndarray, P: float, gamma: float) -> np.ndarray:
    # k = sqrt( nu * Q_{ncx2}(P; df=1, ncp=norm^2) / Q_{chi2}(1-gamma; df=nu) )
    num = ncx2.ppf(P, df=1, nc=norm_lx_h**2)
    den = chi2.ppf(1.0 - gamma, df=nu)
    return np.sqrt(nu * num / den)

# ----------------------------
# Ours(1D) via spline+ridge (mean) and spline+ridge (var)
# ----------------------------
def fit_ours_1d(train_x: np.ndarray, train_y: np.ndarray,
                cal_x: np.ndarray, cal_y: np.ndarray,
                alpha: float, delta: float, seed: int) -> dict:
    train_x = np.asarray(train_x).reshape(-1, 1)
    cal_x = np.asarray(cal_x).reshape(-1, 1)
    train_y = np.asarray(train_y).ravel()
    cal_y = np.asarray(cal_y).ravel()

    train_z = tf(train_y)
    cal_z = tf(cal_y)

    mean_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=12, degree=3, include_bias=True)),
        ("ridge", Ridge(alpha=1e-3, random_state=seed)),
    ])

    var_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=12, degree=3, include_bias=True)),
        ("ridge", Ridge(alpha=1e-3, random_state=seed)),
    ])

    mean_pipe.fit(train_x, train_z)
    mu_tr = mean_pipe.predict(train_x)
    res2_tr = (train_z - mu_tr) ** 2

    var_pipe.fit(train_x, res2_tr)

    mu_cal = mean_pipe.predict(cal_x)
    var_cal = np.maximum(var_pipe.predict(cal_x), 1e-6)

    lam = find_lambda_fast(alpha, delta, cal_z, mu_cal, var_cal)

    return {
        "mean_pipe": mean_pipe,
        "var_pipe": var_pipe,
        "lambda_hat": float(lam),
        "train_z": train_z, "cal_z": cal_z,
        "mu_tr": mu_tr, "res2_tr": res2_tr,
        "mu_cal": mu_cal, "var_cal": var_cal,
        "cal_x": cal_x.ravel(), "cal_y": cal_y,
        "train_x": train_x.ravel(), "train_y": train_y
    }

def ours_predict_intervals(model: dict, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_new = np.asarray(x_new).reshape(-1, 1)
    mu = model["mean_pipe"].predict(x_new)
    var = np.maximum(model["var_pipe"].predict(x_new), 1e-6)
    lam = model["lambda_hat"]

    lo = np.maximum(itf(mu - lam * np.sqrt(var)), 0.0)
    up = itf(mu + lam * np.sqrt(var))
    pred = itf(mu)
    return lo, up, pred

# ----------------------------
# GY(1D) implementation (penalized B-spline surrogate)
# ----------------------------
def fit_gy_1d(x: np.ndarray, y: np.ndarray,
              alpha: float, gamma: float,
              df_mean: int = 12, seed: int = 0) -> dict:
    """
    Inputs:
      x, y: training (proper+cal) for GY, 1D covariate
      alpha: content -> P = 1-alpha
      gamma: confidence
    Pipeline (log1p scale):
      1) mean fit on z
      2) var fit on res^2
      3) transform y_std = z / sqrt(varhat)
      4) fit penalized spline on y_std, choose lambda by GCV
      5) compute est_var, nu, norm_lx_h, k_factors
      6) form TI on y_std, then back to z, then back to y
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = tf(y)

    # (1) mean z ~ x : use spline+ridge as stable surrogate
    mean_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=12, degree=3, include_bias=True)),
        ("ridge", Ridge(alpha=1e-3, random_state=seed)),
    ])
    mean_pipe.fit(x.reshape(-1, 1), z)
    mu_z = mean_pipe.predict(x.reshape(-1, 1))
    res = z - mu_z

    # (2) variance model on res^2
    var_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=12, degree=3, include_bias=True)),
        ("ridge", Ridge(alpha=1e-3, random_state=seed)),
    ])
    var_pipe.fit(x.reshape(-1, 1), res**2)
    varhat = np.maximum(var_pipe.predict(x.reshape(-1, 1)), 1e-6)

    # (3) standardized response on z-scale
    y_std = z / np.sqrt(varhat)

    # (4) penalized spline fit on y_std with GCV-chosen lambda
    fit_std = fit_penalized_spline_gcv(x, y_std, df=df_mean)
    y_std_hat = fit_std["yhat"]
    B = fit_std["B"]
    Minv = fit_std["Minv"]
    BtB = fit_std["BtB"]
    df_eff = fit_std["df_eff"]

    resid_std = y_std - y_std_hat
    rss = float(resid_std.T @ resid_std)

    # (5) est_var, nu (use df_eff)
    nu = max(5.0, float(len(x) - df_eff))
    est_var = rss / max(1.0, (len(x) - df_eff))

    norm_lx_h = compute_norm_lx_h(B, Minv, BtB)
    k = find_k_factor(nu=nu, norm_lx_h=norm_lx_h, P=1.0 - alpha, gamma=gamma)

    # (6) TI on standardized scale
    lo_std = y_std_hat - np.sqrt(est_var) * k
    up_std = y_std_hat + np.sqrt(est_var) * k

    # back to z-scale
    lo_z = lo_std * np.sqrt(varhat)
    up_z = up_std * np.sqrt(varhat)

    # back to y-scale
    lo = np.maximum(itf(lo_z), 0.0)
    up = itf(up_z)

    return {
        "mean_pipe_z": mean_pipe,
        "var_pipe": var_pipe,
        "varhat": varhat,
        "y_std": y_std,
        "fit_std": fit_std,
        "df_eff": df_eff,
        "lam_gcv": fit_std["lam"],
        "nu": nu,
        "est_var": est_var,
        "norm_lx_h": norm_lx_h,
        "k": k,
        "x": x,
        "y": y,
        "z": z,
        "y_std_hat": y_std_hat,
        "lo": lo,
        "up": up,
    }

def gy_predict_intervals(gy: dict, x_new: np.ndarray, df_mean: int = 12) -> tuple[np.ndarray, np.ndarray]:
    x_new = np.asarray(x_new).ravel()
    # predict varhat on new
    varhat_new = np.maximum(gy["var_pipe"].predict(x_new.reshape(-1, 1)), 1e-6)

    # predict standardized mean on new x via penalized spline beta (approx)
    mu_std_new = predict_penalized_spline(gy["x"], gy["fit_std"], x_new, df=df_mean)

    lo_std = mu_std_new - np.sqrt(gy["est_var"]) * np.interp(x_new, gy["x"], gy["k"])
    up_std = mu_std_new + np.sqrt(gy["est_var"]) * np.interp(x_new, gy["x"], gy["k"])

    lo_z = lo_std * np.sqrt(varhat_new)
    up_z = up_std * np.sqrt(varhat_new)

    lo = np.maximum(itf(lo_z), 0.0)
    up = itf(up_z)
    return lo, up

# ----------------------------
# Plotting
# ----------------------------

# ----------------------------
# Paper plotting helpers
# ----------------------------
def paper_rcparams():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
    })

def save_fig(fig, out_base: Path, dpi: int = 300):
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)

def rolling_mean(x: np.ndarray, y: np.ndarray, window: int = 101) -> np.ndarray:
    """Simple rolling mean assuming x is sorted; window should be odd."""
    window = max(5, int(window))
    if window % 2 == 0:
        window += 1
    # pad edges
    y_pad = np.pad(y, (window//2, window//2), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode="valid")


# ----------------------------
# Paper figures
# ----------------------------
def make_paper_figures(
    x_test_s: np.ndarray,
    y_test_s: np.ndarray,
    ours_lo: np.ndarray, ours_up: np.ndarray,
    gy_lo: np.ndarray, gy_up: np.ndarray,
    ours_content: float, gy_content: float,
    ours_width: float, gy_width: float,
    out_dir: Path,
    seed: int,
):
    paper_rcparams()
    out_dir.mkdir(exist_ok=True)

    # Colors (colorblind-friendly-ish)
    c_ours = "#1f77b4"   # blue
    c_gy   = "#d62728"   # red/brick

    # ============
    # Fig A: Overlay ribbons on test set (main)
    # ============
    fig = plt.figure(figsize=(6.8, 3.6))
    ax = fig.add_subplot(1, 1, 1)

    # scatter (thin)
    ax.scatter(x_test_s, y_test_s,color='black', s=5, alpha=0.15, linewidths=0, zorder=1)

    # ribbons
    ax.fill_between(x_test_s, gy_lo, gy_up, alpha=0.18, color=c_gy, label=None, zorder=2)
    ax.fill_between(x_test_s, ours_lo, ours_up, alpha=0.18, color=c_ours, label=None, zorder=2)

    # median lines (optional but nice in papers)
    ax.plot(x_test_s, 0.5*(gy_lo+gy_up), color=c_gy, linewidth=1.2, zorder=3)
    ax.plot(x_test_s, 0.5*(ours_lo+ours_up), color=c_ours, linewidth=1.2, zorder=3)

    # axes labels only (title removed; explain in caption)
    ax.set_xlabel(r"$\mathrm{mag}_r$")
    ax.set_ylabel(r"$z_{\mathrm{spec}}$")

    # minimalist legend: method + (content, width)
    # (Keep numbers out if you prefer caption-only; I’ll keep it compact)
    lab_gy   = "Parametric TI"
    lab_ours = "HCTI"

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=c_gy,   edgecolor="none", alpha=0.18, label=lab_gy),
        Patch(facecolor=c_ours, edgecolor="none", alpha=0.18, label=lab_ours),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"happyB_overlay_seed{seed}")

    # ============
    # Fig B: Local width vs x (smoothed), on test grid (main or appendix)
    # ============
    w_ours = ours_up - ours_lo
    w_gy   = gy_up - gy_lo

    # smooth for readability (x already sorted)
    w_ours_sm = rolling_mean(x_test_s, w_ours, window=151)
    w_gy_sm   = rolling_mean(x_test_s, w_gy, window=151)

    fig = plt.figure(figsize=(6.8, 3.2))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_test_s, w_gy_sm, color=c_gy, linewidth=1.6, linestyle="-", label="Parametric TI")
    ax.plot(x_test_s, w_ours_sm, color=c_ours, linewidth=1.6, linestyle="-", label="HCTI")

    ax.set_xlabel(r"$\mathrm{mag}_r$")
    ax.set_ylabel("Interval width")

    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"happyB_width_curve_seed{seed}")

    # ============
    # Fig C (optional): coverage indicator vs x (smoothed empirical)
    # ============
    hit_ours = ((y_test_s >= ours_lo) & (y_test_s <= ours_up)).astype(float)
    hit_gy   = ((y_test_s >= gy_lo) & (y_test_s <= gy_up)).astype(float)
    hit_ours_sm = rolling_mean(x_test_s, hit_ours, window=301)
    hit_gy_sm   = rolling_mean(x_test_s, hit_gy, window=301)

    fig = plt.figure(figsize=(6.8, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_test_s, hit_gy_sm, color=c_gy, linewidth=1.6, linestyle="-", label="Parametric TI")
    ax.plot(x_test_s, hit_ours_sm, color=c_ours, linewidth=1.6, linestyle="-", label="HCTI")
    ax.axhline(0.90, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_xlabel(r"$\mathrm{mag}_r$")
    ax.set_ylabel("Empirical content (local)")
    ax.set_ylim(0.0, 1.0)

    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"happyB_local_content_seed{seed}")


def main():
    # Settings
    seed = 123
    n_sample = 5000
    alpha = 0.10
    delta = 0.05
    gamma = delta
    df_gy = 12

    root = Path(__file__).resolve().parents[1]
    dfA = load_happy(root / "data" / "Happy" / "happy_A")
    dfB = load_happy(root / "data" / "Happy" / "happy_B")

    # sample from A
    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    x_all = dfA_s["mag_r"].to_numpy()
    y_all = dfA_s["z_spec"].to_numpy()

    # split train/cal for Ours
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        x_all, y_all, test_size=0.5, random_state=seed, shuffle=True
    )

    # Ours fit (train/cal)
    ours = fit_ours_1d(X_tr, y_tr, X_cal, y_cal, alpha=alpha, delta=delta, seed=seed)

    # GY fit uses (train+cal) like your R code (proper_train_idx = idx[1:(n_train+n_cal)])
    gy = fit_gy_1d(x=np.concatenate([X_tr, X_cal]),
                   y=np.concatenate([y_tr, y_cal]),
                   alpha=alpha, gamma=gamma, df_mean=df_gy, seed=seed)

    # Test set
    x_test = dfB["mag_r"].to_numpy()
    y_test = dfB["z_spec"].to_numpy()
    order = np.argsort(x_test)
    x_test_s = x_test[order]
    y_test_s = y_test[order]

    # intervals on test
    ours_lo, ours_up, ours_pred = ours_predict_intervals(ours, x_test_s)
    gy_lo, gy_up = gy_predict_intervals(gy, x_test_s, df_mean=df_gy)

    ours_content = float(np.mean((y_test_s >= ours_lo) & (y_test_s <= ours_up)))
    gy_content = float(np.mean((y_test_s >= gy_lo) & (y_test_s <= gy_up)))
    ours_width = float(np.mean(ours_up - ours_lo))
    gy_width = float(np.mean(gy_up - gy_lo))

    print(f"[seed={seed}] HCTI lambda_hat={ours['lambda_hat']:.6f}  content={ours_content:.6f}  mean_width={ours_width:.6f}")
    print(f"[seed={seed}] Parametric TI   lam_gcv={gy['lam_gcv']:.6e} df_eff={gy['df_eff']:.2f}  content={gy_content:.6f}  mean_width={gy_width:.6f}")


    out_dir = root / "fig"
    make_paper_figures(
        x_test_s=x_test_s,
        y_test_s=y_test_s,
        ours_lo=ours_lo, ours_up=ours_up,
        gy_lo=gy_lo, gy_up=gy_up,
        ours_content=ours_content, gy_content=gy_content,
        ours_width=ours_width, gy_width=gy_width,
        out_dir=out_dir,
        seed=seed,
    )
    print(f"Saved paper figures into: {out_dir}")

    summary_df = pd.DataFrame({
        "seed": [seed, seed],
        "method": ["HCTI", "Parametric TI"],
        "content": [ours_content, gy_content],
        "mean_width": [ours_width, gy_width],
        "lambda_hat": [ours["lambda_hat"], np.nan],
        "lam_gcv": [np.nan, gy["lam_gcv"]],
        "df_eff": [np.nan, gy["df_eff"]],
    })


    summary_path = out_dir / f"happyB_summary_seed{seed}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

if __name__ == "__main__":
    main()
