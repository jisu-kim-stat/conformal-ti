from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2
from patsy import dmatrix
from sklearn.model_selection import train_test_split


# ----------------------------
# I/O
# ----------------------------
HAPPY_COLS = [
    "id", "mag_r", "u_g", "g_r", "r_i", "i_z",
    "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5",
]

def load_happy(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                       names=HAPPY_COLS, engine="python")


def tf(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)

def itf(z: np.ndarray) -> np.ndarray:
    return np.expm1(z)


# ----------------------------
# Penalized B-spline smoother with GCV
#   yhat = S y,  S = B (B'B + lam D'D)^{-1} B'
# ----------------------------
def make_bspline_basis(x: np.ndarray, df: int) -> np.ndarray:
    B = dmatrix(
        f"bs(x, df={df}, degree=3, include_intercept=True) - 1",
        {"x": np.asarray(x).ravel()},
        return_type="dataframe",
    )
    return np.asarray(B)

def second_diff_matrix(p: int, order: int = 2) -> np.ndarray:
    D = np.eye(p)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D

def fit_pspline_gcv(x: np.ndarray, y: np.ndarray, df: int,
                    lam_grid: np.ndarray | None = None) -> dict:
    """
    Fit penalized spline with 2nd-diff penalty and choose lambda by GCV.
    Returns:
      beta, lam, B, Minv, BtB, S, df_eff, yhat, rss
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
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        beta = Minv @ Bty
        yhat = B @ beta
        resid = y - yhat
        rss = float(resid.T @ resid)

        S = B @ Minv @ B.T
        df_eff = float(np.trace(S))

        denom = (n - df_eff)
        if denom <= 1e-8:
            continue
        gcv = (n * rss) / (denom ** 2)

        if (best is None) or (gcv < best["gcv"]):
            best = dict(
                gcv=float(gcv), lam=float(lam),
                beta=beta, B=B, Minv=Minv, BtB=BtB, S=S,
                df_eff=df_eff, yhat=yhat, rss=rss
            )

    if best is None:
        raise RuntimeError("GCV failed; try different df or lam_grid.")
    return best

def predict_pspline(x_new: np.ndarray, df: int, beta: np.ndarray) -> np.ndarray:
    B_new = make_bspline_basis(np.asarray(x_new).ravel(), df=df)
    return B_new @ beta

def compute_norm_lx_h(B: np.ndarray, Minv: np.ndarray, BtB: np.ndarray) -> np.ndarray:
    # norm_j = sqrt( b_j^T (Minv BtB Minv) b_j )
    M = Minv @ BtB @ Minv
    return np.sqrt(np.sum((B @ M) * B, axis=1))

def find_lambda_hoeffding(alpha: float, delta: float,
                          z_cal: np.ndarray, mu_cal: np.ndarray, var_cal: np.ndarray) -> float:
    n_cal = len(z_cal)
    threshold = max(0.0, alpha - np.sqrt(np.log(1.0 / delta) / (2.0 * n_cal)))
    r = np.abs(z_cal - mu_cal) / np.sqrt(np.maximum(var_cal, 1e-12))
    q = 1.0 - threshold
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(r[~np.isnan(r)], q))

def find_k_factor(nu: float, norm_vals: np.ndarray, P: float, gamma: float) -> np.ndarray:
    # k(x) = sqrt( nu * Q_{ncx2}(P; df=1, nc=norm^2) / Q_{chi2}(1-gamma; df=nu) )
    num = ncx2.ppf(P, df=1, nc=norm_vals**2)
    den = chi2.ppf(1.0 - gamma, df=nu)
    return np.sqrt(nu * num / den)


# ----------------------------
# Fair comparison (same mean/var models)
# ----------------------------
def main():
    # settings
    seed = 123
    n_sample = 5000
    alpha = 0.10
    delta = 0.05
    df_mean = 12
    df_var  = 12
    df_std  = 12  # for GY standardized spline

    root = Path(__file__).resolve().parents[1]
    dfA = load_happy(root / "data" / "Happy" / "happy_A")
    dfB = load_happy(root / "data" / "Happy" / "happy_B")

    # sample fixed 5000 from Happy A
    dfA_s = dfA.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    x = dfA_s["mag_r"].to_numpy()
    y = dfA_s["z_spec"].to_numpy()

    # common split: train/cal
    x_tr, x_cal, y_tr, y_cal = train_test_split(
        x, y, test_size=0.5, random_state=seed, shuffle=True
    )

    # common transforms
    z_tr  = tf(y_tr)
    z_cal = tf(y_cal)

    # ============================
    # (Common) mean and var models
    # ============================
    # mean: z ~ x (fit on train)
    mean_fit = fit_pspline_gcv(x_tr, z_tr, df=df_mean)
    mu_tr = mean_fit["yhat"]

    # var: (z - mu)^2 ~ x (fit on train)
    res2_tr = (z_tr - mu_tr) ** 2
    var_fit = fit_pspline_gcv(x_tr, res2_tr, df=df_var)

    # predict mean/var on calibration
    mu_cal  = predict_pspline(x_cal, df=df_mean, beta=mean_fit["beta"])
    var_cal = np.maximum(predict_pspline(x_cal, df=df_var, beta=var_fit["beta"]), 1e-6)

    # ==========================================
    # (Ours) Hoeffding lambda based on calibration
    # ==========================================
    lam_hat = find_lambda_hoeffding(alpha, delta, z_cal, mu_cal, var_cal)

    # ==========================================
    # (GY) k(x): needs standardized smoother on ALL (train+cal)
    #     BUT mean/var are the SAME as above.
    # ==========================================
    x_all = np.concatenate([x_tr, x_cal])
    y_all = np.concatenate([y_tr, y_cal])
    z_all = tf(y_all)

    # predict the SAME mean/var on x_all using the same fitted betas
    mu_all  = predict_pspline(x_all, df=df_mean, beta=mean_fit["beta"])
    var_all = np.maximum(predict_pspline(x_all, df=df_var, beta=var_fit["beta"]), 1e-6)

    y_std = z_all / np.sqrt(var_all)

    # fit standardized spline on y_std ~ x_all (GY step; linear smoother needed)
    std_fit = fit_pspline_gcv(x_all, y_std, df=df_std)

    # residual variance on standardized scale
    resid_std = y_std - std_fit["yhat"]
    rss_std = float(resid_std.T @ resid_std)

    # degrees of freedom nu: use effective df of standardized smoother
    df_eff = std_fit["df_eff"]
    nu = max(5.0, float(len(x_all) - df_eff))
    est_var = rss_std / max(1.0, (len(x_all) - df_eff))

    # compute k(x_all) then later predict to x_test
    norm_vals_all = compute_norm_lx_h(std_fit["B"], std_fit["Minv"], std_fit["BtB"])
    k_all = find_k_factor(nu=nu, norm_vals=norm_vals_all, P=1.0 - alpha, gamma=delta)

    # ============================
    # Evaluate on Happy B test set
    # ============================
    x_test = dfB["mag_r"].to_numpy()
    y_test = dfB["z_spec"].to_numpy()
    z_test = tf(y_test)

    # mean/var on test (same models!)
    mu_test  = predict_pspline(x_test, df=df_mean, beta=mean_fit["beta"])
    var_test = np.maximum(predict_pspline(x_test, df=df_var, beta=var_fit["beta"]), 1e-6)

    # ---- Ours interval on test (lambda_hat) ----
    lo_ours_z = mu_test - lam_hat * np.sqrt(var_test)
    up_ours_z = mu_test + lam_hat * np.sqrt(var_test)
    lo_ours = np.maximum(itf(lo_ours_z), 0.0)
    up_ours = itf(up_ours_z)

    # ---- GY interval on test (k(x)) ----
    # standardized mean on test
    mu_std_test = predict_pspline(x_test, df=df_std, beta=std_fit["beta"])
    # k(x) on test: interpolate from training points (visualization/benchmarking 목적)
    k_test = np.interp(x_test, x_all, k_all)

    lo_gy_std = mu_std_test - np.sqrt(est_var) * k_test
    up_gy_std = mu_std_test + np.sqrt(est_var) * k_test

    lo_gy_z = lo_gy_std * np.sqrt(var_test)
    up_gy_z = up_gy_std * np.sqrt(var_test)
    lo_gy = np.maximum(itf(lo_gy_z), 0.0)
    up_gy = itf(up_gy_z)

    # content + width
    ours_content = float(np.mean((y_test >= lo_ours) & (y_test <= up_ours)))
    gy_content   = float(np.mean((y_test >= lo_gy) & (y_test <= up_gy)))
    ours_width   = float(np.mean(up_ours - lo_ours))
    gy_width     = float(np.mean(up_gy - lo_gy))

    print(f"[FAIR seed={seed}] Ours: lambda={lam_hat:.6f} content={ours_content:.6f} width={ours_width:.6f}")
    print(f"[FAIR seed={seed}]   GY: df_eff={df_eff:.2f} nu={nu:.2f} est_var={est_var:.6f} content={gy_content:.6f} width={gy_width:.6f}")

    # ============================
    # Plots: overlay + diagnostics
    # ============================
    out_dir = root / "fig"
    out_dir.mkdir(exist_ok=True)

    # (1) overlay ribbon on test (sorted by x)
    order = np.argsort(x_test)
    xs = x_test[order]
    ys = y_test[order]
    loO, upO = lo_ours[order], up_ours[order]
    loG, upG = lo_gy[order], up_gy[order]

    plt.figure(figsize=(12, 5))
    plt.scatter(xs, ys, s=6, alpha=0.15)
    plt.fill_between(xs, loG, upG, alpha=0.25,
                     label=f"GY (content={gy_content:.3f}, width={gy_width:.3f})")
    plt.fill_between(xs, loO, upO, alpha=0.25,
                     label=f"Ours (content={ours_content:.3f}, width={ours_width:.3f})")
    plt.xlabel("mag_r")
    plt.ylabel("z_spec")
    plt.title(f"FAIR overlay on Happy B (seed={seed})")
    plt.legend()
    plt.tight_layout()
    overlay_path = out_dir / f"fair_overlay_ours_vs_gy_seed{seed}.pdf"
    plt.savefig(overlay_path)
    plt.close()

    # (2) diagnostics: mean/var fit (train) + r_cal histogram + k histogram
    x_min, x_max = np.percentile(x, [1, 99])
    x_grid = np.linspace(x_min, x_max, 400)

    mu_grid = predict_pspline(x_grid, df=df_mean, beta=mean_fit["beta"])
    var_grid = np.maximum(predict_pspline(x_grid, df=df_var, beta=var_fit["beta"]), 1e-6)

    r_cal = np.abs(z_cal - mu_cal) / np.sqrt(var_cal)

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(x_tr, z_tr, s=8, alpha=0.25)
    ax1.plot(x_grid, mu_grid, linewidth=2)
    ax1.set_title("Common mean fit: z ~ x (train)")
    ax1.set_xlabel("mag_r"); ax1.set_ylabel("z")

    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(x_tr, res2_tr, s=8, alpha=0.25)
    ax2.plot(x_grid, var_grid, linewidth=2)
    ax2.set_title("Common var fit: residual^2 ~ x (train)")
    ax2.set_xlabel("mag_r"); ax2.set_ylabel("var(z)")

    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(r_cal, bins=40, alpha=0.9)
    ax3.axvline(lam_hat, linestyle="--", linewidth=2)
    ax3.set_title("Ours: calibration r with lambda_hat")
    ax3.set_xlabel("r"); ax3.set_ylabel("count")

    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(k_all, bins=40, alpha=0.9)
    ax4.set_title("GY: k(x) distribution (train+cal points)")
    ax4.set_xlabel("k"); ax4.set_ylabel("count")

    plt.suptitle(f"FAIR diagnostics (seed={seed}, n_sample={n_sample})", y=1.02)
    plt.tight_layout()
    diag_path = out_dir / f"fair_diagnostics_seed{seed}.pdf"
    plt.savefig(diag_path)
    plt.close()

    print(f"Saved: {overlay_path}")
    print(f"Saved: {diag_path}")


if __name__ == "__main__":
    main()
