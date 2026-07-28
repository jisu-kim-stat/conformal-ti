from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHODS = ["Parametric TI", "SR-TI", "ASR-TI", "CQR-TI"]

METHOD_LABELS = {
    "Parametric TI": "Parametric TI",
    "SR-TI": "SR-TI",
    "ASR-TI": "ASR-TI",
    "CQR-TI": "CQR-TI",
}

METHOD_ORDER = {m: i for i, m in enumerate(METHODS)}

LEGACY_METHOD_NAMES = {
    "HCTI": "SR-TI",
    "HCTI-asym": "ASR-TI",
}


def load_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"method", "seed", "content", "mean_width"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary csv: {sorted(missing)}")

    df["method"] = df["method"].replace(LEGACY_METHOD_NAMES)
    df = df[df["method"].isin(METHODS)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)

    if "median_width" not in df.columns:
        df["median_width"] = np.nan
    if "q90_width" not in df.columns:
        df["q90_width"] = np.nan

    return df.sort_values(["method", "seed"]).reset_index(drop=True)


def load_intervals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"seed", "method", "y", "lower", "upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in intervals csv: {sorted(missing)}")

    df["method"] = df["method"].replace(LEGACY_METHOD_NAMES)
    df = df[df["method"].isin(METHODS)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
    df["seed"] = df["seed"].astype(int)

    for c in ["y", "lower", "upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["seed", "method", "y", "lower", "upper"]).copy()
    return df.sort_values(["method", "seed"]).reset_index(drop=True)


def theme_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9)


def savefig(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_summary_layer(
    df: pd.DataFrame,
    out_path: str | Path,
    content_level: float = 0.90,
):
    """
    Paper-style real-data summary:
      Panel (a): empirical content over repeated splits
      Panel (b): average interval width over repeated splits
    """
    methods = METHODS
    x = np.arange(len(methods))

    content_mean = df.groupby("method", observed=True)["content"].mean().reindex(methods)
    content_sd = df.groupby("method", observed=True)["content"].std().reindex(methods)

    width_mean = df.groupby("method", observed=True)["mean_width"].mean().reindex(methods)
    width_sd = df.groupby("method", observed=True)["mean_width"].std().reindex(methods)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    ax = axes[0]
    ax.errorbar(
        x,
        content_mean.to_numpy(),
        yerr=content_sd.to_numpy(),
        fmt="o",
        capsize=4,
        linewidth=1.4,
        markersize=5,
    )
    ax.axhline(content_level, linestyle="--", color="0.35", linewidth=1.0)
    ax.set_ylabel("Empirical content")
    ax.set_ylim(0, 1.03)
    ax.set_title("TSA data: empirical content")
    theme_axes(ax)

    ax = axes[1]
    ax.errorbar(
        x,
        width_mean.to_numpy(),
        yerr=width_sd.to_numpy(),
        fmt="o",
        capsize=4,
        linewidth=1.4,
        markersize=5,
    )
    ax.set_ylabel("Average interval width")
    ax.set_title("TSA data: average interval width")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=20, ha="right")
    theme_axes(ax)

    fig.suptitle("Real-data performance on TSA passenger throughput", fontsize=13, fontweight="bold")
    savefig(out_path)


def compute_pointwise_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each method/date, compute:
      coverage = mean over seeds of 1{Y in interval}
      width    = mean over seeds of interval width
    """
    d = df.copy()
    d["hit"] = ((d["y"] >= d["lower"]) & (d["y"] <= d["upper"])).astype(float)
    d["width"] = d["upper"] - d["lower"]

    if "date" in d.columns and d["date"].notna().any():
        d["x"] = d["date"]
    else:
        d = d.sort_values(["seed", "method"]).copy()
        d["x"] = d.groupby(["seed", "method"]).cumcount()

    out = (
        d.groupby(["method", "x"], observed=True, as_index=False)
        .agg(
            coverage=("hit", "mean"),
            width=("width", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "x"])
        .reset_index(drop=True)
    )
    return out


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def plot_pointwise_layer(
    intervals: pd.DataFrame,
    out_path: str | Path,
    content_level: float = 0.90,
    window: int = 14,
):
    """
    Paper-style real-data pointwise diagnostic:
      Panel (a): rolling empirical coverage over dates
      Panel (b): rolling interval width over dates
    """
    pw = compute_pointwise_stats(intervals)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)

    for method in METHODS:
        d = pw[pw["method"] == method].copy()
        if d.empty:
            continue
        x = d["x"].to_numpy()
        cov = rolling_mean(d["coverage"].to_numpy(dtype=float), window)
        wid = rolling_mean(d["width"].to_numpy(dtype=float), window)

        axes[0].plot(x, cov, linewidth=1.6, label=METHOD_LABELS[method])
        axes[1].plot(x, wid, linewidth=1.6, label=METHOD_LABELS[method])

    axes[0].axhline(content_level, linestyle="--", color="0.35", linewidth=1.0)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title(f"Rolling empirical coverage, window={window}")
    theme_axes(axes[0])

    axes[1].set_ylabel("Interval width")
    axes[1].set_title(f"Rolling interval width, window={window}")
    theme_axes(axes[1])

    if pd.api.types.is_datetime64_any_dtype(pw["x"]):
        axes[1].set_xlabel("Date")
    else:
        axes[1].set_xlabel("Index")

    axes[0].legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("TSA data: local empirical behavior", fontsize=13, fontweight="bold")
    savefig(out_path)


def choose_representative_seed(summary: pd.DataFrame) -> int:
    """
    Pick a seed whose SR-TI content is closest to the median SR-TI content.
    This avoids cherry-picking an unusually good/bad split.
    """
    d = summary[summary["method"] == "SR-TI"].copy()
    if d.empty:
        return int(summary["seed"].iloc[0])
    med = d["content"].median()
    idx = (d["content"] - med).abs().idxmin()
    return int(d.loc[idx, "seed"])


def plot_intervals_one_seed(
    intervals: pd.DataFrame,
    seed: int,
    out_path: str | Path,
):
    """
    Visualize actual prediction intervals for one representative split.
    This is useful for appendix or real-data illustration.
    """
    d0 = intervals[intervals["seed"] == seed].copy()
    if d0.empty:
        raise ValueError(f"No rows for seed={seed}")

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(11.5, 2.25 * len(METHODS)), sharex=True)
    if len(METHODS) == 1:
        axes = [axes]

    for ax, method in zip(axes, METHODS):
        d = d0[d0["method"] == method].copy()
        if d.empty:
            ax.set_title(f"{method} (no data)")
            continue

        if "date" in d.columns and d["date"].notna().any():
            d = d.sort_values("date")
            x = d["date"].to_numpy()
        else:
            d = d.reset_index(drop=True)
            x = np.arange(len(d))

        y = d["y"].to_numpy(dtype=float)
        lo = d["lower"].to_numpy(dtype=float)
        hi = d["upper"].to_numpy(dtype=float)

        ax.plot(x, y, linewidth=1.0, label="Observed")
        ax.fill_between(x, lo, hi, alpha=0.22, label="Interval")
        ax.plot(x, lo, linewidth=0.7)
        ax.plot(x, hi, linewidth=0.7)

        hit = np.mean((y >= lo) & (y <= hi))
        width = np.mean(hi - lo)
        ax.set_title(f"{METHOD_LABELS[method]} | content={hit:.3f}, width={width:,.0f}")
        ax.set_ylabel("Throughput")
        theme_axes(ax)

    axes[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.03))

    if "date" in d0.columns and d0["date"].notna().any():
        axes[-1].set_xlabel("Date")
    else:
        axes[-1].set_xlabel("Index")

    fig.suptitle(f"TSA intervals for representative split, seed={seed}", fontsize=13, fontweight="bold")
    savefig(out_path)


def save_latex_table(summary: pd.DataFrame, out_path: str | Path):
    g = (
        summary.groupby("method", observed=True)
        .agg(
            content_mean=("content", "mean"),
            content_sd=("content", "std"),
            width_mean=("mean_width", "mean"),
            width_sd=("mean_width", "std"),
            median_width=("median_width", "mean"),
            q90_width=("q90_width", "mean"),
        )
        .reindex(METHODS)
    )

    lines = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Content & Content SD & Mean width & Q90 width \\")
    lines.append(r"\midrule")
    for method, row in g.iterrows():
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{row['content_mean']:.3f} & "
            f"{row['content_sd']:.3f} & "
            f"{row['width_mean']:,.0f} & "
            f"{row['q90_width']:,.0f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="results_tsa_4methods.csv")
    ap.add_argument("--intervals", type=str, default="results_tsa_4methods_intervals.csv")
    ap.add_argument("--out_dir", type=str, default="fig/real/tsa")
    ap.add_argument("--content_level", type=float, default=0.90)
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--seed", type=int, default=None, help="seed for interval visualization")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    summary = load_summary(args.summary)
    intervals = load_intervals(args.intervals)

    plot_summary_layer(
        summary,
        out_dir / "fig_tsa_summary_layer.png",
        content_level=args.content_level,
    )

    plot_pointwise_layer(
        intervals,
        out_dir / "fig_tsa_pointwise_layer.png",
        content_level=args.content_level,
        window=args.window,
    )

    seed = args.seed if args.seed is not None else choose_representative_seed(summary)

    plot_intervals_one_seed(
        intervals,
        seed=seed,
        out_path=out_dir / f"fig_tsa_intervals_seed{seed}.png",
    )

    save_latex_table(
        summary,
        out_dir / "table_tsa_summary.tex",
    )


if __name__ == "__main__":
    main()
