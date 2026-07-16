from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHODS = ["Parametric TI", "HCTI", "HCTI-asym", "CQR-TI"]

METHOD_COLORS = {
    "HCTI": "#D55E00",
    "HCTI-asym": "#CC79A7",
    "CQR-TI": "#0072B2",
    "Parametric TI": "#555555",
}


def load_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"method", "seed", "content", "mean_width"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary csv: {sorted(missing)}")

    df = df[df["method"].isin(METHODS)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)

    if "q90_width" not in df.columns:
        df["q90_width"] = np.nan
    if "median_width" not in df.columns:
        df["median_width"] = np.nan

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


def jitter_x(x, n, scale=0.055, seed=123):
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, scale, size=n)


def plot_happy_split_distribution(
    df: pd.DataFrame,
    out_path: str | Path,
    content_level: float = 0.90,
):
    methods = METHODS
    x_base = np.arange(len(methods))

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=True)

    for j, method in enumerate(methods):
        d = df[df["method"] == method].copy()
        color = METHOD_COLORS[method]
        x = jitter_x(x_base[j], len(d), seed=100 + j)

        # Panel (a): content, individual splits
        axes[0].scatter(
            x,
            d["content"],
            s=18,
            color=color,
            alpha=0.45,
            edgecolor="none",
        )

        # Mean and +/- 1 sd
        m = d["content"].mean()
        s = d["content"].std()
        axes[0].errorbar(
            x_base[j],
            m,
            yerr=s,
            fmt="o",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            capsize=4,
            linewidth=1.2,
            zorder=5,
        )

        # Panel (b): width, individual splits
        axes[1].scatter(
            x,
            d["mean_width"],
            s=18,
            color=color,
            alpha=0.45,
            edgecolor="none",
        )

        m = d["mean_width"].mean()
        s = d["mean_width"].std()
        axes[1].errorbar(
            x_base[j],
            m,
            yerr=s,
            fmt="o",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            capsize=4,
            linewidth=1.2,
            zorder=5,
        )

    axes[0].axhline(content_level, linestyle="--", color="0.35", linewidth=1.0)
    axes[0].set_ylabel("Empirical content")
    axes[0].set_title("Empirical content over repeated splits")
    axes[0].set_ylim(0.91, 0.94)
    theme_axes(axes[0])

    axes[1].set_ylabel("Average interval width")
    axes[1].set_title("Average interval width over repeated splits")
    axes[1].set_xticks(x_base)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    theme_axes(axes[1])

    fig.suptitle(
        "Happy photometric redshift data",
        fontsize=13,
        fontweight="bold",
    )

    savefig(out_path)


def plot_happy_content_width_scatter(
    df: pd.DataFrame,
    out_path: str | Path,
    content_level: float = 0.90,
):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for method in METHODS:
        d = df[df["method"] == method].copy()
        ax.scatter(
            d["mean_width"],
            d["content"],
            s=28,
            alpha=0.55,
            color=METHOD_COLORS[method],
            label=method,
            edgecolor="none",
        )

        ax.scatter(
            d["mean_width"].mean(),
            d["content"].mean(),
            s=95,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

    ax.axhline(content_level, linestyle="--", color="0.35", linewidth=1.0)
    ax.set_xlabel("Average interval width")
    ax.set_ylabel("Empirical content")
    ax.set_title("Happy data: content-width tradeoff")
    ax.legend(frameon=False, ncol=2)
    theme_axes(ax)

    savefig(out_path)


def save_latex_table(df: pd.DataFrame, out_path: str | Path):
    g = (
        df.groupby("method", observed=True)
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
            f"{method} & "
            f"{row['content_mean']:.3f} & "
            f"{row['content_sd']:.3f} & "
            f"{row['width_mean']:.3f} & "
            f"{row['q90_width']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="results_happy_4methods.csv")
    ap.add_argument("--out_dir", type=str, default="fig/real/happy")
    ap.add_argument("--content_level", type=float, default=0.90)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    df = load_summary(args.summary)

    plot_happy_split_distribution(
        df,
        out_dir / "fig_happy_split_distribution.png",
        content_level=args.content_level,
    )

    plot_happy_content_width_scatter(
        df,
        out_dir / "fig_happy_content_width_scatter.png",
        content_level=args.content_level,
    )

    save_latex_table(
        df,
        out_dir / "table_happy_summary.tex",
    )


if __name__ == "__main__":
    main()