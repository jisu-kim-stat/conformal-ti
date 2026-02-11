import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METHODS = ["HCTI-hetero", "HCTI-homo", "HCTI-hetero(PS)", "PTI-homo", "PTI-hetero"]
PAPER_MAIN_METHODS = ["HCTI-hetero", "PTI-hetero"]


def load_intervals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    req = {"seed", "method", "y", "lower", "upper"}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in intervals csv: {missing}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["y", "lower", "upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seed", "method", "y", "lower", "upper"]).copy()
    df["seed"] = df["seed"].astype(int)
    df["method"] = df["method"].astype(str)
    return df


def compute_pointwise_stats(
    df: pd.DataFrame,
    methods: list[str],
    x_col: str = "date",
) -> pd.DataFrame:
    """
    Returns tidy df with columns:
      - method
      - x (date or index)
      - coverage (mean over seeds of indicator)
      - width (mean over seeds of interval width)
      - n_seeds (how many seeds contributed)
    """
    d = df.copy()

    d["hit"] = ((d["y"] >= d["lower"]) & (d["y"] <= d["upper"])).astype(float)
    d["width"] = (d["upper"] - d["lower"]).astype(float)

    if x_col == "date" and "date" in d.columns and d["date"].notna().any():
        d["x"] = d["date"]
        use_date = True
    else:
        d = d.sort_values(["seed", "method"] + (["date"] if "date" in d.columns else []))
        d["x"] = d.groupby(["seed", "method"]).cumcount()
        use_date = False

    d = d[d["method"].isin(methods)].copy()

    out = (
        d.groupby(["method", "x"], as_index=False)
        .agg(
            coverage=("hit", "mean"),
            width=("width", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "x"])
        .reset_index(drop=True)
    )
    out.attrs["use_date"] = use_date
    return out


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def _xlabel_from_pw(pw: pd.DataFrame) -> str:
    x_is_date = pd.api.types.is_datetime64_any_dtype(pw["x"])
    return "Date" if x_is_date else "Index"


def _target_line(ax_or_plt, target: float):
    # gray thin dashed line
    ax_or_plt.axhline(target, linestyle="--", linewidth=0.8, color="0.5", alpha=0.8, zorder=0)


def plot_overlay(
    pw: pd.DataFrame,
    methods: list[str],
    out_path: str | Path,
    metric: str,  # "coverage" or "width"
    target: float = 0.90,   # used only if metric == "coverage"
    window: int = 1,
    title: str | None = None,
    appendix_mode: bool = False,
):
    """
    Overlay plot for either coverage or width.
    - paper (appendix_mode=False): expects 2 methods (HCTI-hetero, PTI-hetero), colored by group.
    - appendix (appendix_mode=True): one color per method (all solid lines), clean legend.
    """
    assert metric in {"coverage", "width"}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xlabel = _xlabel_from_pw(pw)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(color_cycle) == 0:
        color_cycle = [None]

    fig = plt.figure(figsize=(11, 4.6))

    if appendix_mode:
        # one color per method
        if len(color_cycle) < len(methods):
            color_cycle = (color_cycle * (len(methods) // len(color_cycle) + 1))[: len(methods)]
        method_color = {m: color_cycle[i] for i, m in enumerate(methods)}

        for m in methods:
            d = pw[pw["method"] == m].copy()
            if len(d) == 0:
                continue
            x = d["x"].to_numpy()
            y = d[metric].to_numpy(dtype=float)
            y_sm = rolling_mean(y, window=window)

            plt.plot(
                x, y_sm,
                label=m,
                linewidth=1.8,
                linestyle="-",
                color=method_color.get(m, None),
                alpha=0.95,
            )

        plt.legend(frameon=False, ncol=1, fontsize=9)

    else:
        # paper: color by group, solid lines only
        def group_of(m: str) -> str:
            if m.startswith("HCTI"):
                return "HCTI"
            if m.startswith("PTI"):
                return "PTI"
            return "OTHER"

        c_hcti = color_cycle[0]
        c_pti = color_cycle[1] if len(color_cycle) > 1 else color_cycle[0]
        group_color = {"HCTI": c_hcti, "PTI": c_pti, "OTHER": None}

        label_map = {"HCTI-hetero": "HCTI (hetero)", "PTI-hetero": "PTI (hetero)"}

        for m in methods:
            d = pw[pw["method"] == m].copy()
            if len(d) == 0:
                continue
            x = d["x"].to_numpy()
            y = d[metric].to_numpy(dtype=float)
            y_sm = rolling_mean(y, window=window)

            plt.plot(
                x, y_sm,
                label=label_map.get(m, m),
                linewidth=2.4,
                linestyle="-",
                color=group_color.get(group_of(m), None),
                alpha=1.0,
            )

        plt.legend(frameon=False)

    if metric == "coverage":
        _target_line(plt, target)

    plt.xlabel(xlabel)
    plt.ylabel("Coverage" if metric == "coverage" else "Interval width")

    if title is None:
        base = "Rolling pointwise coverage" if metric == "coverage" else "Rolling pointwise width"
        title = base + (f" (w={window})" if window and window > 1 else "")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_5in1(
    pw: pd.DataFrame,
    methods: list[str],
    out_path: str | Path,
    metric: str,  # "coverage" or "width"
    target: float = 0.90,   # used only if metric == "coverage"
    window: int = 1,
    title: str | None = None,
):
    """Appendix-style: separate panel per method (clean)."""
    assert metric in {"coverage", "width"}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    xlabel = _xlabel_from_pw(pw)

    for ax, m in zip(axes, methods):
        d = pw[pw["method"] == m].copy()
        if metric == "coverage":
            _target_line(ax, target)

        if len(d) == 0:
            ax.set_title(f"{m} (no data)")
            ax.set_ylim(0, 1 if metric == "coverage" else None)
            ax.set_ylabel("Coverage" if metric == "coverage" else "Width")
            continue

        x = d["x"].to_numpy()
        y = d[metric].to_numpy(dtype=float)
        y_sm = rolling_mean(y, window=window)

        ax.plot(x, y_sm, linewidth=1.6)
        if metric == "coverage":
            ax.set_ylim(0, 1)
        ax.set_ylabel("Coverage" if metric == "coverage" else "Width")
        ax.set_title(m)

    axes[-1].set_xlabel(xlabel)

    if title is None:
        base = "Rolling pointwise coverage" if metric == "coverage" else "Rolling pointwise width"
        title = base + (f" (w={window})" if window and window > 1 else "")
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", type=str, required=True, help="*_intervals.csv (long format)")
    ap.add_argument("--out", type=str, required=True, help="output png path (paper/main base name)")
    ap.add_argument("--out_csv", type=str, default=None, help="optional: save pointwise stats csv")
    ap.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="comma-separated list for appendix plots",
    )
    ap.add_argument("--target", type=float, default=0.90, help="target content/coverage line")
    ap.add_argument("--window", type=int, default=1, help="rolling mean window (>=1)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    main_methods = PAPER_MAIN_METHODS

    df = load_intervals(args.intervals)
    pw = compute_pointwise_stats(df, methods=methods, x_col="date")

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pw.to_csv(out_csv, index=False)
        print(f"Saved pointwise csv: {out_csv}")

    out_main = Path(args.out)

    # ----- MAIN (paper): hetero only (coverage + width) -----
    plot_overlay(
        pw, methods=main_methods,
        out_path=out_main.with_name(out_main.stem + "_cov" + out_main.suffix),
        metric="coverage",
        target=args.target,
        window=args.window,
        appendix_mode=False,
    )
    plot_overlay(
        pw, methods=main_methods,
        out_path=out_main.with_name(out_main.stem + "_width" + out_main.suffix),
        metric="width",
        window=args.window,
        appendix_mode=False,
    )

    # ----- APPENDIX overlay: all methods (coverage + width) -----
    plot_overlay(
        pw, methods=methods,
        out_path=out_main.with_name(out_main.stem + "_appendix_cov_overlay" + out_main.suffix),
        metric="coverage",
        target=args.target,
        window=args.window,
        appendix_mode=True,
    )
    plot_overlay(
        pw, methods=methods,
        out_path=out_main.with_name(out_main.stem + "_appendix_width_overlay" + out_main.suffix),
        metric="width",
        window=args.window,
        appendix_mode=True,
    )

    # ----- APPENDIX panels: 5-in-1 (optional) -----
    plot_5in1(
        pw, methods=methods,
        out_path=out_main.with_name(out_main.stem + "_appendix_cov_5in1" + out_main.suffix),
        metric="coverage",
        target=args.target,
        window=args.window,
    )
    plot_5in1(
        pw, methods=methods,
        out_path=out_main.with_name(out_main.stem + "_appendix_width_5in1" + out_main.suffix),
        metric="width",
        window=args.window,
    )


if __name__ == "__main__":
    main()
