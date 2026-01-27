import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METHODS = ["Ours", "Ours-homo", "Ours-PS-hetero", "GY-homo", "GY-hetero"]


def load_intervals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # required columns
    req = {"seed", "method", "y", "lower", "upper"}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in intervals csv: {missing}")

    # date optional
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # numeric
    for c in ["y", "lower", "upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seed", "method", "y", "lower", "upper"]).copy()
    df["seed"] = df["seed"].astype(int)
    df["method"] = df["method"].astype(str)

    return df


def compute_pointwise_coverage(
    df: pd.DataFrame,
    methods: list[str],
    x_col: str = "date",
) -> pd.DataFrame:
    """
    Returns tidy df with columns:
      - method
      - x (date or index)
      - coverage (mean over seeds of indicator)
      - n_seeds (how many seeds contributed)
    """
    d = df.copy()

    # indicator
    d["hit"] = ((d["y"] >= d["lower"]) & (d["y"] <= d["upper"])).astype(float)

    # choose x-axis
    if x_col == "date" and "date" in d.columns and d["date"].notna().any():
        d["x"] = d["date"]
        use_date = True
    else:
        # fallback: build within (seed, method) time index
        # assumes each (seed, method) has same ordering
        d = d.sort_values(["seed", "method"] + (["date"] if "date" in d.columns else []))
        d["x"] = d.groupby(["seed", "method"]).cumcount()
        use_date = False

    # filter methods
    d = d[d["method"].isin(methods)].copy()

    # aggregate over seeds per (method, x)
    out = (
        d.groupby(["method", "x"], as_index=False)
        .agg(coverage=("hit", "mean"), n_seeds=("hit", "size"))
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


def plot_pointwise_coverage_5in1(
    pw: pd.DataFrame,
    methods: list[str],
    out_path: str | Path,
    target: float = 0.90,
    window: int = 1,
    title: str | None = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, methods):
        d = pw[pw["method"] == m].copy()
        if len(d) == 0:
            ax.set_title(f"{m} (no data)")
            ax.set_ylim(0, 1)
            ax.axhline(target, linestyle="--", linewidth=0.8)
            continue

        x = d["x"].to_numpy()
        y = d["coverage"].to_numpy(dtype=float)
        y_sm = rolling_mean(y, window=window)

        ax.plot(x, y_sm, linewidth=1.2, label=f"{m}")
        ax.axhline(target, linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("coverage")
        ax.legend(loc="upper right")

        # show how many seeds contributed (optional small text)
        # n_seeds may vary if rows missing; show median
        med_n = int(np.median(d["n_seeds"].to_numpy()))
        ax.set_title(f"{m} (seeds per x ~ {med_n})")

    axes[-1].set_xlabel("date" if isinstance(pw["x"].iloc[0], pd.Timestamp) else "index")

    if title is None:
        title = "Pointwise coverage over seeds (TSA)"
        if window and window > 1:
            title += f" | rolling window={window}"
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", type=str, required=True, help="*_intervals.csv (long format)")
    ap.add_argument("--out", type=str, required=True, help="output png path (single figure)")
    ap.add_argument("--out_csv", type=str, default=None, help="optional: save pointwise coverage csv")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                    help="comma-separated list")
    ap.add_argument("--target", type=float, default=0.90, help="target content/coverage line")
    ap.add_argument("--window", type=int, default=1, help="rolling mean window (>=1)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    df = load_intervals(args.intervals)
    pw = compute_pointwise_coverage(df, methods=methods, x_col="date")

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pw.to_csv(out_csv, index=False)
        print(f"Saved pointwise csv: {out_csv}")

    plot_pointwise_coverage_5in1(
        pw,
        methods=methods,
        out_path=args.out,
        target=args.target,
        window=args.window,
        title=None,
    )


if __name__ == "__main__":
    main()
