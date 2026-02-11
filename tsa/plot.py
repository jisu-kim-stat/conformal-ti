import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_intervals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # date가 있으면 datetime으로
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def plot_one(df: pd.DataFrame, seed: int, method: str, out: str | None = None):
    d = df[(df["seed"] == seed) & (df["method"] == method)].copy()
    if len(d) == 0:
        raise ValueError(f"No rows for seed={seed}, method={method}")

    # x축: date 있으면 date, 없으면 index
    if "date" in d.columns and d["date"].notna().any():
        d = d.sort_values("date")
        x = d["date"].to_numpy()
        xlabel = "date"
    else:
        d = d.reset_index(drop=True)
        x = np.arange(len(d))
        xlabel = "index"

    y = d["y"].to_numpy(dtype=float)
    lo = d["lower"].to_numpy(dtype=float)
    hi = d["upper"].to_numpy(dtype=float)

    fig = plt.figure()
    plt.plot(x, y, linewidth=1.0, label="y")
    plt.fill_between(x, lo, hi, alpha=0.25, label="interval")
    plt.plot(x, lo, linewidth=0.8, label="lower")
    plt.plot(x, hi, linewidth=0.8, label="upper")

    plt.title(f"Intervals | seed={seed} | method={method}")
    plt.xlabel(xlabel)
    plt.ylabel("throughput")
    plt.legend()
    plt.tight_layout()

    if out is not None:
        out = str(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved: {out}")
    else:
        plt.show()


def plot_seed_all_methods(df: pd.DataFrame, seed: int, methods: list[str] | None, out: str | None = None):
    ds = df[df["seed"] == seed].copy()
    if len(ds) == 0:
        raise ValueError(f"No rows for seed={seed}")

    if methods is None or len(methods) == 0:
        methods = sorted(ds["method"].unique().tolist())

    # x축 기준(모든 method 공통 date라고 가정)
    if "date" in ds.columns and ds["date"].notna().any():
        xlabel = "date"
    else:
        xlabel = "index"

    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.7 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, methods):
        d = ds[ds["method"] == m].copy()
        if len(d) == 0:
            ax.set_title(f"{m} (no data)")
            ax.set_ylabel("throughput")
            continue

        if xlabel == "date":
            d = d.sort_values("date")
            x = d["date"].to_numpy()
        else:
            d = d.reset_index(drop=True)
            x = np.arange(len(d))

        y = d["y"].to_numpy(dtype=float)
        lo = d["lower"].to_numpy(dtype=float)
        hi = d["upper"].to_numpy(dtype=float)

        ax.plot(x, y, linewidth=1.0, label="y")
        ax.fill_between(x, lo, hi, alpha=0.25, label="interval")
        ax.set_title(f"seed={seed} | {m}")
        ax.set_ylabel("throughput")
        ax.legend()

    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved: {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="*_intervals.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default=None, help="png path (optional)")
    args = ap.parse_args()

    df = load_intervals(args.csv)

    methods = [
        "HCTI-hetero",
        "HCTI-homo",
        "HCTI-hetero(PS)",
        "PTI-homo",
        "PTI-hetero",
    ]

    # out을 안 주면 기본 저장 경로를 seed 기준으로 생성
    out_path = args.out
    if out_path is None:
        out_path = f"plots/seed{args.seed}_all_methods.png"

    plot_seed_all_methods(
        df,
        seed=args.seed,
        methods=methods,
        out=out_path,
    )


if __name__ == "__main__":
    main()
