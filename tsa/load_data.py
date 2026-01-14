import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
URL = "https://raw.githubusercontent.com/hunj/tsa-passenger-throughput/main/output.csv"
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(
    URL,
    header=None,
    names=["date", "throughput"]
)

# 타입 정리
df["date"] = pd.to_datetime(df["date"])
df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")

df = df.dropna().sort_values("date").reset_index(drop=True)

# ----------------------------
# Basic inspection
# ----------------------------
print("Head:")
print(df.head(10))
print("\nTail:")
print(df.tail(10))
print("\nDescribe:")
print(df.describe())
print("\nInfo:")
print(df.info())

# ----------------------------
# Plot: full time series
# ----------------------------
plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["throughput"], linewidth=1)
plt.xlabel("Date")
plt.ylabel("TSA passenger throughput")
plt.title("Daily TSA Passenger Throughput")
plt.tight_layout()

out_path = OUT_DIR / "tsa_throughput_timeseries.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {out_path.resolve()}")

# ----------------------------
# Optional: log / asinh transformed views (for diagnosis)
# ----------------------------
plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["throughput"].apply(lambda x: pd.NA if x <= 0 else x).astype(float), linewidth=1)
plt.yscale("log")
plt.xlabel("Date")
plt.ylabel("log-scale throughput")
plt.title("TSA Passenger Throughput (log scale)")
plt.tight_layout()

out_path_log = OUT_DIR / "tsa_throughput_logscale.png"
plt.savefig(out_path_log, dpi=150)
plt.close()

print(f"Saved log-scale plot to: {out_path_log.resolve()}")
