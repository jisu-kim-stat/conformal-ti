import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load results
# ----------------------------
df = pd.read_csv("results_happy_1d.csv")

# method 순서 고정 (논문용)
order = ["HCTI(1D mag_r)", "Parametric TI(1D mag_r)"]

# ----------------------------
# Plot style 
# ----------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ----------------------------
# Figure 1: Content (Coverage)
# ----------------------------
plt.figure(figsize=(6, 4))

sns.boxplot(
    data=df,
    x="method",
    y="content",
    order=order,
    width=0.5,
    showfliers=False
)

plt.axhline(0.90, linestyle="--", color="black", linewidth=1)

plt.xlabel("")
plt.ylabel("Empirical content")
plt.title("Happy dataset (1D): Content comparison")

plt.tight_layout()
plt.savefig("fig_happy_1d_content_boxplot.pdf")
plt.close()

# ----------------------------
# Figure 2: Mean width
# ----------------------------
plt.figure(figsize=(6, 4))

sns.boxplot(
    data=df,
    x="method",
    y="mean_width",
    order=order,
    width=0.5,
    showfliers=False
)

plt.xlabel("")
plt.ylabel("Mean interval width")
plt.title("Happy dataset (1D): Width comparison")

plt.tight_layout()
plt.savefig("fig_happy_1d_width_boxplot.pdf")
plt.close()

print("Saved:")
print("- fig_happy_1d_content_boxplot.pdf")
print("- fig_happy_1d_width_boxplot.pdf")
