import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------
# Load results
# ----------------------------
df = pd.read_csv("results_happy_1d.csv")
df = df[df["is_failure"] == False].copy()


# method 순서 고정
order = ["HCTI", "Parametric TI"]
rename = {
    "HCTI": "HCTI",
    "Parametric TI": "Parametric TI",
}

df = df[df["method"].isin(order)].copy()
df["method_disp"] = df["method"].map(rename)

# ----------------------------
# Paper style
# ----------------------------
sns.set_style("white")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# 우리가 쓰는 색
palette = {
    "HCTI": "#1f77b4",        # blue
    "Parametric TI": "#d62728"  # red
}

# ----------------------------
# Figure: 1x2 panel
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=200)

# ---- Content ----
ax = axes[0]
sns.boxplot(
    data=df, x="method_disp", y="content",
    order=["HCTI", "Parametric TI"],
    palette=palette,
    width=0.5,
    showfliers=False,
    linewidth=1.0,
    ax=ax
)
sns.stripplot(
    data=df, x="method_disp", y="content",
    order=["HCTI", "Parametric TI"],
    color="black",
    size=2.5,
    alpha=0.55,
    jitter=0.18,
    ax=ax
)

# 🔹 target line (dashed)
ax.axhline(
    0.90,
    linestyle="--",      # 점선
    color="black",
    linewidth=1.2,
    zorder=0             # 박스 뒤로
)

ax.set_title("Content")
ax.set_ylabel("Empirical content")
ax.set_xlabel("")
ax.set_ylim(0.88, 1.00)   # 약간 여유 주면 보기 좋음

# ---- Width ----
ax = axes[1]

# 메인 박스/점 (메인은 outlier 제외하고 보기 좋게)
sns.boxplot(
    data=df, x="method_disp", y="mean_width",
    order=["HCTI", "Parametric TI"],
    palette=palette,
    width=0.5,
    showfliers=False,          # 메인에서는 outlier 숨김
    linewidth=1.0,
    ax=ax
)
sns.stripplot(
    data=df, x="method_disp", y="mean_width",
    order=["HCTI", "Parametric TI"],
    color="black",
    size=2.5,
    alpha=0.55,
    jitter=0.18,
    ax=ax
)

ax.set_title("Width")
ax.set_ylabel("Mean interval width")
ax.set_xlabel("")

ax.set_ylim(0.2, 0.7)

# -------------------------
# 🔹 Inset: Parametric TI의 extreme widths만 확대
# -------------------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# outlier 기준: Parametric TI에서 Q3 + 1.5*IQR 초과를 outlier로 정의
w_param = df.loc[df["method_disp"] == "Parametric TI", "mean_width"].dropna()
q1, q3 = w_param.quantile([0.25, 0.75])
iqr = q3 - q1
thr = q3 + 1.5 * iqr

w_out = w_param[w_param > thr]

# outlier가 실제로 있을 때만 inset을 그림 (없으면 깔끔하게 생략)
if len(w_out) > 0:
    axins = inset_axes(ax, width="42%", height="46%", loc="upper right", borderpad=0.8)

    # inset은 "Parametric TI outliers만" 찍기 (중복 점 문제 방지)
    df_out = df[(df["method_disp"] == "Parametric TI") & (df["mean_width"] > thr)].copy()
    df_out["method_disp"] = "Parametric TI"  # 단일 카테고리로 고정

    sns.stripplot(
        data=df_out, x="method_disp", y="mean_width",
        order=["Parametric TI"],
        color="black",
        size=3.0,
        alpha=0.75,
        jitter=0.02,
        ax=axins
    )

    # ✅ y-range: 데이터 기반 자동 설정
    y_min = float(w_out.min()) * 0.95
    y_max = float(w_out.max()) * 1.05
    axins.set_ylim(y_min, y_max)

    # ✅ inset 라벨/눈금 정리 (컴팩트)
    axins.set_title("Extreme widths ", fontsize=8, pad=2)
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.set_xticklabels([])          # x축 라벨 제거
    axins.tick_params(axis="x", length=0)
    axins.tick_params(labelsize=7)

    # inset 테두리 깔끔
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)

# ----------------------------

# 논문용 마감
for ax in axes:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

fig.tight_layout()
fig.savefig("fig_happy_1d_boxplots.pdf", bbox_inches="tight")
fig.savefig("fig_happy_1d_boxplots.png", bbox_inches="tight")

print("Saved: fig_happy_1d_boxplots.pdf / .png")
