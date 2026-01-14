#!/usr/bin/env Rscript

# ============================================================
# scripts/make_plot.R
# - Read pointwise summary CSV produced by run_simulation_grid
# - Create comparison plots: Ours vs GY
#
# Input (default):
#   results/sim/models/pointwise_df_hoeffding.csv
# Output (default dir):
#   results/sim/models/plots/
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
  library(stringr)
})

# ---------------------------
# Config (edit if needed)
# ---------------------------
in_path  <- "results/sim/models/pointwise_df_ours_vs_gy.csv"
out_dir  <- "results/sim/models/plots"
confidence_level <- 0.95

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("[make_plot] input:", in_path, "\n")
cat("[make_plot] output dir:", out_dir, "\n")

# ---------------------------
# Read
# ---------------------------
df <- readr::read_csv(in_path, show_col_types = FALSE)

required_cols <- c("x","coverage","mean_width","na_proportion","model","n","Method")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Input CSV missing columns: ", paste(missing_cols, collapse = ", "))
}

# Basic cleaning / types
df <- df %>%
  mutate(
    model = as.integer(model),
    n     = as.integer(n),
    Method = as.character(Method)
  )

# Order methods (optional)
method_levels <- intersect(c("Ours","GY"), unique(df$Method))
df$Method <- factor(df$Method, levels = method_levels)

# ---------------------------
# Summary across x (per model,n,Method)
# ---------------------------
df_summary <- df %>%
  group_by(model, n, Method) %>%
  summarise(
    avg_coverage   = mean(coverage, na.rm = TRUE),
    avg_width      = mean(mean_width, na.rm = TRUE),
    avg_na_prop    = mean(na_proportion, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================
# Plot 1: average coverage vs n (per model)
# ============================================================
p_cov <- ggplot(df_summary, aes(x = n, y = avg_coverage, group = Method, color = Method)) +
  geom_line(linewidth = 1.0) +
  geom_hline(yintercept = confidence_level, linetype = "dashed", color = "black") +
  geom_point(size = 2.0) +
  facet_wrap(~ model, scales = "fixed") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Average pointwise coverage (mean over x)",
    x = "Sample size (n)",
    y = "Average coverage",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(out_dir, "avg_coverage_vs_n.png"),
  plot = p_cov,
  width = 12, height = 7, dpi = 300
)

# ============================================================
# Plot 2: average width vs n (per model)
# ============================================================
p_wid <- ggplot(df_summary, aes(x = n, y = avg_width, group = Method, color = Method)) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.0) +
  facet_wrap(~ model, scales = "free_y") +
  labs(
    title = "Average interval width (mean over x)",
    x = "Sample size (n)",
    y = "Average width",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(out_dir, "avg_width_vs_n.png"),
  plot = p_wid,
  width = 12, height = 7, dpi = 300
)

# ============================================================
# Plot 3: pointwise coverage curve (x-wise)
# facet: model x n, color: Method
# ============================================================
# If many facets, you may want to restrict sample sizes or increase output size.
p_pt_cov <- ggplot(df, aes(x = x, y = coverage, group = Method, color = Method)) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = confidence_level, linetype = "dashed", linewidth = 0.6) +
  facet_grid(model ~ n, labeller = label_both) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Pointwise coverage curve",
    subtitle = paste0("Dashed line: confidence level = ", confidence_level),
    x = "x",
    y = "P(content ≥ 0.90)",
    color = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(out_dir, "pointwise_coverage_curve.png"),
  plot = p_pt_cov,
  width = 16, height = 10, dpi = 300
)