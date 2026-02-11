#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
  library(stringr)
})

# ---------------------------
# Config
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
    model  = as.integer(model),
    n      = as.integer(n),
    Method = as.character(Method)
  )

# Fail fast if empty
if (nrow(df) == 0) stop("Input CSV has 0 rows: ", in_path)

# ---------------------------
# Paper-style aesthetics
# ---------------------------
# 1) Fix method order & labels (논문에서 보이는 이름)
method_levels <- c("HCTI", "Parametric TI")
method_labels <- c("HCTI" = "HCTI", "Parametric TI" = "Parametric TI")

# CSV에 둘 중 일부만 있어도 factor 레벨은 고정해두자 (palette(0) 방지)
df$Method <- factor(df$Method, levels = method_levels)

# 2) Manual scales (색/선 고정, 드롭 금지)
method_colors   <- c("HCTI" = "#1f77b4", "Parametric TI" = "#d62728")
method_linetype <- c("HCTI" = "solid", "Parametric TI" = "dashed")

scale_method_color <- scale_color_manual(
  values = method_colors, limits = method_levels, breaks = method_levels,
  labels = method_labels, drop = FALSE
)
scale_method_linetype <- scale_linetype_manual(
  values = method_linetype, limits = method_levels, breaks = method_levels,
  labels = method_labels, drop = FALSE
)

# 3) A clean theme for papers
theme_paper <- theme_classic(base_size = 12) +
  theme(
    # legend
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.margin = margin(t = 2, r = 2, b = 2, l = 2),
    legend.box.margin = margin(t = 0, r = 0, b = 0, l = 0),

    # facet
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text = element_text(size = 10),

    # axes
    axis.title = element_text(size = 11),
    axis.text  = element_text(size = 10),

    # remove extra padding
    plot.margin = margin(6, 8, 6, 8),

    # thin grid off (classic already removes)
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.4)
  )


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

# Helper: save both png and pdf
save_both <- function(plot, name, w, h, dpi = 300) {
  ggsave(file.path(out_dir, paste0(name, ".png")), plot = plot, width = w, height = h, dpi = dpi)
  ggsave(file.path(out_dir, paste0(name, ".pdf")), plot = plot, width = w, height = h, device = cairo_pdf)
}

# ============================================================
# Plot 1: average coverage vs n (per model)
# ============================================================
p_cov <- ggplot(df_summary, aes(x = n, y = avg_coverage, color = Method)) +
  geom_hline(yintercept = confidence_level, linetype = "dotdash", linewidth = 0.5, color = "black") +
  geom_line(linewidth = 0.9, na.rm = TRUE) +
  geom_point(size = 2.0, stroke = 0.2, na.rm = TRUE) +
  facet_wrap(~ model, scales = "fixed") +
  scale_y_continuous(limits = c(0.5, 1), breaks = seq(0.5, 1, by = 0.1)) +
  scale_method_color +
  scale_method_linetype +
  labs(
    x = "Sample size (n)",
    y = "Average pointwise coverage"
    # title/subtitle 제거 (caption에서 설명)
  ) +
  theme_paper

save_both(p_cov, "sim_marginal", w = 12, h = 6.5)

# ============================================================
# Plot 2: average width vs n (per model)
# ============================================================
p_wid <- ggplot(df_summary, aes(x = n, y = avg_width, color = Method)) +
  geom_line(linewidth = 0.9, na.rm = TRUE) +
  geom_point(size = 2.0, stroke = 0.2, na.rm = TRUE) +
  facet_wrap(~ model, scales = "free_y") +
  scale_method_color +
  scale_method_linetype +
  labs(
    x = "Sample size (n)",
    y = "Average interval width"
  ) +
  theme_paper

save_both(p_wid, "sim_width", w = 12, h = 6.5)

# ============================================================
# Plot 3: pointwise coverage curve (x-wise)
# facet: model x n, color/linetype: Method
# ============================================================
p_pt_cov <- ggplot(df, aes(x = x, y = coverage, color = Method)) +
  geom_hline(yintercept = confidence_level, linetype = "dotdash", linewidth = 0.45, color = "black") +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  facet_grid(model ~ n, labeller = label_both) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  scale_method_color +
  scale_method_linetype +
  labs(
    x = "x",
    y = "Pointwise coverage"
  ) +
  theme_paper +
  theme(
    strip.text = element_text(size = 9),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8)
  )

save_both(p_pt_cov, "sim_pointwise", w = 16, h = 10)

# ============================================================
# Plot 4: pointwise width curve (x-wise)
# facet: model x n, color/linetype: Method
# ============================================================
p_pt_wid <- ggplot(df, aes(x = x, y = mean_width, color = Method)) +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  facet_grid(model ~ n, labeller = label_both, scales = "free_y") +
  scale_method_color +
  scale_method_linetype +
  labs(
    x = "x",
    y = "Pointwise interval width"
  ) +
  theme_paper +
  theme(
    strip.text = element_text(size = 9),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8)
  )

save_both(p_pt_wid, "sim_pointwise_width", w = 16, h = 10)

cat("[make_plot] Done. Saved PNG+PDF into:", out_dir, "\n")