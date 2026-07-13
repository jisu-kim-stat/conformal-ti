# scripts/make_plot.R

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

# ---------------------------
# Paths
# ---------------------------
pointwise_path <- "results/sim/models/pointwise_success_hcti_cqr_pti_ncal_grid.csv"
marginal_path  <- "results/sim/models/marginal_pac_hcti_cqr_pti_ncal_grid.csv"
px_good_path   <- "results/sim/models/px_good_proportion_hcti_cqr_pti_ncal_grid.csv"


out_dir <- "results/sim/models/plots"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

content_level <- 0.90
confidence_level <- 0.95
pd <- position_dodge(width = 25)

cat("[make_plot] pointwise input:", pointwise_path, "\n")
cat("[make_plot] marginal input:", marginal_path, "\n")
cat("[make_plot] PX-good input:", px_good_path, "\n")
cat("[make_plot] output dir:", out_dir, "\n")

# ---------------------------
# Read
# ---------------------------
pointwise_df <- readr::read_csv(pointwise_path, show_col_types = FALSE)
marginal_df  <- readr::read_csv(marginal_path, show_col_types = FALSE)
px_good_df   <- readr::read_csv(px_good_path, show_col_types = FALSE)

# ---------------------------
# Type cleanup
# ---------------------------
method_levels <- c("HCTI", "CQR-TI", "Parametric-TI")

pointwise_df <- pointwise_df %>%
  mutate(
    model = as.integer(model),
    n_train = as.integer(n_train),
    n_cal = as.integer(n_cal),
    n_test = as.integer(n_test),
    Method = factor(Method, levels = method_levels)
  )

marginal_df <- marginal_df %>%
  mutate(
    model = as.integer(model),
    n_train = as.integer(n_train),
    n_cal = as.integer(n_cal),
    n_test = as.integer(n_test),
    Method = factor(Method, levels = method_levels)
  )

px_good_df <- px_good_df %>%
  mutate(
    model = as.integer(model),
    n_train = as.integer(n_train),
    n_cal = as.integer(n_cal),
    n_test = as.integer(n_test),
    Method = factor(Method, levels = method_levels)
  )

# ============================================================
# Plot 1: Marginal PAC success
# This estimates:
#   P_D{ P_{X,Y}(Y in T(X;D)) >= C }
# Target: >= 1 - alpha = 0.95
# ============================================================

p_marginal_pac <- ggplot(
  marginal_df,
  aes(
    x = n_cal,
    y = marginal_pac_success,
    color = Method,
    group = Method
  )
) +
  geom_point(
  aes(shape = Method),
  size = 3.0,
  alpha = 0.85,
  position = pd
) +
  geom_hline(
    yintercept = confidence_level,
    linetype = "dashed",
    color = "black"
  ) +
  facet_wrap(~ model, labeller = label_both) +
  scale_x_continuous(breaks = sort(unique(marginal_df$n_cal))) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Marginal PAC success",
    subtitle = paste0(
      "Target: P_D{marginal content >= ",
      content_level,
      "} >= ",
      confidence_level
    ),
    x = "Calibration sample size",
    y = "Marginal PAC success",
    color = "Method",
    shape = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

if (length(unique(marginal_df$n_cal)) > 1) {
  p_marginal_pac <- p_marginal_pac +
    geom_line(linewidth = 1.0)
}

ggsave(
  filename = file.path(out_dir, "marginal_pac_success_vs_ncal.png"),
  plot = p_marginal_pac,
  width = 12,
  height = 7,
  dpi = 300
)

# ============================================================
# Plot 2: Marginal content mean
# This reports the average marginal content itself.
# Target line: C = 0.90
# ============================================================

p_marginal_content <- ggplot(
  marginal_df,
  aes(
    x = n_cal,
    y = marginal_content_mean,
    color = Method,
    group = Method
  )
) +
  geom_point(
  aes(shape = Method),
  size = 3.0,
  alpha = 0.85,
  position = pd
) +
  geom_hline(
    yintercept = content_level,
    linetype = "dashed",
    color = "black"
  ) +
  facet_wrap(~ model, labeller = label_both) +
  scale_x_continuous(breaks = sort(unique(marginal_df$n_cal))) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Mean marginal content",
    subtitle = paste0("Dashed line: content level C = ", content_level),
    x = "Calibration sample size",
    y = "Mean marginal content",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

if (length(unique(marginal_df$n_cal)) > 1) {
  p_marginal_content <- p_marginal_content +
    geom_line(linewidth = 1.0)
}

ggsave(
  filename = file.path(out_dir, "mean_marginal_content_vs_ncal.png"),
  plot = p_marginal_content,
  width = 12,
  height = 7,
  dpi = 300
)

# ============================================================
# Plot 3: PX-good proportion
# This estimates:
#   P_X{x : P_D[content(x;D) >= C] >= 1-alpha}
# For Notion 3, this should approach 1 asymptotically.
# ============================================================

p_px_good <- ggplot(
  px_good_df,
  aes(
    x = n_cal,
    y = px_good_proportion,
    color = Method,
    group = Method
  )
) +
  geom_point(
  aes(shape = Method),
  size = 3.0,
  alpha = 0.85,
  position = pd
) +
  geom_hline(
    yintercept = 1.0,
    linetype = "dashed",
    color = "black"
  ) +
  facet_wrap(~ model, labeller = label_both) +
  scale_x_continuous(breaks = sort(unique(px_good_df$n_cal))) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "PX-good proportion",
    subtitle = paste0(
      "Good x if P_D{content(x;D) >= ",
      content_level,
      "} >= ",
      confidence_level
    ),
    x = "Calibration sample size",
    y = "Proportion of good x values",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

if (length(unique(px_good_df$n_cal)) > 1) {
  p_px_good <- p_px_good +
    geom_line(linewidth = 1.0)
}

ggsave(
  filename = file.path(out_dir, "px_good_proportion_vs_ncal.png"),
  plot = p_px_good,
  width = 12,
  height = 7,
  dpi = 300
)

# ============================================================
# Plot 4: Average interval width
# All models included, including model 6.
# ============================================================

p_width <- ggplot(
  marginal_df,
  aes(
    x = n_cal,
    y = average_width_mean,
    color = Method,
    group = Method
  )
) +
  geom_point(
  aes(shape = Method),
  size = 3.0,
  alpha = 0.85,
  position = pd
) +
  facet_wrap(~ model, scales = "free_y", labeller = label_both) +
  scale_x_continuous(breaks = sort(unique(marginal_df$n_cal))) +
  labs(
    title = "Average interval width",
    subtitle = "Mean over test points and Monte Carlo replications",
    x = "Calibration sample size",
    y = "Average width",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

if (length(unique(marginal_df$n_cal)) > 1) {
  p_width <- p_width +
    geom_line(linewidth = 1.0)
}

ggsave(
  filename = file.path(out_dir, "average_width_vs_ncal.png"),
  plot = p_width,
  width = 12,
  height = 7,
  dpi = 300
)

# ============================================================
# Plot 5: Pointwise PAC success curve, models 1--5 only
# y-axis:
#   P_D{content(x;D) >= C}
# target line: 1-alpha = 0.95
# ============================================================

pointwise_plot_df <- pointwise_df %>%
  filter(model != 6)

p_pointwise_success <- ggplot(
  pointwise_plot_df,
  aes(
    x = x,
    y = pointwise_success,
    color = Method,
    group = Method
  )
) +
  geom_line(linewidth = 0.85) +
  geom_hline(
    yintercept = confidence_level,
    linetype = "dashed",
    color = "black",
    linewidth = 0.6
  ) +
  facet_grid(
    rows = vars(model),
    cols = vars(n_cal),
    labeller = label_both,
    scales = "free_x"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Pointwise PAC success curve",
    subtitle = paste0(
      "Models 1--5 only; y-axis = P_D{content(x;D) >= ",
      content_level,
      "}; dashed line = ",
      confidence_level
    ),
    x = "x",
    y = "Pointwise PAC success",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(out_dir, "pointwise_pac_success_curve_models1to5.png"),
  plot = p_pointwise_success,
  width = 14,
  height = 9,
  dpi = 300
)

# ============================================================
# Plot 6: Mean conditional content curve, models 1--5 only
# This is auxiliary: average of content(x;D) over D.
# Target line: C = 0.90
# ============================================================

p_mean_content <- ggplot(
  pointwise_plot_df,
  aes(
    x = x,
    y = mean_content,
    color = Method,
    group = Method
  )
) +
  geom_line(linewidth = 0.85) +
  geom_hline(
    yintercept = content_level,
    linetype = "dashed",
    color = "black",
    linewidth = 0.6
  ) +
  facet_grid(
    rows = vars(model),
    cols = vars(n_cal),
    labeller = label_both,
    scales = "free_x"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Mean conditional content curve",
    subtitle = paste0(
      "Models 1--5 only; dashed line: C = ",
      content_level
    ),
    x = "x",
    y = "Mean conditional content",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(out_dir, "mean_conditional_content_curve_models1to5.png"),
  plot = p_mean_content,
  width = 14,
  height = 9,
  dpi = 300
)

# ============================================================
# Plot 7: Pointwise width curve, models 1--5 only
# ============================================================

p_pointwise_width <- ggplot(
  pointwise_plot_df,
  aes(
    x = x,
    y = mean_width,
    color = Method,
    group = Method
  )
) +
  geom_line(linewidth = 0.85) +
  facet_grid(
    rows = vars(model),
    cols = vars(n_cal),
    labeller = label_both,
    scales = "free"
  ) +
  labs(
    title = "Pointwise mean interval width",
    subtitle = "Models 1--5 only",
    x = "x",
    y = "Mean width",
    color = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(out_dir, "pointwise_width_curve_models1to5.png"),
  plot = p_pointwise_width,
  width = 14,
  height = 9,
  dpi = 300
)

# ============================================================
# Save summary tables into plots folder as well
# ============================================================

readr::write_csv(
  marginal_df,
  file.path(out_dir, "summary_marginal_pac.csv")
)

readr::write_csv(
  px_good_df,
  file.path(out_dir, "summary_px_good_proportion.csv")
)

model6_summary <- list(
  marginal = marginal_df %>% filter(model == 6),
  px_good  = px_good_df %>% filter(model == 6)
)

readr::write_csv(
  model6_summary$marginal,
  file.path(out_dir, "summary_model6_marginal.csv")
)

readr::write_csv(
  model6_summary$px_good,
  file.path(out_dir, "summary_model6_px_good.csv")
)

cat("[make_plot] saved plots to:", out_dir, "\n")
cat("[make_plot] saved marginal summary:", file.path(out_dir, "summary_marginal_pac.csv"), "\n")
cat("[make_plot] saved PX-good summary:", file.path(out_dir, "summary_px_good_proportion.csv"), "\n")
cat("[make_plot] saved model 6 marginal summary:", file.path(out_dir, "summary_model6_marginal.csv"), "\n")
cat("[make_plot] saved model 6 PX-good summary:", file.path(out_dir, "summary_model6_px_good.csv"), "\n")