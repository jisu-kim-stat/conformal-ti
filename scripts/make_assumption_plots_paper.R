# scripts/make_assumption_plots_paper.R
# ------------------------------------------------------------
# Paper-style assumption diagnostic plots
# Uses existing CSV outputs from check_assumptions_alt_dgp.R
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tidyr)
  library(stringr)
})

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

diag_dir <- "results/sim/models/assumption_diagnostics_alt_dgp"
out_dir  <- file.path(diag_dir, "paper_plots")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

estimation_path <- file.path(diag_dir, "assumption_estimation_summary.csv")
ks_path         <- file.path(diag_dir, "score_pivotality_ks_summary.csv")
good_path       <- file.path(diag_dir, "score_pivotality_good_summary.csv")

cat("[input] estimation:", estimation_path, "\n")
cat("[input] KS:", ks_path, "\n")
cat("[input] pivotality-good:", good_path, "\n")
cat("[output dir]", out_dir, "\n")

# ------------------------------------------------------------
# Read data
# ------------------------------------------------------------

estimation_summary <- readr::read_csv(estimation_path, show_col_types = FALSE)
pivotality_summary <- readr::read_csv(ks_path, show_col_types = FALSE)
pivotality_good_summary <- readr::read_csv(good_path, show_col_types = FALSE)

# ------------------------------------------------------------
# Labels and style
# ------------------------------------------------------------

method_levels <- c("SR-TI", "ASR-TI", "CQR-TI")

method_cols <- c(
  "SR-TI" = "#D55E00",
  "ASR-TI" = "#CC79A7",
  "CQR-TI" = "#0072B2"
)

method_shapes <- c(
  "SR-TI" = 16,
  "ASR-TI" = 18,
  "CQR-TI" = 17
)

design_labels <- c(
  "uniform" = "Uniform",
  "normal" = "Normal"
)

theme_paper_diag <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2, hjust = 0),
      plot.subtitle = element_blank(),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.35),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      panel.grid.major.y = element_line(color = "grey88", linewidth = 0.3),
      panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(
        fill = "grey94",
        color = "grey65",
        linewidth = 0.35
      ),
      strip.text = element_text(face = "bold", color = "black", size = base_size - 0.5),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.key.width = unit(1.4, "lines"),
      plot.margin = margin(8, 10, 8, 10)
    )
}

clean_common <- function(df) {
  df %>%
    mutate(
      model = as.integer(model),
      n_train = as.integer(n_train)
    ) %>%
    filter(model <= 5) %>%
    mutate(
      design = as.character(design),
      model_lab = factor(
        paste0("Model ", model),
        levels = paste0("Model ", sort(unique(model)))
      ),
      design_lab = factor(
        recode(design, !!!design_labels),
        levels = unname(design_labels)
      )
    )
}

estimation_summary <- clean_common(estimation_summary)

pivotality_summary <- pivotality_summary %>%
  clean_common() %>%
  mutate(
    Method = factor(Method, levels = method_levels)
  )

pivotality_good_summary <- pivotality_good_summary %>%
  clean_common() %>%
  mutate(
    Method = factor(Method, levels = method_levels),
    ks_eps_lab = paste0("KS \u2264 ", ks_eps)
  )

n_train_breaks <- sort(unique(estimation_summary$n_train))

# ------------------------------------------------------------
# Main Figure: Score pivotality KS q90
# ------------------------------------------------------------

p_ks_main <- ggplot(
  pivotality_summary,
  aes(
    x = n_train,
    y = ks_q90,
    color = Method,
    shape = Method,
    group = Method
  )
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.5, alpha = 0.95) +
  facet_grid(design_lab ~ model_lab, scales = "free_y") +
  scale_color_manual(values = method_cols, drop = FALSE) +
  scale_shape_manual(values = method_shapes, drop = FALSE) +
  scale_x_continuous(breaks = n_train_breaks) +
  labs(
    title = "Score pivotality diagnostic",
    x = "Training sample size",
    y = expression("90th percentile of " * Delta(x))
  ) +
  theme_paper_diag(base_size = 11)

ggsave(
  filename = file.path(out_dir, "fig_score_pivotality_KS_q90_paper.png"),
  plot = p_ks_main,
  width = 12.5,
  height = 6.4,
  dpi = 300
)

# Optional: one-row version by design, easier for paper if too wide
for (des in sort(unique(pivotality_summary$design))) {

  df_des <- pivotality_summary %>%
    filter(design == des)

  des_lab <- unique(df_des$design_lab)

  p_ks_des <- ggplot(
    df_des,
    aes(
      x = n_train,
      y = ks_q90,
      color = Method,
      shape = Method,
      group = Method
    )
  ) +
    geom_line(linewidth = 0.85) +
    geom_point(size = 2.5, alpha = 0.95) +
    facet_wrap(~ model_lab, nrow = 1, scales = "free_y") +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = n_train_breaks) +
    labs(
      title = paste0("Score pivotality diagnostic: ", des_lab, " design"),
      x = "Training sample size",
      y = expression("90th percentile of " * Delta(x))
    ) +
    theme_paper_diag(base_size = 11)

  ggsave(
    filename = file.path(out_dir, paste0("fig_score_pivotality_KS_q90_", des, "_paper.png")),
    plot = p_ks_des,
    width = 13,
    height = 4.2,
    dpi = 300
  )
}

# ------------------------------------------------------------
# Appendix Figure A1: Mean estimation error
# ------------------------------------------------------------

p_mean <- ggplot(
  estimation_summary,
  aes(x = n_train, y = mean_l2_error_mean, group = 1)
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.5) +
  facet_grid(design_lab ~ model_lab, scales = "free_y") +
  scale_x_continuous(breaks = n_train_breaks) +
  labs(
    title = "Mean estimation error",
    x = "Training sample size",
    y = expression(E_X[(hat(mu)(X) - mu(X))^2])
  ) +
  theme_paper_diag(base_size = 11)

ggsave(
  filename = file.path(out_dir, "app_mean_estimation_error_paper.png"),
  plot = p_mean,
  width = 12.5,
  height = 6.4,
  dpi = 300
)

# ------------------------------------------------------------
# Appendix Figure A2: Variance estimation error
# ------------------------------------------------------------

p_var <- ggplot(
  estimation_summary,
  aes(x = n_train, y = var_l1_error_mean, group = 1)
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.5) +
  facet_grid(design_lab ~ model_lab, scales = "free_y") +
  scale_x_continuous(breaks = n_train_breaks) +
  labs(
    title = "Variance estimation error",
    x = "Training sample size",
    y = expression(E_X[abs(hat(sigma)^2(X) - sigma^2(X))])
  ) +
  theme_paper_diag(base_size = 11)

ggsave(
  filename = file.path(out_dir, "app_variance_estimation_error_paper.png"),
  plot = p_var,
  width = 12.5,
  height = 6.4,
  dpi = 300
)

# ------------------------------------------------------------
# Appendix Figure A3: Asymmetric tail-scale error
# ------------------------------------------------------------

tail_long <- estimation_summary %>%
  select(
    model, model_lab, design, design_lab, n_train,
    a_minus_abs_error_mean,
    a_plus_abs_error_mean
  ) %>%
  pivot_longer(
    cols = c(a_minus_abs_error_mean, a_plus_abs_error_mean),
    names_to = "tail_scale",
    values_to = "abs_error"
  ) %>%
  mutate(
    tail_scale = recode(
      tail_scale,
      "a_minus_abs_error_mean" = "a_minus",
      "a_plus_abs_error_mean" = "a_plus"
    ),
    tail_scale = factor(tail_scale, levels = c("a_minus", "a_plus"))
  )

tail_cols <- c(
  "a_minus" = "#E69F00",
  "a_plus" = "#009E73"
)

tail_labs <- c(
  "a_minus" = "a-",
  "a_plus" = "a+"
)

p_tail <- ggplot(
  tail_long,
  aes(
    x = n_train,
    y = abs_error,
    color = tail_scale,
    shape = tail_scale,
    group = tail_scale
  )
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.5, alpha = 0.95) +
  facet_grid(design_lab ~ model_lab, scales = "free_y") +
  scale_color_manual(values = tail_cols, labels = tail_labs) +
  scale_shape_manual(values = c("a_minus" = 16, "a_plus" = 17), labels = tail_labs) +
  scale_x_continuous(breaks = n_train_breaks) +
  labs(
    title = "Asymmetric tail-scale estimation error",
    x = "Training sample size",
    y = "Absolute error"
  ) +
  theme_paper_diag(base_size = 11)

ggsave(
  filename = file.path(out_dir, "app_tail_scale_error_paper.png"),
  plot = p_tail,
  width = 12.5,
  height = 6.4,
  dpi = 300
)
# ------------------------------------------------------------
# Appendix Figure A4: Pivotality-good proportion
# ------------------------------------------------------------

p_good <- ggplot(
  pivotality_good_summary,
  aes(
    x = n_train,
    y = px_good_pivotality,
    color = Method,
    shape = Method,
    group = Method
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.3, alpha = 0.95) +
  facet_grid(ks_eps_lab ~ model_lab + design_lab) +
  scale_color_manual(values = method_cols, drop = FALSE) +
  scale_shape_manual(values = method_shapes, drop = FALSE) +
  scale_x_continuous(breaks = n_train_breaks) +
  coord_cartesian(ylim = c(0, 1.03)) +
  labs(
    title = "Pivotality-good proportion",
    x = "Training sample size",
    y = "Proportion"
  ) +
  theme_paper_diag(base_size = 8.8) +
  theme(
    strip.text.x = element_text(size = 7.4),
    strip.text.y = element_text(size = 8.2)
  )

ggsave(
  filename = file.path(out_dir, "app_pivotality_good_proportion_paper.png"),
  plot = p_good,
  width = 15,
  height = 8.5,
  dpi = 300
)

# Optional: pivotality-good proportion by design
for (des in sort(unique(pivotality_good_summary$design))) {

  df_des <- pivotality_good_summary %>%
    filter(design == des)

  des_lab <- unique(df_des$design_lab)

  p_good_des <- ggplot(
    df_des,
    aes(
      x = n_train,
      y = px_good_pivotality,
      color = Method,
      shape = Method,
      group = Method
    )
  ) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2.3, alpha = 0.95) +
    facet_grid(ks_eps_lab ~ model_lab) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = n_train_breaks) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(
      title = paste0("Pivotality-good proportion: ", des_lab, " design"),
      x = "Training sample size",
      y = "Proportion"
    ) +
    theme_paper_diag(base_size = 9.5) +
    theme(
      strip.text.x = element_text(size = 8.2),
      strip.text.y = element_text(size = 8.8)
    )

  ggsave(
    filename = file.path(out_dir, paste0("app_pivotality_good_proportion_", des, "_paper.png")),
    plot = p_good_des,
    width = 13,
    height = 7.2,
    dpi = 300
  )
}

cat("[done] paper-style diagnostic plots saved under:", out_dir, "\n")