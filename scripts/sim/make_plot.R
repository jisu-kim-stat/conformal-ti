# scripts/make_plot.R

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

# ============================================================
# Paths
# ============================================================
pointwise_path <- "results/sim/models/pointwise_success_hcti_asym_cqr_ncqr_pti_alt_dgp_design_uniform_normal.csv"
marginal_path  <- "results/sim/models/marginal_pac_hcti_asym_cqr_ncqr_pti_alt_dgp_design_uniform_normal.csv"
px_good_path   <- "results/sim/models/px_good_proportion_hcti_asym_cqr_ncqr_pti_alt_dgp_design_uniform_normal.csv"
out_dir <- "results/sim/models/plots"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

content_level <- 0.90
confidence_level <- 0.95

cat("[make_plot] pointwise input:", pointwise_path, "\n")
cat("[make_plot] marginal input:", marginal_path, "\n")
cat("[make_plot] PX-good input:", px_good_path, "\n")
cat("[make_plot] output dir:", out_dir, "\n")

# ============================================================
# Plot style
# ============================================================

theme_pac_paper <- function(base_size = 12) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(
        face = "bold",
        size = base_size + 3,
        hjust = 0
      ),
      plot.subtitle = element_text(
        size = base_size,
        color = "grey25",
        hjust = 0,
        margin = margin(b = 8)
      ),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.35),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      panel.grid.major.y = element_line(color = "grey88", linewidth = 0.3),
      panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(
        fill = "grey93",
        color = "grey55",
        linewidth = 0.35
      ),
      strip.text = element_text(
        face = "bold",
        color = "black",
        size = base_size - 0.5
      ),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.key.width = unit(1.5, "lines"),
      legend.margin = margin(t = 2, b = 2),
      plot.margin = margin(10, 12, 10, 12)
    )
}

method_levels <- c("HCTI", "HCTI-asym", "CQR-TI", "NCQR-TI", "Parametric-TI")

method_cols <- c(
  "HCTI" = "#D55E00",
  "HCTI-asym" = "#CC79A7",
  "CQR-TI" = "#0072B2",
  "NCQR-TI" = "#009E73",
  "Parametric-TI" = "#555555"
)

method_shapes <- c(
  "HCTI" = 16,
  "HCTI-asym" = 18,
  "CQR-TI" = 17,
  "NCQR-TI" = 3,
  "Parametric-TI" = 15
)

# ============================================================
# Read data
# ============================================================

pointwise_df <- readr::read_csv(pointwise_path, show_col_types = FALSE)
marginal_df  <- readr::read_csv(marginal_path, show_col_types = FALSE)
px_good_df   <- readr::read_csv(px_good_path, show_col_types = FALSE)

# ============================================================
# Type cleanup and labels
# ============================================================

pointwise_df <- pointwise_df %>%
  mutate(
    model = as.integer(model),
    design = as.character(design),
    n_train = as.integer(n_train),
    n_cal = as.integer(n_cal),
    n_test = as.integer(n_test),
    epsilon = as.numeric(epsilon),
    x_bin = as.integer(x_bin),
    Method = factor(Method, levels = method_levels),
    model_lab = paste0("Model ", model),
    ncal_lab = paste0("n_cal = ", n_cal)
  )

marginal_df <- marginal_df %>%
  mutate(
    model = as.integer(model),
    design = as.character(design),
    n_train = as.integer(n_train),
    n_cal = as.integer(n_cal),
    n_test = as.integer(n_test),
    Method = factor(Method, levels = method_levels),
    model_lab = paste0("Model ", model)
  )

px_good_df <- px_good_df %>%
  mutate(
    model = as.integer(model),
    design = as.character(design),
    n_train = as.integer(n_train),
    n_cal = as.integer(n_cal),
    n_test = as.integer(n_test),
    epsilon = as.numeric(epsilon),
    Method = factor(Method, levels = method_levels),
    model_lab = paste0("Model ", model),
    eps_lab = paste0("\u03b5 = ", epsilon)
  )

design_vec <- sort(unique(marginal_df$design))

design_title <- function(des) {
  if (des == "uniform") {
    "uniform covariate design"
  } else if (des == "normal") {
    "normal covariate design"
  } else {
    paste0(des, " design")
  }
}

design_eval_text <- function(des) {
  if (des == "uniform") {
    "Evaluation uses a Uniform[-2,2] quantile grid."
  } else if (des == "normal") {
    "Evaluation uses a N(0,1) quantile grid."
  } else {
    "Evaluation uses the specified covariate design."
  }
}

# ============================================================
# Plot by design
# ============================================================

for (des in design_vec) {

  cat("[make_plot] plotting design:", des, "\n")

  design_out_dir <- file.path(out_dir, des)
  dir.create(design_out_dir, recursive = TRUE, showWarnings = FALSE)

  marginal_des <- marginal_df %>%
    filter(design == des)

  px_good_des <- px_good_df %>%
    filter(design == des)

  pointwise_des <- pointwise_df %>%
    filter(design == des)

  des_title <- design_title(des)
  des_eval <- design_eval_text(des)

  # ------------------------------------------------------------
  # Figure 1: Marginal PAC success
  # Main-text figure
  # ------------------------------------------------------------

  p_marginal_pac <- ggplot(
    marginal_des,
    aes(
      x = n_cal,
      y = marginal_pac_success,
      color = Method,
      shape = Method,
      group = Method
    )
  ) +
    geom_hline(
      yintercept = confidence_level,
      linetype = "dashed",
      color = "grey25",
      linewidth = 0.45
    ) +
    geom_line(linewidth = 0.85) +
    geom_point(size = 2.6, alpha = 0.95) +
    facet_wrap(~ model_lab, labeller = label_value) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(marginal_des$n_cal))) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(
      title = paste0("Marginal PAC success under ", des_title),
      subtitle = paste0("Dashed line: target probability ", confidence_level),
      x = "Calibration sample size",
      y = "Marginal PAC success"
    ) +
    theme_pac_paper(base_size = 12)

  ggsave(
    filename = file.path(design_out_dir, paste0("fig_marginal_pac_success_", des, ".png")),
    plot = p_marginal_pac,
    width = 12,
    height = 7,
    dpi = 300
  )

  # ------------------------------------------------------------
  # Figure 2: Mean marginal content
  # Appendix-style figure
  # ------------------------------------------------------------

  p_marginal_content <- ggplot(
    marginal_des,
    aes(
      x = n_cal,
      y = marginal_content_mean,
      color = Method,
      shape = Method,
      group = Method
    )
  ) +
    geom_hline(
      yintercept = content_level,
      linetype = "dashed",
      color = "grey25",
      linewidth = 0.45
    ) +
    geom_line(linewidth = 0.85) +
    geom_point(size = 2.6, alpha = 0.95) +
    facet_wrap(~ model_lab, labeller = label_value) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(marginal_des$n_cal))) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(
      title = paste0("Mean marginal content under ", des_title),
      subtitle = paste0("Dashed line: nominal content C = ", content_level),
      x = "Calibration sample size",
      y = "Mean marginal content"
    ) +
    theme_pac_paper(base_size = 12)

  ggsave(
    filename = file.path(design_out_dir, paste0("app_mean_marginal_content_", des, ".png")),
    plot = p_marginal_content,
    width = 12,
    height = 7,
    dpi = 300
  )

  # ------------------------------------------------------------
  # Figure 3: PX-good proportion by epsilon
  # Main-text figure
  # ------------------------------------------------------------

  p_px_eps <- ggplot(
    px_good_des,
    aes(
      x = n_cal,
      y = px_good_proportion,
      color = Method,
      shape = Method,
      group = Method
    )
  ) +
    geom_hline(
      yintercept = 1,
      linetype = "dashed",
      color = "grey25",
      linewidth = 0.45
    ) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2.25, alpha = 0.95) +
    facet_grid(eps_lab ~ model_lab) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(px_good_des$n_cal))) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(
      title = paste0("PX-good proportion under ", des_title),
      subtitle = "A covariate point is good if its conditional success probability exceeds 1 - alpha.",
      x = "Calibration sample size",
      y = "PX-good proportion"
    ) +
    theme_pac_paper(base_size = 9.8) +
    theme(
      strip.text.y = element_text(angle = 0),
      legend.position = "bottom"
    )

  ggsave(
    filename = file.path(design_out_dir, paste0("fig_px_good_by_epsilon_", des, ".png")),
    plot = p_px_eps,
    width = 14,
    height = 10,
    dpi = 300
  )

  # ------------------------------------------------------------
  # Figure 3b: PX-good proportion, separate epsilon plots
  # Useful for slides / appendix
  # ------------------------------------------------------------

  for (eps in sort(unique(px_good_des$epsilon))) {

    df_eps <- px_good_des %>%
      filter(abs(epsilon - eps) < 1e-12)

    eps_name <- gsub("\\.", "p", as.character(eps))

    p_px_one <- ggplot(
      df_eps,
      aes(
        x = n_cal,
        y = px_good_proportion,
        color = Method,
        shape = Method,
        group = Method
      )
    ) +
      geom_hline(
        yintercept = 1,
        linetype = "dashed",
        color = "grey25",
        linewidth = 0.45
      ) +
      geom_line(linewidth = 0.85) +
      geom_point(size = 2.5, alpha = 0.95) +
      facet_wrap(~ model_lab, nrow = 2, labeller = label_value) +
      scale_color_manual(values = method_cols, drop = FALSE) +
      scale_shape_manual(values = method_shapes, drop = FALSE) +
      scale_x_continuous(breaks = sort(unique(df_eps$n_cal))) +
      coord_cartesian(ylim = c(0, 1.03)) +
      labs(
        title = paste0("PX-good proportion under ", des_title, ", \u03b5 = ", eps),
        subtitle = paste0(
          "Good if conditional content exceeds C - \u03b5 = ",
          content_level - eps,
          " with probability at least ",
          confidence_level,
          "."
        ),
        x = "Calibration sample size",
        y = "PX-good proportion"
      ) +
      theme_pac_paper(base_size = 12)

    ggsave(
      filename = file.path(
        design_out_dir,
        paste0("fig_px_good_epsilon_", eps_name, "_", des, ".png")
      ),
      plot = p_px_one,
      width = 13,
      height = 7.5,
      dpi = 300
    )
  }

  # ------------------------------------------------------------
  # Figure 4: Average interval width
  # Main-text figure
  # ------------------------------------------------------------

  p_width <- ggplot(
    marginal_des,
    aes(
      x = n_cal,
      y = average_width_mean,
      color = Method,
      shape = Method,
      group = Method
    )
  ) +
    geom_line(linewidth = 0.85) +
    geom_point(size = 2.6, alpha = 0.95) +
    facet_wrap(~ model_lab, scales = "free_y", labeller = label_value) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(marginal_des$n_cal))) +
    labs(
      title = paste0("Average interval width under ", des_title),
      subtitle = "Average over evaluation points and Monte Carlo replications.",
      x = "Calibration sample size",
      y = "Average interval width"
    ) +
    theme_pac_paper(base_size = 12)

  ggsave(
    filename = file.path(design_out_dir, paste0("fig_average_width_", des, ".png")),
    plot = p_width,
    width = 12,
    height = 7,
    dpi = 300
  )

  # ------------------------------------------------------------
  # Figure 5: Pointwise / conditional PAC success curve
  # Appendix-style diagnostic
  # ------------------------------------------------------------

  pointwise_plot_df <- pointwise_des %>%
    filter(
      model != 6,
      abs(epsilon - 0) < 1e-12
    )

  p_pointwise_success <- ggplot(
    pointwise_plot_df,
    aes(
      x = x,
      y = pointwise_success,
      color = Method,
      group = Method
    )
  ) +
    geom_hline(
      yintercept = confidence_level,
      linetype = "dashed",
      color = "grey25",
      linewidth = 0.45
    ) +
    geom_line(linewidth = 0.7) +
    facet_grid(
      rows = vars(model_lab),
      cols = vars(ncal_lab),
      labeller = label_value,
      scales = "free_x"
    ) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(
      title = paste0("Pointwise PAC success under ", des_title),
      subtitle = paste0(des_eval, " Dashed line: target probability ", confidence_level, "."),
      x = "Covariate x",
      y = "Success probability"
    ) +
    theme_pac_paper(base_size = 10.5)

  ggsave(
    filename = file.path(design_out_dir, paste0("app_pointwise_pac_success_", des, ".png")),
    plot = p_pointwise_success,
    width = 14,
    height = 9,
    dpi = 300
  )

  # ------------------------------------------------------------
  # Figure 6: Mean conditional content curve
  # Appendix-style diagnostic
  # ------------------------------------------------------------

  p_mean_content <- ggplot(
    pointwise_plot_df,
    aes(
      x = x,
      y = mean_content,
      color = Method,
      group = Method
    )
  ) +
    geom_hline(
      yintercept = content_level,
      linetype = "dashed",
      color = "grey25",
      linewidth = 0.45
    ) +
    geom_line(linewidth = 0.7) +
    facet_grid(
      rows = vars(model_lab),
      cols = vars(ncal_lab),
      labeller = label_value,
      scales = "free_x"
    ) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(
      title = paste0("Mean conditional content under ", des_title),
      subtitle = paste0(des_eval, " Dashed line: nominal content C = ", content_level, "."),
      x = "Covariate x",
      y = "Mean conditional content"
    ) +
    theme_pac_paper(base_size = 10.5)

  ggsave(
    filename = file.path(design_out_dir, paste0("app_mean_conditional_content_", des, ".png")),
    plot = p_mean_content,
    width = 14,
    height = 9,
    dpi = 300
  )

  # ------------------------------------------------------------
  # Figure 7: Conditional mean interval width curve
  # Appendix-style diagnostic
  # ------------------------------------------------------------

  p_pointwise_width <- ggplot(
    pointwise_plot_df,
    aes(
      x = x,
      y = mean_width,
      color = Method,
      group = Method
    )
  ) +
    geom_line(linewidth = 0.7) +
    facet_grid(
      rows = vars(model_lab),
      cols = vars(ncal_lab),
      labeller = label_value,
      scales = "free"
    ) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    labs(
      title = paste0("Conditional mean interval width under ", des_title),
      subtitle = des_eval,
      x = "Covariate x",
      y = "Mean interval width"
    ) +
    theme_pac_paper(base_size = 10.5)

  ggsave(
    filename = file.path(design_out_dir, paste0("app_conditional_width_", des, ".png")),
    plot = p_pointwise_width,
    width = 14,
    height = 9,
    dpi = 300
  )

  # ------------------------------------------------------------
  # Save summary tables
  # ------------------------------------------------------------

  readr::write_csv(
    marginal_des,
    file.path(design_out_dir, "summary_marginal_pac.csv")
  )

  readr::write_csv(
    px_good_des,
    file.path(design_out_dir, "summary_px_good_proportion.csv")
  )

  readr::write_csv(
    marginal_des %>% filter(model == 6),
    file.path(design_out_dir, "summary_model6_marginal.csv")
  )

  readr::write_csv(
    px_good_des %>% filter(model == 6),
    file.path(design_out_dir, "summary_model6_px_good.csv")
  )

  cat("[make_plot] saved plots to:", design_out_dir, "\n")
}

cat("[make_plot] all plots saved under:", out_dir, "\n")