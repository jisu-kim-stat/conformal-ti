# scripts/make_plot.R

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

# ============================================================
# Paths
# ============================================================
pointwise_path <- "/Users/jisukim/ti_project/results/sim/models/pointwise_success_hcti_asym_cqr_pti_alt_dgp_design_uniform_normal.csv"
marginal_path  <- "/Users/jisukim/ti_project/results/sim/models/marginal_pac_hcti_asym_cqr_pti_alt_dgp_design_uniform_normal.csv"
px_good_path   <- "/Users/jisukim/ti_project/results/sim/models/px_good_proportion_hcti_asym_cqr_pti_alt_dgp_design_uniform_normal.csv"

out_dir <- "results/sim/models/plots_alt_dgp"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

models_keep <- 1:5
model_labels <- c(
  "1" = "Gaussian",
  "2" = "Heavy-tailed",
  "3" = "Heteroscedastic",
  "4" = "Globally skewed",
  "5" = "X-dependent\nskewness"
)

content_level <- 0.90
confidence_level <- 0.95

cat("[make_plot] pointwise input:", pointwise_path, "\n")
cat("[make_plot] marginal input:", marginal_path, "\n")
cat("[make_plot] PX-good input:", px_good_path, "\n")
cat("[make_plot] output dir:", out_dir, "\n")

# ============================================================
# Plot style
# ============================================================

theme_pac_paper <- function(base_size = 10) {
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

method_levels <- c("SR-TI", "ASR-TI", "CQR-TI", "Parametric-TI")

method_cols <- c(
  "SR-TI" = "#D55E00",
  "ASR-TI" = "#CC79A7",
  "CQR-TI" = "#0072B2",
  "Parametric-TI" = "#555555"
)

method_shapes <- c(
  "SR-TI" = 16,
  "ASR-TI" = 18,
  "CQR-TI" = 17,
  "Parametric-TI" = 15
)

method_linetypes <- c(
  "SR-TI" = "solid",
  "ASR-TI" = "solid",
  "CQR-TI" = "solid",
  "Parametric-TI" = "solid"
)

# ============================================================
# Read data
# ============================================================

pointwise_df <- readr::read_csv(pointwise_path, show_col_types = FALSE) %>%
  dplyr::filter(
    as.integer(model) %in% models_keep
  )

marginal_df <- readr::read_csv(marginal_path, show_col_types = FALSE) %>%
  dplyr::filter(
    as.integer(model) %in% models_keep
  )

px_good_df <- readr::read_csv(px_good_path, show_col_types = FALSE) %>%
  dplyr::filter(
    as.integer(model) %in% models_keep
  )

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
    model_lab = factor(
      paste0("Model ", model),
      levels = paste0("Model ", models_keep),
      labels = unname(model_labels[as.character(models_keep)])
    ),
    ncal_lab = factor(
      paste0("n_cal = ", n_cal),
      levels = paste0("n_cal = ", sort(unique(n_cal)))
    )
  )

marginal_df <- marginal_df %>%
  mutate(
    model = as.integer(model),
    design = as.character(design),
    n_train = as.integer(n_train),
    n_cal = as.integer(n_cal),
    n_test = as.integer(n_test),
    Method = factor(Method, levels = method_levels),
    model_lab = factor(
      paste0("Model ", model),
      levels = paste0("Model ", models_keep),
      labels = unname(model_labels[as.character(models_keep)])
    )
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
    model_lab = factor(
      paste0("Model ", model),
      levels = paste0("Model ", models_keep),
      labels = unname(model_labels[as.character(models_keep)])
    ),
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
    coord_cartesian(ylim = c(0.80, 1.01)) +
    scale_y_continuous(
      breaks = c(0.80, 0.85, 0.90, 0.95, 1.00)
    ) + 
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
  # Figure 7: Conditional interval width diagnostics
  # ------------------------------------------------------------

  if (des == "normal") {

    # Central 98% region under N(0,1)
    central_prob <- 0.98
    tail_prob <- (1 - central_prob) / 2
    central_x_lim <- qnorm(c(tail_prob, 1 - tail_prob))

    central_width_df <- pointwise_plot_df %>%
      filter(
        x >= central_x_lim[1],
        x <= central_x_lim[2]
      )

    # ==========================================================
    # 7a. All methods over the central 98% covariate region
    # Main conditional-width comparison
    # ==========================================================

    p_width_central <- ggplot(
      central_width_df,
      aes(
        x = x,
        y = mean_width,
        color = Method,
        group = Method
      )
    ) +
      geom_line(linewidth = 0.75) +
      facet_grid(
        rows = vars(model_lab),
        cols = vars(ncal_lab),
        labeller = label_value,
        scales = "free_y",
        drop = FALSE
      ) +
      scale_color_manual(
        values = method_cols,
        drop = FALSE
      ) +
      scale_x_continuous(
        breaks = c(-2, -1, 0, 1, 2)
      ) +
      labs(
        title = paste0(
          "Conditional mean interval width in the central ",
          central_prob * 100,
          "% covariate region"
        ),
        subtitle = paste0(
          "Normal covariate design; ",
          round(central_x_lim[1], 2),
          " \u2264 x \u2264 ",
          round(central_x_lim[2], 2),
          "."
        ),
        x = "Covariate x",
        y = "Mean interval width"
      ) +
      theme_pac_paper(base_size = 10.5)

    ggsave(
      filename = file.path(
        design_out_dir,
        "app_conditional_width_central98_normal.png"
      ),
      plot = p_width_central,
      width = 14,
      height = 10.5,
      dpi = 300
    )

    # ==========================================================
    # 7b. Score-based intervals over the full covariate range
    # Parametric extrapolation does not compress the other curves
    # ==========================================================

    score_width_df <- pointwise_plot_df %>%
      filter(
        Method %in% c("SR-TI", "ASR-TI", "CQR-TI")
      ) %>%
      droplevels()

    p_width_score_full <- ggplot(
      score_width_df,
      aes(
        x = x,
        y = mean_width,
        color = Method,
        group = Method
      )
    ) +
      geom_line(linewidth = 0.75) +
      facet_grid(
        rows = vars(model_lab),
        cols = vars(ncal_lab),
        labeller = label_value,
        scales = "free_y",
        drop = FALSE
      ) +
      scale_color_manual(
        values = method_cols[
          c("SR-TI", "ASR-TI", "CQR-TI")
        ],
        drop = FALSE
      ) +
      labs(
        title = paste0(
          "Conditional mean width of score-based intervals under ",
          des_title
        ),
        subtitle = paste0(
          des_eval,
          " Parametric-TI is displayed separately because of tail extrapolation."
        ),
        x = "Covariate x",
        y = "Mean interval width"
      ) +
      theme_pac_paper(base_size = 10.5)

    ggsave(
      filename = file.path(
        design_out_dir,
        "app_conditional_width_score_based_full_normal.png"
      ),
      plot = p_width_score_full,
      width = 14,
      height = 10.5,
      dpi = 300
    )

    # ==========================================================
    # 7c. Parametric-TI over the full covariate range
    # Pseudo-log scale preserves small widths and tail explosion
    # ==========================================================

    parametric_width_df <- pointwise_plot_df %>%
      filter(Method == "Parametric-TI") %>%
      droplevels()

    p_width_parametric_full <- ggplot(
      parametric_width_df,
      aes(
        x = x,
        y = mean_width,
        group = Method
      )
    ) +
      geom_line(
        color = method_cols["Parametric-TI"],
        linewidth = 0.8
      ) +
      facet_grid(
        rows = vars(model_lab),
        cols = vars(ncal_lab),
        labeller = label_value,
        scales = "free_y",
        drop = FALSE
      ) +
      scale_y_continuous(
        trans = scales::pseudo_log_trans(sigma = 1)
      ) +
      labs(
        title = paste0(
          "Conditional mean width of Parametric-TI under ",
          des_title
        ),
        subtitle = paste0(
          des_eval,
          " The vertical axis uses a pseudo-log scale to display tail extrapolation."
        ),
        x = "Covariate x",
        y = "Mean interval width (pseudo-log scale)"
      ) +
      theme_pac_paper(base_size = 10.5) +
      theme(
        legend.position = "none"
      )

    ggsave(
      filename = file.path(
        design_out_dir,
        "app_conditional_width_parametric_full_normal.png"
      ),
      plot = p_width_parametric_full,
      width = 14,
      height = 10.5,
      dpi = 300
    )

  } else {

    # Uniform design: the full common support can be shown directly
    p_pointwise_width <- ggplot(
      pointwise_plot_df,
      aes(
        x = x,
        y = mean_width,
        color = Method,
        group = Method
      )
    ) +
      geom_line(linewidth = 0.75) +
      facet_grid(
        rows = vars(model_lab),
        cols = vars(ncal_lab),
        labeller = label_value,
        scales = "free_y",
        drop = FALSE
      ) +
      scale_color_manual(
        values = method_cols,
        drop = FALSE
      ) +
      labs(
        title = paste0(
          "Conditional mean interval width under ",
          des_title
        ),
        subtitle = des_eval,
        x = "Covariate x",
        y = "Mean interval width"
      ) +
      theme_pac_paper(base_size = 10.5)

    ggsave(
      filename = file.path(
        design_out_dir,
        paste0("app_conditional_width_", des, ".png")
      ),
      plot = p_pointwise_width,
      width = 14,
      height = 10.5,
      dpi = 300
    )
  }

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

  cat("[make_plot] saved plots to:", design_out_dir, "\n")
}

cat("[make_plot] all plots saved under:", out_dir, "\n")