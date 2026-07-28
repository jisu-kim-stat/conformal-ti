# scripts/plot_score_bootstrap_width_alt_dgp.R
#
# Run from the project root:
#
#   Rscript scripts/plot_score_bootstrap_width_alt_dgp.R


suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
})


# ============================================================
# 1. Paths
# ============================================================

result_dir <- "results/sim/bootstrap_score_process"

summary_path <- file.path(
  result_dir,
  "score_bootstrap_summary_alt_dgp.csv"
)

width_path <- file.path(
  result_dir,
  "score_bootstrap_width_comparison_alt_dgp.csv"
)

plot_dir <- file.path(
  result_dir,
  "plots"
)

dir.create(
  plot_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

if (!file.exists(summary_path)) {
  stop(
    "Missing summary file: ",
    summary_path,
    call. = FALSE
  )
}

if (!file.exists(width_path)) {
  stop(
    "Missing width file: ",
    width_path,
    call. = FALSE
  )
}

summary_df <- readr::read_csv(
  summary_path,
  show_col_types = FALSE
)

width_df <- readr::read_csv(
  width_path,
  show_col_types = FALSE
)


# ============================================================
# 2. Labels and style
# ============================================================

method_levels <- c(
  "SR-TI",
  "ASR-TI",
  "CQR-TI"
)

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

calibration_levels <- c(
  "plain",
  "binomial",
  "hoeffding"
)

calibration_labels <- c(
  "plain" = "Plain",
  "binomial" = "Binomial rank",
  "hoeffding" = "Hoeffding"
)

buffer_levels <- c(
  "Original",
  "Bootstrap"
)

buffer_linetypes <- c(
  "Original" = "solid",
  "Bootstrap" = "dashed"
)

theme_bootstrap_paper <- function(base_size = 11) {
  ggplot2::theme_classic(base_size = base_size) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold",
        size = base_size + 3,
        hjust = 0
      ),
      plot.subtitle = ggplot2::element_text(
        size = base_size,
        color = "grey25",
        margin = ggplot2::margin(b = 8)
      ),
      axis.title = ggplot2::element_text(
        face = "bold"
      ),
      axis.text = ggplot2::element_text(
        color = "black"
      ),
      panel.grid.major.y = ggplot2::element_line(
        color = "grey88",
        linewidth = 0.3
      ),
      panel.grid.major.x = ggplot2::element_line(
        color = "grey93",
        linewidth = 0.25
      ),
      panel.grid.minor = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(
        fill = "grey93",
        color = "grey55",
        linewidth = 0.35
      ),
      strip.text = ggplot2::element_text(
        face = "bold"
      ),
      legend.position = "bottom",
      legend.title = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(
        10,
        12,
        10,
        12
      )
    )
}


design_title <- function(design) {
  if (design == "uniform") {
    return("uniform covariate design")
  }

  if (design == "normal") {
    return("normal covariate design")
  }

  paste0(design, " design")
}


empty_plot <- function(title, subtitle) {
  ggplot2::ggplot() +
    ggplot2::annotate(
      "text",
      x = 0,
      y = 0,
      label = "No feasible bootstrap replications",
      size = 5,
      fontface = "bold"
    ) +
    ggplot2::xlim(-1, 1) +
    ggplot2::ylim(-1, 1) +
    ggplot2::labs(
      title = title,
      subtitle = subtitle
    ) +
    ggplot2::theme_void(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold"
      ),
      plot.subtitle = ggplot2::element_text(
        color = "grey25"
      )
    )
}


# ============================================================
# 3. Type cleanup
# ============================================================

summary_df <- summary_df %>%
  dplyr::mutate(
    model = as.integer(model),
    n_cal = as.integer(n_cal),
    design = as.character(design),
    Method = factor(
      Method,
      levels = method_levels
    ),
    Calibration = factor(
      Calibration,
      levels = calibration_levels,
      labels = unname(
        calibration_labels[
          calibration_levels
        ]
      )
    ),
    Buffer = factor(
      Buffer,
      levels = buffer_levels
    ),
    model_lab = factor(
      paste0("Model ", model),
      levels = paste0(
        "Model ",
        sort(unique(model))
      )
    )
  )

width_df <- width_df %>%
  dplyr::mutate(
    model = as.integer(model),
    n_cal = as.integer(n_cal),
    design = as.character(design),
    Method = factor(
      Method,
      levels = method_levels
    ),
    Calibration = factor(
      Calibration,
      levels = calibration_levels,
      labels = unname(
        calibration_labels[
          calibration_levels
        ]
      )
    ),
    model_lab = factor(
      paste0("Model ", model),
      levels = paste0(
        "Model ",
        sort(unique(model))
      )
    )
  )


# ============================================================
# 4. Paired width-ratio summary
# ============================================================

width_ratio_summary <- width_df %>%
  dplyr::filter(
    original_feasible,
    bootstrap_feasible,
    is.finite(width_ratio),
    width_ratio > 0
  ) %>%
  dplyr::group_by(
    model,
    model_lab,
    design,
    n_cal,
    Method,
    Calibration
  ) %>%
  dplyr::summarise(
    n_feasible = dplyr::n(),
    width_ratio_mean = mean(
      width_ratio,
      na.rm = TRUE
    ),
    width_ratio_median = stats::median(
      width_ratio,
      na.rm = TRUE
    ),
    width_ratio_q25 = stats::quantile(
      width_ratio,
      probs = 0.25,
      names = FALSE,
      na.rm = TRUE
    ),
    width_ratio_q75 = stats::quantile(
      width_ratio,
      probs = 0.75,
      names = FALSE,
      na.rm = TRUE
    ),
    width_difference_mean = mean(
      width_difference,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

readr::write_csv(
  width_ratio_summary,
  file.path(
    plot_dir,
    "summary_paired_width_ratio.csv"
  )
)


# ============================================================
# 5. Plots by covariate design
# ============================================================

design_vec <- sort(
  unique(summary_df$design)
)

for (des in design_vec) {
  cat("[plot] design:", des, "\n")

  des_title <- design_title(des)

  summary_des <- summary_df %>%
    dplyr::filter(design == des)

  ratio_des <- width_ratio_summary %>%
    dplyr::filter(design == des)

  n_cal_breaks <- sort(
    unique(summary_des$n_cal)
  )

  # ----------------------------------------------------------
  # Figure 1: Bootstrap feasibility
  # ----------------------------------------------------------

  feasibility_df <- summary_des %>%
    dplyr::filter(Buffer == "Bootstrap")

  p_feasible <- ggplot2::ggplot(
    feasibility_df,
    ggplot2::aes(
      x = n_cal,
      y = feasible_rate,
      color = Method,
      shape = Method,
      group = Method
    )
  ) +
    ggplot2::geom_hline(
      yintercept = 1,
      linetype = "dotted",
      color = "grey35",
      linewidth = 0.4
    ) +
    ggplot2::geom_line(
      linewidth = 0.8,
      na.rm = TRUE
    ) +
    ggplot2::geom_point(
      size = 2.3,
      na.rm = TRUE
    ) +
    ggplot2::facet_grid(
      rows = ggplot2::vars(Calibration),
      cols = ggplot2::vars(model_lab)
    ) +
    ggplot2::scale_color_manual(
      values = method_cols,
      drop = FALSE
    ) +
    ggplot2::scale_shape_manual(
      values = method_shapes,
      drop = FALSE
    ) +
    ggplot2::scale_x_continuous(
      breaks = n_cal_breaks
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.25)
    ) +
    ggplot2::labs(
      title = paste0(
        "Bootstrap feasibility under ",
        des_title
      ),
      subtitle = paste0(
        "A finite threshold requires the buffered calibration ",
        "level to remain at most one."
      ),
      x = "Calibration sample size",
      y = "Feasible replication proportion"
    ) +
    theme_bootstrap_paper(base_size = 9.5)

  ggplot2::ggsave(
    filename = file.path(
      plot_dir,
      paste0(
        "fig_bootstrap_feasible_rate_",
        des,
        ".png"
      )
    ),
    plot = p_feasible,
    width = 15,
    height = 8.5,
    dpi = 300
  )

  # ----------------------------------------------------------
  # Figure 2: Original calibration width
  #
  # This directly shows the small-n inflation caused by the
  # Hoeffding correction, before adding any bootstrap buffer.
  # ----------------------------------------------------------

  original_width_df <- summary_des %>%
    dplyr::filter(Buffer == "Original")

  p_original_width <- ggplot2::ggplot(
    original_width_df,
    ggplot2::aes(
      x = n_cal,
      y = average_width_mean,
      color = Method,
      shape = Method,
      linetype = Calibration,
      group = interaction(
        Method,
        Calibration
      )
    )
  ) +
    ggplot2::geom_line(
      linewidth = 0.8,
      na.rm = TRUE
    ) +
    ggplot2::geom_point(
      size = 2.25,
      na.rm = TRUE
    ) +
    ggplot2::facet_wrap(
      ~ model_lab,
      nrow = 2,
      scales = "free_y"
    ) +
    ggplot2::scale_color_manual(
      values = method_cols,
      drop = FALSE
    ) +
    ggplot2::scale_shape_manual(
      values = method_shapes,
      drop = FALSE
    ) +
    ggplot2::scale_x_continuous(
      breaks = n_cal_breaks
    ) +
    ggplot2::labs(
      title = paste0(
        "Original interval width under ",
        des_title
      ),
      subtitle = paste0(
        "Comparison of Plain, Binomial-rank, and Hoeffding ",
        "calibration without the bootstrap buffer."
      ),
      x = "Calibration sample size",
      y = "Average interval width",
      linetype = NULL
    ) +
    theme_bootstrap_paper(base_size = 11)

  ggplot2::ggsave(
    filename = file.path(
      plot_dir,
      paste0(
        "fig_original_width_by_calibration_",
        des,
        ".png"
      )
    ),
    plot = p_original_width,
    width = 13,
    height = 8,
    dpi = 300
  )

  # ----------------------------------------------------------
  # Figure 3: Original vs bootstrap absolute width
  # One file per calibration rule.
  #
  # Bootstrap width is averaged only over feasible replications.
  # Always interpret this figure together with Figure 1.
  # ----------------------------------------------------------

  for (calibration_name in levels(
    summary_des$Calibration
  )) {
    width_calibration_df <- summary_des %>%
      dplyr::filter(
        Calibration == calibration_name
      )

    calibration_file <- tolower(
      gsub(
        "[^A-Za-z0-9]+",
        "_",
        calibration_name
      )
    )

    p_absolute_width <- ggplot2::ggplot(
      width_calibration_df,
      ggplot2::aes(
        x = n_cal,
        y = average_width_mean,
        color = Method,
        shape = Method,
        linetype = Buffer,
        group = interaction(
          Method,
          Buffer
        )
      )
    ) +
      ggplot2::geom_line(
        linewidth = 0.8,
        na.rm = TRUE
      ) +
      ggplot2::geom_point(
        size = 2.25,
        na.rm = TRUE
      ) +
      ggplot2::facet_wrap(
        ~ model_lab,
        nrow = 2,
        scales = "free_y"
      ) +
      ggplot2::scale_color_manual(
        values = method_cols,
        drop = FALSE
      ) +
      ggplot2::scale_shape_manual(
        values = method_shapes,
        drop = FALSE
      ) +
      ggplot2::scale_linetype_manual(
        values = buffer_linetypes,
        drop = FALSE
      ) +
      ggplot2::scale_x_continuous(
        breaks = n_cal_breaks
      ) +
      ggplot2::labs(
        title = paste0(
          "Original versus bootstrap width: ",
          calibration_name
        ),
        subtitle = paste0(
          des_title,
          ". Bootstrap means use feasible replications only; ",
          "check the feasibility plot."
        ),
        x = "Calibration sample size",
        y = "Average interval width",
        linetype = NULL
      ) +
      theme_bootstrap_paper(base_size = 11)

    ggplot2::ggsave(
      filename = file.path(
        plot_dir,
        paste0(
          "fig_width_original_vs_bootstrap_",
          calibration_file,
          "_",
          des,
          ".png"
        )
      ),
      plot = p_absolute_width,
      width = 13,
      height = 8,
      dpi = 300
    )
  }

  # ----------------------------------------------------------
  # Figure 4: Paired bootstrap/original width ratio
  # One file per calibration rule.
  # ----------------------------------------------------------

  for (calibration_name in levels(
    summary_des$Calibration
  )) {
    ratio_calibration_df <- ratio_des %>%
      dplyr::filter(
        Calibration == calibration_name
      )

    calibration_file <- tolower(
      gsub(
        "[^A-Za-z0-9]+",
        "_",
        calibration_name
      )
    )

    ratio_title <- paste0(
      "Paired width inflation: ",
      calibration_name
    )

    ratio_subtitle <- paste0(
      des_title,
      ". Points are medians and bars are interquartile ranges ",
      "over feasible paired replications."
    )

    if (nrow(ratio_calibration_df) == 0L) {
      p_ratio <- empty_plot(
        title = ratio_title,
        subtitle = ratio_subtitle
      )
    } else {
      p_ratio <- ggplot2::ggplot(
        ratio_calibration_df,
        ggplot2::aes(
          x = n_cal,
          y = width_ratio_median,
          color = Method,
          shape = Method,
          group = Method
        )
      ) +
        ggplot2::geom_hline(
          yintercept = 1,
          linetype = "dashed",
          color = "grey25",
          linewidth = 0.45
        ) +
        ggplot2::geom_errorbar(
          ggplot2::aes(
            ymin = width_ratio_q25,
            ymax = width_ratio_q75
          ),
          width = 0,
          linewidth = 0.45,
          na.rm = TRUE
        ) +
        ggplot2::geom_line(
          linewidth = 0.8,
          na.rm = TRUE
        ) +
        ggplot2::geom_point(
          size = 2.4,
          na.rm = TRUE
        ) +
        ggplot2::facet_wrap(
          ~ model_lab,
          nrow = 2,
          scales = "free_y"
        ) +
        ggplot2::scale_color_manual(
          values = method_cols,
          drop = FALSE
        ) +
        ggplot2::scale_shape_manual(
          values = method_shapes,
          drop = FALSE
        ) +
        ggplot2::scale_x_continuous(
          breaks = n_cal_breaks
        ) +
        ggplot2::scale_y_log10() +
        ggplot2::labs(
          title = ratio_title,
          subtitle = ratio_subtitle,
          x = "Calibration sample size",
          y = "Bootstrap width / original width"
        ) +
        theme_bootstrap_paper(base_size = 11)
    }

    ggplot2::ggsave(
      filename = file.path(
        plot_dir,
        paste0(
          "fig_paired_width_ratio_",
          calibration_file,
          "_",
          des,
          ".png"
        )
      ),
      plot = p_ratio,
      width = 13,
      height = 8,
      dpi = 300
    )
  }

  # ----------------------------------------------------------
  # Figure 5: Finite-grid simultaneous empirical success
  # ----------------------------------------------------------

  p_simultaneous <- ggplot2::ggplot(
    summary_des,
    ggplot2::aes(
      x = n_cal,
      y = simultaneous_success,
      color = Method,
      shape = Method,
      linetype = Buffer,
      group = interaction(
        Method,
        Buffer
      )
    )
  ) +
    ggplot2::geom_hline(
      yintercept = 0.95,
      linetype = "dotted",
      color = "grey25",
      linewidth = 0.4
    ) +
    ggplot2::geom_line(
      linewidth = 0.75,
      na.rm = TRUE
    ) +
    ggplot2::geom_point(
      size = 2.1,
      na.rm = TRUE
    ) +
    ggplot2::facet_grid(
      rows = ggplot2::vars(Calibration),
      cols = ggplot2::vars(model_lab)
    ) +
    ggplot2::scale_color_manual(
      values = method_cols,
      drop = FALSE
    ) +
    ggplot2::scale_shape_manual(
      values = method_shapes,
      drop = FALSE
    ) +
    ggplot2::scale_linetype_manual(
      values = buffer_linetypes,
      drop = FALSE
    ) +
    ggplot2::scale_x_continuous(
      breaks = n_cal_breaks
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.25)
    ) +
    ggplot2::labs(
      title = paste0(
        "Finite-grid simultaneous empirical success under ",
        des_title
      ),
      subtitle = paste0(
        "Dashed/dotted reference at 0.95. Missing bootstrap ",
        "curves indicate infeasible calibration."
      ),
      x = "Calibration sample size",
      y = "Simultaneous success"
    ) +
    theme_bootstrap_paper(base_size = 9.5)

  ggplot2::ggsave(
    filename = file.path(
      plot_dir,
      paste0(
        "fig_simultaneous_success_",
        des,
        ".png"
      )
    ),
    plot = p_simultaneous,
    width = 15,
    height = 8.5,
    dpi = 300
  )
}


cat(
  "Saved bootstrap plots to:",
  plot_dir,
  "\n"
)
