
plot_marginal <- function(coverage_df, out_path = NULL) {
  coverage_summary <- coverage_df %>%
    group_by(model, m) %>%
    summarize(mean_coverage = mean(coverage, na.rm = TRUE), .groups = "drop")

  p <- ggplot(coverage_summary, aes(x = m, y = mean_coverage,
                                   group = model, color = as.factor(model))) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "black") +
    scale_x_continuous(breaks = unique(coverage_summary$m)) +
    labs(
      title = "Coverage by Model and Sample Size (m)",
      x = "Sample Size (m)",
      y = "Mean Coverage",
      color = "Model"
    ) +
    theme_minimal() +
    ylim(0, 1)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(out_path, plot = p, width = 8, height = 5, dpi = 300)
  }
  p
}

plot_pointwise <- function(coverage_df,
                           out_path = NULL,
                           confidence_level = 0.95,
                           content_level = 0.90) {

  coverage_df <- .prepare_plot_df(coverage_df) %>%
    dplyr::filter(model != 6)

  p <- ggplot(
    coverage_df,
    aes(
      x = x,
      y = coverage,
      color = Method,
      group = Method
    )
  ) +
    geom_line(linewidth = 0.85) +
    geom_hline(
      yintercept = confidence_level,
      color = "black",
      linetype = "dashed",
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
      title = "Pointwise PAC success proportion",
      subtitle = paste0(
        "Models 1--5 only; y-axis = P(content(x) >= ",
        content_level, "); dashed line = ", confidence_level
      ),
      x = "x",
      y = "PAC success proportion",
      color = "Method"
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom")

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(out_path, plot = p, width = 14, height = 9, dpi = 300)
  }

  p
}

plot_na_rate <- function(coverage_df,
                         out_path = NULL) {

  coverage_df <- .prepare_plot_df(coverage_df) %>%
    dplyr::filter(model != 6)

  p <- ggplot(
    coverage_df,
    aes(
      x = x,
      y = na_proportion,
      color = Method,
      group = Method
    )
  ) +
    geom_line(linewidth = 0.85) +
    facet_grid(
      rows = vars(model),
      cols = vars(n_cal),
      labeller = label_both,
      scales = "free_x"
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      title = "Proportion of infeasible threshold estimation",
      subtitle = "Models 1--5 only",
      x = "x",
      y = "NA proportion",
      color = "Method"
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom")

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(out_path, plot = p, width = 14, height = 9, dpi = 300)
  }

  p
}

plot_width <- function(coverage_df,
                       out_path = NULL) {

  coverage_df <- .prepare_plot_df(coverage_df) %>%
    dplyr::filter(model != 6)

  p <- ggplot(
    coverage_df,
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

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(out_path, plot = p, width = 14, height = 9, dpi = 300)
  }

  p
}


model6_summary <- df %>%
  dplyr::filter(model == 6) %>%
  dplyr::group_by(model, n_train, n_cal, n_test, Method) %>%
  dplyr::summarise(
    avg_pac_success = mean(coverage, na.rm = TRUE),
    avg_width = mean(mean_width, na.rm = TRUE),
    avg_na_prop = mean(na_proportion, na.rm = TRUE),
    min_pac_success = min(coverage, na.rm = TRUE),
    q25_pac_success = quantile(coverage, 0.25, na.rm = TRUE),
    median_pac_success = median(coverage, na.rm = TRUE),
    q75_pac_success = quantile(coverage, 0.75, na.rm = TRUE),
    max_pac_success = max(coverage, na.rm = TRUE),
    .groups = "drop"
  )

readr::write_csv(
  model6_summary,
  file.path(out_dir, "summary_model6_highdim.csv")
)

cat("[make_plot] saved model 6 summary:",
    file.path(out_dir, "summary_model6_highdim.csv"), "\n")