# R/70_plots.R
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

plot_pointwise <- function(coverage_df, out_path = NULL) {
  p <- ggplot(coverage_df, aes(x = x, y = coverage)) +
    geom_line(color = "blue") +
    geom_hline(yintercept = 0.95, color = "red", linetype = "dashed") +
    facet_grid(rows = vars(model), cols = vars(m), labeller = label_both) +
    labs(
      title = "Pointwise Coverage Proportion (content ≥ 0.90)",
      x = "x", y = "Coverage Proportion"
    ) +
    ylim(0, 1) +
    theme_minimal(base_size = 13)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(out_path, plot = p, width = 14, height = 10, dpi = 300)
  }
  p
}

plot_na_rate <- function(coverage_df, out_path = NULL) {
  p <- ggplot(coverage_df, aes(x = x, y = na_proportion)) +
    geom_line(color = "gray40", linewidth = 1) +
    facet_grid(rows = vars(model), cols = vars(m), labeller = label_both) +
    labs(
      title = "Proportion of NA (λ_hat estimation failure)",
      x = "x", y = "NA Proportion"
    ) +
    theme_minimal(base_size = 13)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(out_path, plot = p, width = 14, height = 10, dpi = 300)
  }
  p
}

plot_width <- function(coverage_df, out_path = NULL) {
  p <- ggplot(coverage_df, aes(x = x, y = mean_width)) +
    geom_line(color = "darkgreen") +
    facet_grid(rows = vars(model), cols = vars(m), labeller = label_both) +
    labs(
      title = "Pointwise Mean Interval Width",
      x = "x", y = "Mean Width"
    ) +
    theme_minimal(base_size = 13)

  if (!is.null(out_path)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(out_path, plot = p, width = 14, height = 10, dpi = 300)
  }
  p
}
