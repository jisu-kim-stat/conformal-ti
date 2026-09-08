# Create paper-ready figures from the five-DGP simulation CSV files.
#
# Example:
# Rscript scripts/sim/make_plot.R --tag=full_cv --out_dir=fig/sim/balanced4_full_cv

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    parts <- strsplit(substring(arg, 3), "=", fixed = TRUE)[[1]]
    out[[parts[1]]] <- if (length(parts) == 2L) parts[2] else TRUE
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
tag <- if (is.null(args$tag)) "full_cv" else args$tag
input_dir <- if (is.null(args$input_dir)) "results/sim/balanced4" else args$input_dir
out_dir <- if (is.null(args$out_dir)) file.path("fig/sim", paste0("balanced4_", tag)) else args$out_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

pointwise_path <- file.path(input_dir, paste0("pointwise_", tag, ".csv"))
marginal_path <- file.path(input_dir, paste0("marginal_", tag, ".csv"))
px_good_path <- file.path(input_dir, paste0("px_good_", tag, ".csv"))
stopifnot(file.exists(pointwise_path), file.exists(marginal_path), file.exists(px_good_path))

method_levels <- c("SR-TI", "ASR-TI", "CQR-TI", "Parametric-TI", "GY-TI")
method_cols <- c(
  "SR-TI" = "#D55E00", "ASR-TI" = "#CC79A7", "CQR-TI" = "#0072B2",
  "Parametric-TI" = "#555555", "GY-TI" = "#009E73"
)
method_shapes <- c("SR-TI" = 16, "ASR-TI" = 18, "CQR-TI" = 17,
                   "Parametric-TI" = 15, "GY-TI" = 3)
model_labels <- c(
  "1" = "Model 1: Gaussian",
  "2" = "Model 2: t(3)",
  "3" = "Model 3: heteroscedastic Gaussian",
  "4" = "Model 4: global skewness",
  "5" = "Model 5: X-dependent skewness"
)

paper_theme <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      legend.position = "bottom", legend.title = element_blank(),
      strip.background = element_rect(fill = "grey93", color = "grey55"),
      strip.text = element_text(face = "bold"),
      axis.title = element_text(face = "bold"),
      panel.spacing = grid::unit(0.8, "lines")
    )
}

save_plot <- function(plot, stem, width, height) {
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot, width = width, height = height)
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot, width = width, height = height, dpi = 300)
}

clean_data <- function(df) {
  df %>% mutate(
    model = factor(as.character(model), levels = names(model_labels), labels = unname(model_labels)),
    Method = factor(Method, levels = method_levels),
    design = factor(design, levels = c("normal", "uniform"), labels = c("Normal design", "Uniform design")),
    n_cal = as.integer(n_cal)
  )
}

pointwise <- clean_data(read_csv(pointwise_path, show_col_types = FALSE))
marginal <- clean_data(read_csv(marginal_path, show_col_types = FALSE))
px_good <- clean_data(read_csv(px_good_path, show_col_types = FALSE))

for (des in levels(px_good$design)) {
  px_des <- filter(px_good, design == des)
  mar_des <- filter(marginal, design == des)
  pw_des <- filter(pointwise, design == des, abs(epsilon - 0.02) < 1e-12)

  # Main-text conditional diagnostic at the prespecified slack epsilon = 0.02.
  p_px <- ggplot(filter(px_des, abs(epsilon - 0.02) < 1e-12),
                 aes(n_cal, px_good_proportion, color = Method, shape = Method, group = Method)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey35") +
    geom_line(linewidth = 0.7) + geom_point(size = 2.2) +
    facet_wrap(~ model, nrow = 1) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(px_des$n_cal))) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(x = expression(n[cal]), y = expression(hat(p)[X]^good(0.02))) +
    paper_theme(10)
  save_plot(p_px, paste0("fig_px_good_eps002_", gsub(" ", "_", tolower(des))), 13, 3.5)

  # Appendix: sensitivity over the complete slack grid.
  p_slack <- ggplot(px_des,
                    aes(n_cal, px_good_proportion, color = Method, shape = Method, group = Method)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey35") +
    geom_line(linewidth = 0.6) + geom_point(size = 1.8) +
    facet_grid(epsilon ~ model) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(px_des$n_cal))) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(x = expression(n[cal]), y = expression(hat(p)[X]^good(epsilon))) +
    paper_theme(8.5)
  save_plot(p_slack, paste0("app_px_good_slack_", gsub(" ", "_", tolower(des))), 14, 9)

  # Appendix: marginal PAC success and mean width.
  p_marginal <- ggplot(mar_des,
                       aes(n_cal, marginal_pac_success, color = Method, shape = Method, group = Method)) +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "grey35") +
    geom_line(linewidth = 0.7) + geom_point(size = 2.2) +
    facet_wrap(~ model, nrow = 1) +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(mar_des$n_cal))) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(x = expression(n[cal]), y = "Marginal PAC success") + paper_theme(10)
  save_plot(p_marginal, paste0("app_marginal_pac_", gsub(" ", "_", tolower(des))), 13, 3.5)

  p_width <- ggplot(mar_des,
                    aes(n_cal, average_width_mean, color = Method, shape = Method, group = Method)) +
    geom_line(linewidth = 0.7) + geom_point(size = 2.2) +
    facet_wrap(~ model, nrow = 1, scales = "free_y") +
    scale_color_manual(values = method_cols, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(mar_des$n_cal))) +
    labs(x = expression(n[cal]), y = "Mean interval width") + paper_theme(10)
  save_plot(p_width, paste0("app_width_", gsub(" ", "_", tolower(des))), 13, 3.5)

  # Appendix: pointwise conditional-success curves at epsilon = 0.02.
  p_pointwise <- ggplot(pw_des,
                        aes(x, pointwise_success, color = Method, group = Method)) +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "grey35") +
    geom_line(linewidth = 0.55) +
    facet_grid(model ~ n_cal, scales = "free_x") +
    scale_color_manual(values = method_cols, drop = FALSE) +
    coord_cartesian(ylim = c(0, 1.03)) +
    labs(x = "Covariate x", y = "Conditional-success probability") + paper_theme(8.5)
  save_plot(p_pointwise, paste0("app_pointwise_success_eps002_", gsub(" ", "_", tolower(des))), 14, 9)
}

message("Saved figures to: ", out_dir)
