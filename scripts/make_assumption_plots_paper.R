#!/usr/bin/env Rscript

# Paper plots for the balanced-four-DGP score-pivotality diagnostic.
# Example:
# Rscript scripts/make_assumption_plots_paper.R --tag=full_cv

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

parse_args <- function(args) {
  out <- list()
  for (arg in args) if (startsWith(arg, "--")) {
    bits <- strsplit(substring(arg, 3), "=", fixed = TRUE)[[1]]
    out[[bits[1]]] <- if (length(bits) == 2L) bits[2] else TRUE
  }
  out
}
args <- parse_args(commandArgs(trailingOnly = TRUE))
tag <- if (is.null(args$tag)) "full_cv" else args$tag
input_dir <- if (is.null(args$input_dir)) "results/sim/balanced4" else args$input_dir
out_dir <- if (is.null(args$out_dir)) file.path("fig/sim", paste0("balanced4_", tag), "diagnostics") else args$out_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ks_path <- file.path(input_dir, paste0("score_pivotality_ks_summary_", tag, ".csv"))
good_path <- file.path(input_dir, paste0("score_pivotality_good_summary_", tag, ".csv"))
estimation_path <- file.path(input_dir, paste0("assumption_estimation_summary_", tag, ".csv"))
cqr_path <- file.path(input_dir, paste0("cqr_assumption_summary_", tag, ".csv"))
oracle_path <- file.path(input_dir, paste0("oracle_residual_stability_summary_", tag, ".csv"))
stopifnot(file.exists(ks_path), file.exists(good_path), file.exists(estimation_path),
          file.exists(cqr_path), file.exists(oracle_path))
ks <- fread(ks_path)
good <- fread(good_path)
estimation <- fread(estimation_path)
cqr <- fread(cqr_path)
oracle <- fread(oracle_path)

methods <- c("SR-TI", "ASR-TI", "CQR-TI")
colors <- c("SR-TI" = "#D55E00", "ASR-TI" = "#CC79A7", "CQR-TI" = "#0072B2")
shapes <- c("SR-TI" = 16, "ASR-TI" = 18, "CQR-TI" = 17)
model_labels <- c(
  "1" = "Model 1: Gaussian", "2" = "Model 2: unit-variance t(5)",
  "3" = "Model 3: heteroscedastic Gaussian", "4" = "Model 4: X-dependent asymmetry"
)

clean_common <- function(d) {
  d[, model := factor(as.character(model), levels = names(model_labels), labels = unname(model_labels))]
  d[, design := factor(design, levels = c("normal", "uniform"), labels = c("Normal design", "Uniform design"))]
  d
}
clean_score <- function(d) {
  d <- clean_common(d)
  d[, Method := factor(Method, levels = methods)]
  d
}
ks <- clean_score(ks); good <- clean_score(good)
estimation <- clean_common(estimation); cqr <- clean_common(cqr); oracle <- clean_common(oracle)
paper_theme <- function(base_size = 10) {
  theme_classic(base_size = base_size) + theme(
    legend.position = "bottom", legend.title = element_blank(),
    strip.background = element_rect(fill = "grey93", color = "grey55"),
    strip.text = element_text(face = "bold"), axis.title = element_text(face = "bold")
  )
}
save_plot <- function(plot, stem, width, height) {
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot, width = width, height = height, dpi = 300)
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot, width = width, height = height)
}

make_ks_plot <- function(design_name, stem) {
  d <- ks[design == design_name]
  if (!nrow(d)) return(invisible(NULL))
  p <- ggplot(d, aes(n_train, ks_q90, color = Method, shape = Method, group = Method)) +
    geom_line(linewidth = .75) + geom_point(size = 2.2) +
    facet_wrap(~ model, nrow = 1) +
    scale_color_manual(values = colors, drop = FALSE) + scale_shape_manual(values = shapes, drop = FALSE) +
    scale_x_continuous(breaks = sort(unique(ks$n_train))) +
    labs(x = expression(n[tr]), y = expression("90th percentile of " * Delta(x))) + paper_theme()
  save_plot(p, stem, 13, 3.6)
}

# Main-text normal-design diagnostic and its uniform-design appendix counterpart.
make_ks_plot("Normal design", "fig_score_pivotality_ks_q90_normal")
make_ks_plot("Uniform design", "app_score_pivotality_ks_q90_uniform")

# Appendix: sensitivity to the KS tolerance.
p_good <- ggplot(good,
                 aes(n_train, px_good_pivotality, color = Method, shape = Method, group = Method)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey35") +
  geom_line(linewidth = .65) + geom_point(size = 1.9) +
  facet_grid(ks_eps + design ~ model) +
  scale_color_manual(values = colors, drop = FALSE) + scale_shape_manual(values = shapes, drop = FALSE) +
  scale_x_continuous(breaks = sort(unique(good$n_train))) + coord_cartesian(ylim = c(0, 1.03)) +
  labs(x = expression(n[tr]), y = "Pivotality-good proportion") + paper_theme(8.5)
save_plot(p_good, "app_pivotality_good", 14, 8)

# Appendix: Assumptions 6.1(A3)--(A4) and 6.2(A4') for the
# location--scale models only (Models 1--3).
est_long <- melt(
  estimation,
  id.vars = c("model", "design", "n_train"),
  measure.vars = c("mean_l2_error_mean", "var_l1_error_mean", "var_sup_error_mean"),
  variable.name = "diagnostic", value.name = "error"
)
est_long[, diagnostic := factor(diagnostic,
  levels = c("mean_l2_error_mean", "var_l1_error_mean", "var_sup_error_mean"),
  labels = c("Mean L2 error (A3)", "Scale L1 error (A4)", "Scale sup error (A4')"))]
p_residual <- ggplot(est_long, aes(n_train, error, group = diagnostic, color = diagnostic, shape = diagnostic)) +
  geom_line(linewidth = .7) + geom_point(size = 2) +
  facet_grid(design ~ model, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(estimation$n_train))) +
  labs(x = expression(n[tr]), y = "Estimation error", color = NULL, shape = NULL) + paper_theme(9)
save_plot(p_residual, "app_residual_nuisance_errors", 12.5, 6)

# Appendix: Assumption 6.3(A7), only where a common residual distribution exists.
tail_long <- melt(estimation,
  id.vars = c("model", "design", "n_train"),
  measure.vars = c("a_minus_abs_error_mean", "a_plus_abs_error_mean"),
  variable.name = "tail", value.name = "absolute_error")
tail_long[, tail := factor(tail, levels = c("a_minus_abs_error_mean", "a_plus_abs_error_mean"),
                           labels = c("a-", "a+"))]
p_tail <- ggplot(tail_long, aes(n_train, absolute_error, color = tail, shape = tail, group = tail)) +
  geom_line(linewidth = .7) + geom_point(size = 2) + facet_grid(design ~ model, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(estimation$n_train))) +
  labs(x = expression(n[tr]), y = "Absolute tail-scale error", color = NULL, shape = NULL) + paper_theme(9)
save_plot(p_tail, "app_asr_tail_scale_error", 12.5, 6)

# Appendix: Assumption 6.4(C1)--(C2), applicable to CQR in all four models.
cqr_long <- melt(cqr,
  id.vars = c("model", "design", "n_train"),
  measure.vars = c("qlo_l1_error_mean", "qhi_l1_error_mean"),
  variable.name = "endpoint", value.name = "l1_error")
cqr_long[, endpoint := factor(endpoint, levels = c("qlo_l1_error_mean", "qhi_l1_error_mean"),
                               labels = c("Lower endpoint", "Upper endpoint"))]
p_cqr_endpoint <- ggplot(cqr_long, aes(n_train, l1_error, color = endpoint, shape = endpoint, group = endpoint)) +
  geom_line(linewidth = .7) + geom_point(size = 2) + facet_grid(design ~ model, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(cqr$n_train))) +
  labs(x = expression(n[tr]), y = "Mean absolute endpoint error", color = NULL, shape = NULL) + paper_theme(9)
save_plot(p_cqr_endpoint, "app_cqr_endpoint_error", 14, 6)

p_cqr_cutoff <- ggplot(cqr, aes(n_train, cqr_cutoff_abs_q90, group = 1)) +
  geom_line(linewidth = .7) + geom_point(size = 2) + facet_grid(design ~ model, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(cqr$n_train))) +
  labs(x = expression(n[tr]), y = expression("90th percentile of " * abs(hat(q)[CQR]))) + paper_theme(9)
save_plot(p_cqr_cutoff, "app_cqr_cutoff_localization", 14, 6)

# A direct finite-sample counterpart of the C2 score-CDF condition.  This is
# reported next to cutoff localization; neither panel is presented as a proof
# of the asymptotic bracketing assumption.
p_cqr_cdf <- ggplot(cqr, aes(n_train, cqr_score_cdf_gap_at_zero_abs_q90, group = 1)) +
  geom_line(linewidth = .7) + geom_point(size = 2) + facet_grid(design ~ model, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(cqr$n_train))) +
  labs(x = expression(n[tr]),
       y = expression("90th percentile of " * abs(hat(F)[CQR](0) - C))) + paper_theme(9)
save_plot(p_cqr_cdf, "app_cqr_score_cdf_at_zero", 14, 6)

p_cqr_c2 <- ggplot(cqr, aes(n_train, c2_bracket_success, group = 1)) +
  geom_line(linewidth = .7) + geom_point(size = 2) + facet_grid(design ~ model) +
  scale_x_continuous(breaks = sort(unique(cqr$n_train))) + coord_cartesian(ylim = c(0, 1.03)) +
  labs(x = expression(n[tr]), y = "C2 bracket-proxy success probability") + paper_theme(9)
save_plot(p_cqr_c2, "app_cqr_c2_bracket_proxy", 14, 6)

# Appendix: direct oracle check for A1; Model 4 is intentionally non-pivotal.
p_oracle <- ggplot(oracle, aes(n_train, oracle_residual_ks_q90, group = 1)) +
  geom_line(linewidth = .7) + geom_point(size = 2) + facet_grid(design ~ model, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(oracle$n_train))) +
  labs(x = expression(n[tr]), y = "Oracle standardized-residual KS q90") + paper_theme(9)
save_plot(p_oracle, "app_oracle_residual_stability", 14, 6)

message("Saved pivotality figures to: ", out_dir)
