#!/usr/bin/env Rscript

# Empirical score-pivotality diagnostic for the four-DGP simulation suite.
# This is a diagnostic for fitted-score stability, not a validity proof.
# Example:
# Rscript scripts/sim/run_score_pivotality_balanced4.R --reps=200 --cores=4 --tag=full_cv

source("R/packages.R")
source("R/sim/base_mean.R")
source("R/sim/data_generate_balanced4.R")
source("R/sim/truth_content_balanced4.R")
source("R/sim/fit_srti.R")
source("R/sim/fit_cqr.R")
source("R/sim/lambda_hoeffding.R")

suppressPackageStartupMessages({ library(data.table) })

parse_args <- function(args) {
  out <- list()
  for (arg in args) if (startsWith(arg, "--")) {
    bits <- strsplit(substring(arg, 3), "=", fixed = TRUE)[[1]]
    out[[bits[1]]] <- if (length(bits) == 2L) bits[2] else TRUE
  }
  out
}
parse_ints <- function(x, default) {
  if (is.null(x)) return(default)
  as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
}
args <- parse_args(commandArgs(trailingOnly = TRUE))
M <- as.integer(if (is.null(args$reps)) 200L else args$reps)
n_cores <- as.integer(if (is.null(args$cores)) 1L else args$cores)
models <- parse_ints(args$models, 1:4)
designs <- if (is.null(args$designs)) c("normal", "uniform") else strsplit(args$designs, ",", fixed = TRUE)[[1]]
n_train_vec <- parse_ints(args$ntrain, c(200L, 500L, 1000L))
tag <- if (is.null(args$tag)) "full_cv" else args$tag
n_marginal <- as.integer(if (is.null(args$n_marginal)) 3000L else args$n_marginal)
n_conditional <- as.integer(if (is.null(args$n_conditional)) 500L else args$n_conditional)
n_x_grid <- as.integer(if (is.null(args$n_x_grid)) 61L else args$n_x_grid)
n_eval <- as.integer(if (is.null(args$n_eval)) 401L else args$n_eval)
stopifnot(all(models %in% 1:4), all(designs %in% c("normal", "uniform")), M >= 1L)

content <- .90
tau <- (1 - content) / 2
cqr_df_grid <- c(4L, 6L, 8L, 10L, 12L)
ks_eps_grid <- c(.05, .10, .15)

eval_grid <- function(n, design) {
  p <- (seq_len(n) - .5) / n
  if (design == "normal") qnorm(p) else qunif(p, -2, 2)
}
ks_distance <- function(x, y) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  if (length(x) < 5L || length(y) < 5L) return(NA_real_)
  grid <- sort(unique(c(x, y)))
  max(abs(ecdf(x)(grid) - ecdf(y)(grid)))
}
true_sigma2 <- function(model_id, x) {
  switch(as.character(model_id),
    "1" = rep(1, length(x)),
    "2" = rep(1, length(x)), # t_5 / sqrt(5/3) has unit variance
    "3" = (1 + abs(x))^2,
    "4" = {
      p <- plogis(2 * x)
      sigma_minus <- .7 + .5 * p
      sigma_plus <- 1.3 - .5 * p
      mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
      .5 * (sigma_minus^2 + sigma_plus^2) - mean_eps^2
    }
  )
}
true_asym_shape <- function(model_id) {
  if (model_id %in% c(1L, 3L)) {
    return(c(a_minus = abs(qnorm(tau)), a_plus = qnorm(1 - tau)))
  }
  if (model_id == 2L) {
    return(c(a_minus = abs(qt(tau, df = 5) / sqrt(5 / 3)),
             a_plus = qt(1 - tau, df = 5) / sqrt(5 / 3)))
  }
  stop("Model 4 has no common standardized tail-scale target.")
}
true_endpoint <- function(model_id, x, probability) {
  base <- base_mean(x)
  if (model_id == 1L) return(base + qnorm(probability))
  if (model_id == 2L) return(base + qt(probability, df = 5) / sqrt(5 / 3))
  if (model_id == 3L) return(base + (1 + abs(x)) * qnorm(probability))
  p <- plogis(2 * x)
  sigma_minus <- .7 + .5 * p
  sigma_plus <- 1.3 - .5 * p
  mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
  scale <- if (probability <= .5) sigma_minus else sigma_plus
  base + scale * qnorm(probability) - mean_eps
}

run_one <- function(model_id, design, n_train, rep_id) {
  set.seed(1000000L * model_id + 10000L * n_train + 1000L * match(design, c("normal", "uniform")) + rep_id)
  train <- generate_data(model_id, n_train, design = design)
  fit_mean <- fit_mean_model_auto(train$x, train$y, model_id)
  fit_var <- fit_var_model_auto(train$x, train$y, fit_mean, model_id)
  mu_train <- predict_mean_auto(fit_mean, train$x, model_id)
  sd_train <- sqrt(pmax(predict_var_auto(fit_var, train$x, model_id), 1e-8))
  shape <- find_asym_shape((train$y - mu_train) / sd_train, tau = tau, eps = 1e-6)

  estimation <- NULL
  if (model_id <= 3L) {
    x_eval <- eval_grid(n_eval, design)
    mu_eval <- predict_mean_auto(fit_mean, x_eval, model_id)
    var_eval <- predict_var_auto(fit_var, x_eval, model_id)
    shape_true <- true_asym_shape(model_id)
    estimation <- data.table(
      model = model_id, design = design, n_train = n_train, repetition = rep_id,
      mean_l2_error = mean((mu_eval - base_mean(x_eval))^2),
      var_l1_error = mean(abs(var_eval - true_sigma2(model_id, x_eval))),
      var_sup_error = max(abs(var_eval - true_sigma2(model_id, x_eval))),
      a_minus_hat = shape[["a_minus"]], a_plus_hat = shape[["a_plus"]],
      a_minus_true = shape_true[["a_minus"]], a_plus_true = shape_true[["a_plus"]],
      a_minus_abs_error = abs(shape[["a_minus"]] - shape_true[["a_minus"]]),
      a_plus_abs_error = abs(shape[["a_plus"]] - shape_true[["a_plus"]])
    )
  }

  folds <- sample(rep(seq_len(5L), length.out = n_train))
  qlo <- fit_quantile_model_auto(train$x, train$y, tau, model_id,
    design = design, basis_type = "cv_fixed_ns", candidate_dfs = cqr_df_grid, fold_id = folds)
  qhi <- fit_quantile_model_auto(train$x, train$y, 1 - tau, model_id,
    design = design, basis_type = "cv_fixed_ns", candidate_dfs = cqr_df_grid, fold_id = folds)

  x_eval_cqr <- eval_grid(n_eval, design)
  qlo_eval <- predict_quantile_auto(qlo, x_eval_cqr, model_id)
  qhi_eval <- predict_quantile_auto(qhi, x_eval_cqr, model_id)
  endpoint_error <- abs(qlo_eval - true_endpoint(model_id, x_eval_cqr, tau)) +
    abs(qhi_eval - true_endpoint(model_id, x_eval_cqr, 1 - tau))
  cqr_diagnostic <- data.table(
    model = model_id, design = design, n_train = n_train, repetition = rep_id,
    qlo_l1_error = mean(abs(qlo_eval - true_endpoint(model_id, x_eval_cqr, tau))),
    qhi_l1_error = mean(abs(qhi_eval - true_endpoint(model_id, x_eval_cqr, 1 - tau))),
    endpoint_l1_error = mean(endpoint_error),
    endpoint_l1_error_q90_x = unname(quantile(endpoint_error, .90)),
    endpoint_l1_error_max_x = max(endpoint_error),
    qlo_selected_df = attr(qlo, "cqr_selected_df"),
    qhi_selected_df = attr(qhi, "cqr_selected_df")
  )

  calibration <- generate_data(model_id, n_train, design = design)
  score_cal_cqr <- pmax(
    predict_quantile_auto(qlo, calibration$x, model_id) - calibration$y,
    calibration$y - predict_quantile_auto(qhi, calibration$x, model_id)
  )
  cqr_diagnostic[, cqr_cutoff := find_score_cutoff(
    mis = 1 - content, alpha = .05, score = score_cal_cqr,
    calibration_rule = "hoeffding"
  )]

  marginal <- generate_data(model_id, n_marginal, design = design)
  marginal_scores <- function(x, y) {
    mu <- predict_mean_auto(fit_mean, x, model_id)
    sd <- sqrt(pmax(predict_var_auto(fit_var, x, model_id), 1e-8))
    z <- (y - mu) / sd
    list(
      `SR-TI` = abs(z),
      `ASR-TI` = asym_residual_score(z, shape[["a_minus"]], shape[["a_plus"]]),
      `CQR-TI` = pmax(predict_quantile_auto(qlo, x, model_id) - y,
                      y - predict_quantile_auto(qhi, x, model_id))
    )
  }
  marginal_score <- marginal_scores(marginal$x, marginal$y)
  # Operational C2 diagnostic.  We evaluate the CDF brackets at the
  # deterministic probes h_n=n^{-1/4} and s_n=h_n/4, which satisfy
  # h_n -> 0, s_n -> 0, and n s_n^2 -> infinity.  This finite-sample
  # check is evidence for, not a proof of, Assumption 6.4(C2).
  lambda_n <- sqrt(log(1 / .05) / (2 * n_train))
  calibration_target <- content + lambda_n
  h_probe <- n_train^(-.25)
  s_probe <- h_probe / 4
  cqr_cdf_zero <- mean(marginal_score[["CQR-TI"]] <= 0)
  cqr_cdf_minus_h <- mean(marginal_score[["CQR-TI"]] <= -h_probe)
  cqr_cdf_plus_h <- mean(marginal_score[["CQR-TI"]] <= h_probe)
  cqr_diagnostic[, `:=`(
    cqr_calibration_target = calibration_target,
    cqr_score_cdf_at_zero = cqr_cdf_zero,
    cqr_score_cdf_gap_at_zero = cqr_cdf_zero - calibration_target,
    c2_h_probe = h_probe,
    c2_s_probe = s_probe,
    c2_lower_margin = calibration_target - s_probe - cqr_cdf_minus_h,
    c2_upper_margin = cqr_cdf_plus_h - calibration_target - s_probe,
    c2_bracket_holds = (cqr_cdf_minus_h <= calibration_target - s_probe) &&
      (cqr_cdf_plus_h >= calibration_target + s_probe)
  )]
  x_grid <- eval_grid(n_x_grid, design)
  pivotality <- rbindlist(lapply(x_grid, function(x0) {
    x <- rep(x0, n_conditional)
    scores <- marginal_scores(x, generate_y_given_x(model_id, x))
    data.table(
      model = model_id, design = design, n_train = n_train, repetition = rep_id,
      x = x0, Method = names(scores),
      ks_to_marginal = vapply(names(scores), function(method) {
        ks_distance(scores[[method]], marginal_score[[method]])
      }, numeric(1))
    )
  }))
  oracle_marginal <- (marginal$y - base_mean(marginal$x)) / sqrt(true_sigma2(model_id, marginal$x))
  oracle_stability <- rbindlist(lapply(x_grid, function(x0) {
    x <- rep(x0, n_conditional)
    y <- generate_y_given_x(model_id, x)
    data.table(
      model = model_id, design = design, n_train = n_train, repetition = rep_id, x = x0,
      oracle_residual_ks = ks_distance(
        (y - base_mean(x)) / sqrt(true_sigma2(model_id, x)), oracle_marginal
      )
    )
  }))
  list(estimation = estimation, cqr = cqr_diagnostic,
       pivotality = pivotality, oracle = oracle_stability)
}

run_setting <- function(model_id, design, n_train) {
  message(sprintf("model=%d design=%s n_train=%d reps=%d", model_id, design, n_train, M))
  worker <- function(rep_id) run_one(model_id, design, n_train, rep_id)
  out <- if (.Platform$OS.type == "unix" && n_cores > 1L) {
    parallel::mclapply(seq_len(M), worker, mc.cores = n_cores)
  } else lapply(seq_len(M), worker)
  out
}

all_runs <- list()
for (design in designs) for (model_id in models) for (n_train in n_train_vec) {
  all_runs[[length(all_runs) + 1L]] <- run_setting(model_id, design, n_train)
}
raw <- rbindlist(lapply(all_runs, function(reps) rbindlist(lapply(reps, `[[`, "pivotality"))))
estimation_raw <- rbindlist(Filter(Negate(is.null), unlist(lapply(
  all_runs, function(reps) lapply(reps, `[[`, "estimation")
), recursive = FALSE)), fill = TRUE)
cqr_raw <- rbindlist(lapply(all_runs, function(reps) rbindlist(lapply(reps, `[[`, "cqr"))))
oracle_raw <- rbindlist(lapply(all_runs, function(reps) rbindlist(lapply(reps, `[[`, "oracle"))))

summary <- raw[, .(
  ks_mean = mean(ks_to_marginal),
  ks_median = median(ks_to_marginal),
  ks_q90 = unname(quantile(ks_to_marginal, .90)),
  ks_q95 = unname(quantile(ks_to_marginal, .95)),
  ks_max = max(ks_to_marginal)
), by = .(model, design, n_train, Method)]
good <- rbindlist(lapply(ks_eps_grid, function(eps) {
  raw[, .(px_good_pivotality = mean(ks_to_marginal <= eps)),
      by = .(model, design, n_train, Method)][, ks_eps := eps]
}))
estimation_summary <- estimation_raw[, .(
  mean_l2_error_mean = mean(mean_l2_error),
  mean_l2_error_sd = sd(mean_l2_error),
  var_l1_error_mean = mean(var_l1_error),
  var_l1_error_sd = sd(var_l1_error),
  var_sup_error_mean = mean(var_sup_error),
  var_sup_error_q90 = unname(quantile(var_sup_error, .90)),
  a_minus_abs_error_mean = mean(a_minus_abs_error),
  a_minus_abs_error_sd = sd(a_minus_abs_error),
  a_plus_abs_error_mean = mean(a_plus_abs_error),
  a_plus_abs_error_sd = sd(a_plus_abs_error)
), by = .(model, design, n_train)]
cqr_summary <- cqr_raw[, .(
  qlo_l1_error_mean = mean(qlo_l1_error),
  qhi_l1_error_mean = mean(qhi_l1_error),
  endpoint_l1_error_mean = mean(endpoint_l1_error),
  endpoint_l1_error_q90 = unname(quantile(endpoint_l1_error, .90)),
  endpoint_l1_error_q90_x_mean = mean(endpoint_l1_error_q90_x),
  endpoint_l1_error_max_x_mean = mean(endpoint_l1_error_max_x),
  cqr_score_cdf_at_zero_mean = mean(cqr_score_cdf_at_zero),
  cqr_score_cdf_gap_at_zero_abs_mean = mean(abs(cqr_score_cdf_gap_at_zero)),
  cqr_score_cdf_gap_at_zero_abs_q90 = unname(quantile(abs(cqr_score_cdf_gap_at_zero), .90)),
  c2_bracket_success = mean(c2_bracket_holds),
  c2_lower_margin_mean = mean(c2_lower_margin),
  c2_upper_margin_mean = mean(c2_upper_margin),
  cqr_cutoff_abs_mean = mean(abs(cqr_cutoff)),
  cqr_cutoff_abs_q90 = unname(quantile(abs(cqr_cutoff), .90)),
  qlo_selected_df_median = median(qlo_selected_df),
  qhi_selected_df_median = median(qhi_selected_df)
), by = .(model, design, n_train)]
oracle_summary <- oracle_raw[, .(
  oracle_residual_ks_mean = mean(oracle_residual_ks),
  oracle_residual_ks_q90 = unname(quantile(oracle_residual_ks, .90))
), by = .(model, design, n_train)]
assumption_status <- data.table(
  model = 1:4,
  location_scale_A1 = c("satisfied", "satisfied", "satisfied", "violated by design"),
  iid_fixed_split_A2 = "satisfied by independent fixed train/calibration draws",
  residual_cdf_continuity_A5 = c("satisfied", "satisfied", "satisfied", "not applicable: no common residual law"),
  oracle_density_near_target_6_2 = "satisfied",
  ASR_tail_target_A6_A7 = c("defined", "defined", "defined", "not defined: residual shape varies with x"),
  CQR_endpoint_target_C1 = "defined",
  CQR_calibration_C2 = "empirically diagnosed via fixed shrinking bracket probes",
  conditional_density_C3 = "satisfied",
  density_upper_bound = c(dnorm(0), sqrt(5 / 3) * dt(0, df = 5), dnorm(0), dnorm(0) / .7)
)

out_dir <- "results/sim/balanced4"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(raw, file.path(out_dir, paste0("score_pivotality_ks_raw_", tag, ".csv")))
fwrite(summary, file.path(out_dir, paste0("score_pivotality_ks_summary_", tag, ".csv")))
fwrite(good, file.path(out_dir, paste0("score_pivotality_good_summary_", tag, ".csv")))
fwrite(estimation_raw, file.path(out_dir, paste0("assumption_estimation_raw_", tag, ".csv")))
fwrite(estimation_summary, file.path(out_dir, paste0("assumption_estimation_summary_", tag, ".csv")))
fwrite(cqr_raw, file.path(out_dir, paste0("cqr_assumption_raw_", tag, ".csv")))
fwrite(cqr_summary, file.path(out_dir, paste0("cqr_assumption_summary_", tag, ".csv")))
fwrite(oracle_raw, file.path(out_dir, paste0("oracle_residual_stability_raw_", tag, ".csv")))
fwrite(oracle_summary, file.path(out_dir, paste0("oracle_residual_stability_summary_", tag, ".csv")))
fwrite(assumption_status, file.path(out_dir, paste0("assumption_status_", tag, ".csv")))
message("Saved score-pivotality diagnostics to: ", out_dir)
