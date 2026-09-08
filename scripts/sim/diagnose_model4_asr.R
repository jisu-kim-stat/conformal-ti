# Ablation diagnostic for ASR--TI under Model 4.
#
# It separates three ingredients of ASR--TI:
#   1. fully fitted ASR--TI;
#   2. ASR--TI with oracle mean and scale but an estimated asymmetric shape;
#   3. ASR--TI with oracle mean, scale, and asymmetric shape.
#
# Run from the project root, for example:
# Rscript scripts/sim/diagnose_model4_asr.R --reps=200 --cores=4 --tag=pilot

source("R/packages.R")
source("R/sim/base_mean.R")
source("R/sim/data_generate_balanced4.R")
source("R/sim/truth_content_balanced4.R")
source("R/sim/fit_srti.R")
source("R/sim/lambda_hoeffding.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    pieces <- strsplit(substring(arg, 3), "=", fixed = TRUE)[[1]]
    out[[pieces[1]]] <- if (length(pieces) == 2L) pieces[2] else TRUE
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
M <- as.integer(if (is.null(args$reps)) 200 else args$reps)
n_cores <- as.integer(if (is.null(args$cores)) 1 else args$cores)
n_train <- as.integer(if (is.null(args$ntrain)) 1000 else args$ntrain)
n_cal <- as.integer(if (is.null(args$ncal)) 1000 else args$ncal)
n_test <- as.integer(if (is.null(args$ntest)) 1000 else args$ntest)
design <- if (is.null(args$design)) "normal" else args$design
tag <- if (is.null(args$tag)) "pilot" else args$tag

stopifnot(
  M >= 1L, n_cores >= 1L, n_train >= 2L, n_cal >= 2L, n_test >= 10L,
  design %in% c("normal", "uniform")
)

model_id <- 4L
content <- 0.90
alpha <- 0.05
epsilon <- 0.02
mis <- 1 - content
tau_asym <- mis / 2

# Model 4 has Z = {Gamma(4, 1) - 4}/2, with E(Z) = 0 and Var(Z) = 1.
true_a_minus <- abs((qgamma(tau_asym, shape = 4, rate = 1) - 4) / 2)
true_a_plus <- (qgamma(1 - tau_asym, shape = 4, rate = 1) - 4) / 2

make_interval <- function(cal_y, cal_mu, cal_sd, test_mu, test_sd,
                          a_minus, a_plus) {
  z_cal <- (cal_y - cal_mu) / pmax(cal_sd, 1e-8)
  q_hat <- find_score_cutoff(
    mis = mis,
    alpha = alpha,
    score = asym_residual_score(z_cal, a_minus = a_minus, a_plus = a_plus),
    calibration_rule = "hoeffding"
  )
  list(
    lower = test_mu - q_hat * a_minus * test_sd,
    upper = test_mu + q_hat * a_plus * test_sd,
    q_hat = q_hat
  )
}

one_replication <- function(rep_id) {
  set.seed(20260908L + rep_id)

  train <- generate_data(model_id, n_train, design = design)
  cal <- generate_data(model_id, n_cal, design = design)
  test <- generate_eval_data(model_id, n_test, design = design)

  true_mu_train <- base_mean(train$x)
  true_sd_train <- hetero_scale(train$x)
  true_mu_cal <- base_mean(cal$x)
  true_sd_cal <- hetero_scale(cal$x)
  true_mu_test <- base_mean(test$x)
  true_sd_test <- hetero_scale(test$x)

  fit_mu <- fit_mean_model(train$x, train$y)
  fit_var <- fit_var_model(train$x, train$y, fit_mu)
  fitted_mu_train <- predict_mean(fit_mu, train$x)
  fitted_sd_train <- sqrt(pmax(predict_var(fit_var, train$x), 1e-8))
  fitted_mu_cal <- predict_mean(fit_mu, cal$x)
  fitted_sd_cal <- sqrt(pmax(predict_var(fit_var, cal$x), 1e-8))
  fitted_mu_test <- predict_mean(fit_mu, test$x)
  fitted_sd_test <- sqrt(pmax(predict_var(fit_var, test$x), 1e-8))

  fitted_shape <- find_asym_shape(
    z = (train$y - fitted_mu_train) / fitted_sd_train,
    tau = tau_asym
  )
  oracle_nuisance_shape <- find_asym_shape(
    z = (train$y - true_mu_train) / true_sd_train,
    tau = tau_asym
  )

  definitions <- list(
    "ASR-TI (fitted)" = list(
      cal_mu = fitted_mu_cal, cal_sd = fitted_sd_cal,
      test_mu = fitted_mu_test, test_sd = fitted_sd_test,
      shape = fitted_shape
    ),
    "ASR-TI (oracle nuisance)" = list(
      cal_mu = true_mu_cal, cal_sd = true_sd_cal,
      test_mu = true_mu_test, test_sd = true_sd_test,
      shape = oracle_nuisance_shape
    ),
    "ASR-TI (oracle nuisance + shape)" = list(
      cal_mu = true_mu_cal, cal_sd = true_sd_cal,
      test_mu = true_mu_test, test_sd = true_sd_test,
      shape = c(a_minus = true_a_minus, a_plus = true_a_plus)
    )
  )

  interval_rows <- bind_rows(lapply(names(definitions), function(method) {
    d <- definitions[[method]]
    interval <- make_interval(
      cal_y = cal$y, cal_mu = d$cal_mu, cal_sd = d$cal_sd,
      test_mu = d$test_mu, test_sd = d$test_sd,
      a_minus = d$shape[["a_minus"]], a_plus = d$shape[["a_plus"]]
    )
    data.frame(
      rep = rep_id,
      method = method,
      x = test$x,
      content = content_function(model_id, interval$lower, interval$upper, test$x),
      width = interval$upper - interval$lower,
      q_hat = interval$q_hat
    )
  }))

  nuisance_rows <- data.frame(
    rep = rep_id,
    x = test$x,
    sigma_ratio = fitted_sd_test / true_sd_test,
    mean_error_standardized = (fitted_mu_test - true_mu_test) / true_sd_test
  )
  shape_rows <- data.frame(
    rep = rep_id,
    method = c("ASR-TI (fitted)", "ASR-TI (oracle nuisance)",
               "ASR-TI (oracle nuisance + shape)"),
    a_minus = c(fitted_shape[["a_minus"]], oracle_nuisance_shape[["a_minus"]], true_a_minus),
    a_plus = c(fitted_shape[["a_plus"]], oracle_nuisance_shape[["a_plus"]], true_a_plus)
  )

  list(intervals = interval_rows, nuisance = nuisance_rows, shape = shape_rows)
}

message(sprintf(
  "Running Model 4 ASR ablation: M=%d, n_train=n_cal=%d, design=%s.",
  M, n_train, design
))
replications <- if (n_cores == 1L) {
  lapply(seq_len(M), one_replication)
} else {
  parallel::mclapply(seq_len(M), one_replication, mc.cores = n_cores)
}

intervals <- bind_rows(lapply(replications, `[[`, "intervals"))
nuisance <- bind_rows(lapply(replications, `[[`, "nuisance"))
shape <- bind_rows(lapply(replications, `[[`, "shape"))

# The target is C-epsilon, not the replication-specific content value.
pointwise <- intervals %>%
  group_by(method, x) %>%
  summarise(
    mean_content = mean(content),
    pointwise_success = mean(content >= 0.90 - epsilon),
    mean_width = mean(width),
    .groups = "drop"
  )

px_good <- pointwise %>%
  group_by(method) %>%
  summarise(
    px_good_proportion = mean(pointwise_success >= 1 - alpha),
    mean_pointwise_success = mean(pointwise_success),
    .groups = "drop"
  )

nuisance_summary <- nuisance %>%
  group_by(x) %>%
  summarise(
    mean_sigma_ratio = mean(sigma_ratio),
    q05_sigma_ratio = quantile(sigma_ratio, 0.05),
    q95_sigma_ratio = quantile(sigma_ratio, 0.95),
    mean_mean_error_standardized = mean(mean_error_standardized),
    .groups = "drop"
  )

shape_summary <- shape %>%
  group_by(method) %>%
  summarise(
    mean_a_minus = mean(a_minus), sd_a_minus = sd(a_minus),
    mean_a_plus = mean(a_plus), sd_a_plus = sd(a_plus),
    .groups = "drop"
  )

out_dir <- file.path("results/sim/asr_diagnostics", paste0("model4_", tag))
fig_dir <- file.path("fig/sim/asr_diagnostics", paste0("model4_", tag))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(pointwise, file.path(out_dir, "pointwise_ablation.csv"))
write_csv(px_good, file.path(out_dir, "px_good_ablation.csv"))
write_csv(nuisance_summary, file.path(out_dir, "nuisance_summary.csv"))
write_csv(shape_summary, file.path(out_dir, "shape_summary.csv"))

p_success <- ggplot(pointwise, aes(x, pointwise_success, color = method)) +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "grey35") +
  geom_line(linewidth = 0.65) +
  coord_cartesian(ylim = c(0, 1.03)) +
  labs(x = "Covariate x", y = "Conditional-success probability", color = NULL) +
  theme_classic(base_size = 11) + theme(legend.position = "bottom")

p_nuisance <- ggplot(nuisance_summary, aes(x, mean_sigma_ratio)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey35") +
  geom_ribbon(aes(ymin = q05_sigma_ratio, ymax = q95_sigma_ratio), alpha = 0.20) +
  geom_line(linewidth = 0.65) +
  labs(x = "Covariate x", y = expression(hat(sigma)(x) / sigma(x))) +
  theme_classic(base_size = 11)

ggsave(file.path(fig_dir, "pointwise_success_ablation.pdf"), p_success,
       width = 8.5, height = 4.2)
ggsave(file.path(fig_dir, "sigma_ratio.pdf"), p_nuisance,
       width = 8.5, height = 4.2)

message("Saved Model 4 ASR diagnostic results to: ", out_dir)
message("Saved Model 4 ASR diagnostic figures to: ", fig_dir)
print(px_good)
print(shape_summary)
