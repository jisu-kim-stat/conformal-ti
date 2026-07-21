# scripts/check_assumptions_alt_dgp.R
# ============================================================
# Empirical diagnostics for Assumptions 6.1, 6.2, 6.3
#
# Assumption 6.1:
#   A2: mean consistency      -> E_X[(mu_hat - mu)^2]
#   A3: variance consistency  -> E_X[|sigma2_hat - sigma2|]
#   score pivotality          -> sup_t |F_{n,x}(t) - F_n(t)|
#
# Assumption 6.2:
#   tail-scale consistency    -> |a_minus_hat - a_minus|, |a_plus_hat - a_plus|
#
# Assumption 6.3:
#   CQR score pivotality      -> sup_t |F_{n,x}^{CQR}(t) - F_n^{CQR}(t)|
#
# This script is diagnostic. It does not prove the assumptions.
# ============================================================

cat("[assumption diagnostics] working directory:", getwd(), "\n")

# ---------------------------
# Source project files
# ---------------------------

source("R/packages.R")

source("R/sim/base_mean.R")

source("R/sim/data_generate_alt.R")
source("R/sim/truth_content_alt.R")

source("R/sim/fit_hcti.R")
source("R/sim/fit_cqr.R")
source("R/sim/lambda_hoeffding.R")
source("R/sim/pti_utils.R")
source("R/sim/one_replication.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tidyr)
})

# ============================================================
# Setup
# ============================================================

set.seed(20260720)

out_dir <- "results/sim/models/assumption_diagnostics_alt_dgp"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

models <- 1:6
design_vec <- c("uniform", "normal")

n_train_vec <- c(200, 500, 1000)

content <- 0.90
tau_asym <- (1 - content) / 2

# Monte Carlo settings for diagnostic approximation
M_rep <- 50
n_eval <- 401
n_marginal <- 5000
n_conditional <- 800
n_x_grid <- 61

# thresholds for reporting approximate pivotality
ks_eps_grid <- c(0.05, 0.10, 0.15)

# ============================================================
# True functions for alternative DGP
# ============================================================

true_mu <- function(model_id, x) {
  base_mean(x)
}

true_sigma2_alt <- function(model_id, x) {
  if (model_id == 1) {
    return(rep(1, length(x)))
  }

  if (model_id == 2) {
    return(rep(3, length(x)))  # Var(t_3) = 3
  }

  if (model_id == 3) {
    return((1 + abs(x))^2)
  }

  if (model_id == 4) {
    return(rep(1, length(x)))  # Var((Chi-square_2 - 2) / 2) = 1
  }

  if (model_id == 5) {
    return(rep(1, length(x)))  # mixture of mean-zero var-one components
  }

  if (model_id == 6) {
    sigma_minus <- 0.6 + 0.4 * as.numeric(x < 0) + 0.15 * abs(x)
    sigma_plus  <- 0.7 + 0.8 * as.numeric(x > 0) + 0.30 * abs(x)

    mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
    second_moment <- 0.5 * (sigma_minus^2 + sigma_plus^2)

    return(second_moment - mean_eps^2)
  }

  stop("Unknown model_id.")
}

eval_x_grid <- function(n, design) {
  p <- (seq_len(n) - 0.5) / n

  if (design == "uniform") {
    return(stats::qunif(p, min = -2, max = 2))
  }

  if (design == "normal") {
    return(stats::qnorm(p, mean = 0, sd = 1))
  }

  stop("Unknown design.")
}

# Approximate oracle standardized residual distribution under P_X.
#
# Models 1, 2, and 4:
#   The oracle residual distribution is independent of X.
#
# Model 3:
#   The raw residual is heteroscedastic, but the standardized oracle residual
#   is independent of X.
#
# Model 5:
#   The residual shape changes with X. The returned sample is from the marginal
#   standardized residual distribution under P_X, but the conditional
#   standardized residual distribution depends on x.
#
# Model 6:
#   The centered split-normal residual has x-dependent lower and upper tail
#   scales. After standardization by the conditional standard deviation, the
#   residual distribution still generally depends on x. Thus this is also a
#   non-pivotal residual-shape setting.
draw_oracle_residuals <- function(model_id, n, design) {
  stopifnot(model_id %in% 1:6)

  x <- eval_x_grid(n, design)
  idx <- sample(seq_along(x), size = n, replace = TRUE)
  x <- x[idx]

  y <- generate_y_given_x(model_id, x)
  mu <- true_mu(model_id, x)
  sig <- sqrt(true_sigma2_alt(model_id, x))

  (y - mu) / sig
}


true_asym_shape <- function(model_id, tau, design, n_mc = 200000) {
  stopifnot(model_id %in% 1:6)

  eps <- draw_oracle_residuals(
    model_id = model_id,
    n = n_mc,
    design = design
  )

  q_lo <- as.numeric(
    stats::quantile(eps, probs = tau, names = FALSE, type = 8)
  )

  q_hi <- as.numeric(
    stats::quantile(eps, probs = 1 - tau, names = FALSE, type = 8)
  )

  c(
    a_minus_true = abs(q_lo),
    a_plus_true = q_hi
  )
}

# ============================================================
# KS distance helper
# ============================================================

ks_distance <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]

  if (length(x) < 5 || length(y) < 5) {
    return(NA_real_)
  }

  z <- sort(unique(c(x, y)))
  max(abs(stats::ecdf(x)(z) - stats::ecdf(y)(z)))
}

# ============================================================
# Score functions
# ============================================================

compute_hcti_score <- function(x, y, fit_mean, fit_var, model_id) {
  mu_hat <- predict_mean_auto(fit_mean, x, model_id)
  var_hat <- predict_var_auto(fit_var, x, model_id)

  abs(y - mu_hat) / sqrt(pmax(var_hat, 1e-8))
}

compute_hcti_asym_score <- function(x, y, fit_mean, fit_var, model_id, a_minus, a_plus) {
  mu_hat <- predict_mean_auto(fit_mean, x, model_id)
  var_hat <- predict_var_auto(fit_var, x, model_id)

  z <- (y - mu_hat) / sqrt(pmax(var_hat, 1e-8))

  asym_residual_score(
    z = z,
    a_minus = a_minus,
    a_plus = a_plus
  )
}

compute_cqr_score <- function(x, y, fit_qlo, fit_qhi, model_id) {
  qlo <- predict_quantile_auto(fit_qlo, x, model_id)
  qhi <- predict_quantile_auto(fit_qhi, x, model_id)

  pmax(qlo - y, y - qhi)
}


# ============================================================
# One diagnostic replication
# ============================================================

run_one_diagnostic <- function(model_id, design, n_train, rep_id) {

  set.seed(100000 * model_id + 1000 * n_train + rep_id)

  # ---------------------------
  # Training data
  # ---------------------------

  train <- generate_data(model_id, n_train, design = design)

  x_train <- train$x
  y_train <- train$y

  # ---------------------------
  # Fit HCTI nuisance functions
  # ---------------------------

  fit_mean <- fit_mean_model_auto(x_train, y_train, model_id)
  fit_var <- fit_var_model_auto(x_train, y_train, fit_mean, model_id)

  # ---------------------------
  # A2/A3 diagnostics on PX quantile grid
  # ---------------------------

  x_eval <- eval_x_grid(n_eval, design)

  mu_hat_eval <- predict_mean_auto(fit_mean, x_eval, model_id)
  var_hat_eval <- predict_var_auto(fit_var, x_eval, model_id)

  mu_true_eval <- true_mu(model_id, x_eval)
  var_true_eval <- true_sigma2_alt(model_id, x_eval)

  mean_l2_error <- mean((mu_hat_eval - mu_true_eval)^2, na.rm = TRUE)
  var_l1_error <- mean(abs(var_hat_eval - var_true_eval), na.rm = TRUE)

  # ---------------------------
  # A6 tail-scale diagnostics
  # ---------------------------

  mu_train_hat <- predict_mean_auto(fit_mean, x_train, model_id)
  var_train_hat <- predict_var_auto(fit_var, x_train, model_id)

  z_train_hat <- (y_train - mu_train_hat) / sqrt(pmax(var_train_hat, 1e-8))

  shape_hat <- find_asym_shape(
    z = z_train_hat,
    tau = tau_asym,
    eps = 1e-6
  )

  shape_true <- true_asym_shape(
    model_id = model_id,
    tau = tau_asym,
    design = design,
    n_mc = 50000
  )

  a_minus_hat <- as.numeric(shape_hat["a_minus"])
  a_plus_hat <- as.numeric(shape_hat["a_plus"])

  a_minus_true <- as.numeric(shape_true["a_minus_true"])
  a_plus_true <- as.numeric(shape_true["a_plus_true"])

  a_minus_abs_error <- abs(a_minus_hat - a_minus_true)
  a_plus_abs_error <- abs(a_plus_hat - a_plus_true)

  # ---------------------------
  # Fit CQR nuisance functions
  # ---------------------------

  alpha_lo <- (1 - content) / 2
  alpha_hi <- 1 - alpha_lo
  
  fit_qlo <- fit_quantile_model_auto(x_train, y_train, alpha_lo, model_id)
  fit_qhi <- fit_quantile_model_auto(x_train, y_train, alpha_hi, model_id)
  
  # ---------------------------
  # Marginal score samples F_n
  # ---------------------------

  marginal_data <- generate_data(model_id, n_marginal, design = design)
  x_marg <- marginal_data$x
  y_marg <- marginal_data$y

  score_marg_hcti <- compute_hcti_score(
    x = x_marg,
    y = y_marg,
    fit_mean = fit_mean,
    fit_var = fit_var,
    model_id = model_id
  )

  score_marg_asym <- compute_hcti_asym_score(
    x = x_marg,
    y = y_marg,
    fit_mean = fit_mean,
    fit_var = fit_var,
    model_id = model_id,
    a_minus = a_minus_hat,
    a_plus = a_plus_hat
  )

  score_marg_cqr <- compute_cqr_score(
    x = x_marg,
    y = y_marg,
    fit_qlo = fit_qlo,
    fit_qhi = fit_qhi,
    model_id = model_id
  )

  # ---------------------------
  # Conditional score samples F_{n,x}
  # ---------------------------

  x_grid <- eval_x_grid(n_x_grid, design)

  pivotality_df <- dplyr::bind_rows(lapply(x_grid, function(x0) {

    x_cond <- rep(x0, n_conditional)
    y_cond <- generate_y_given_x(model_id, x_cond)

    score_cond_hcti <- compute_hcti_score(
      x = x_cond,
      y = y_cond,
      fit_mean = fit_mean,
      fit_var = fit_var,
      model_id = model_id
    )

    score_cond_asym <- compute_hcti_asym_score(
      x = x_cond,
      y = y_cond,
      fit_mean = fit_mean,
      fit_var = fit_var,
      model_id = model_id,
      a_minus = a_minus_hat,
      a_plus = a_plus_hat
    )

    score_cond_cqr <- compute_cqr_score(
      x = x_cond,
      y = y_cond,
      fit_qlo = fit_qlo,
      fit_qhi = fit_qhi,
      model_id = model_id
    )
    dplyr::tibble(
    x = x0,
    Method = c("HCTI", "HCTI-asym", "CQR-TI"),
    ks_to_marginal = c(
        ks_distance(score_cond_hcti, score_marg_hcti),
        ks_distance(score_cond_asym, score_marg_asym),
        ks_distance(score_cond_cqr, score_marg_cqr)
    )
    )
  }))

  estimation_df <- dplyr::tibble(
    model = model_id,
    design = design,
    n_train = n_train,
    rep = rep_id,
    mean_l2_error = mean_l2_error,
    var_l1_error = var_l1_error,
    a_minus_hat = a_minus_hat,
    a_plus_hat = a_plus_hat,
    a_minus_true = a_minus_true,
    a_plus_true = a_plus_true,
    a_minus_abs_error = a_minus_abs_error,
    a_plus_abs_error = a_plus_abs_error
  )

  pivotality_df <- pivotality_df %>%
    mutate(
      model = model_id,
      design = design,
      n_train = n_train,
      rep = rep_id
    ) %>%
    select(model, design, n_train, rep, Method, x, ks_to_marginal)

  list(
    estimation = estimation_df,
    pivotality = pivotality_df
  )
}

# ============================================================
# Run diagnostics
# ============================================================

all_estimation <- list()
all_pivotality <- list()

counter <- 1

for (design in design_vec) {
  for (model_id in models) {
    for (n_train in n_train_vec) {

      cat(
        "[START]",
        "design:", design,
        "model:", model_id,
        "n_train:", n_train,
        "\n"
      )

      for (b in seq_len(M_rep)) {

        if (b %% 10 == 0) {
          cat("  rep", b, "of", M_rep, "\n")
        }

        res <- tryCatch(
          run_one_diagnostic(
            model_id = model_id,
            design = design,
            n_train = n_train,
            rep_id = b
          ),
          error = function(e) {
            message(
              "[ERROR] design=", design,
              " model=", model_id,
              " n_train=", n_train,
              " rep=", b,
              ": ", conditionMessage(e)
            )
            NULL
          }
        )

        if (!is.null(res)) {
          all_estimation[[counter]] <- res$estimation
          all_pivotality[[counter]] <- res$pivotality
          counter <- counter + 1
        }
      }
    }
  }
}

estimation_df <- dplyr::bind_rows(all_estimation)
pivotality_df <- dplyr::bind_rows(all_pivotality)

# ============================================================
# Summaries
# ============================================================

estimation_summary <- estimation_df %>%
  group_by(model, design, n_train) %>%
  summarise(
    mean_l2_error_mean = mean(mean_l2_error, na.rm = TRUE),
    mean_l2_error_sd = sd(mean_l2_error, na.rm = TRUE),

    var_l1_error_mean = mean(var_l1_error, na.rm = TRUE),
    var_l1_error_sd = sd(var_l1_error, na.rm = TRUE),

    a_minus_abs_error_mean = mean(a_minus_abs_error, na.rm = TRUE),
    a_minus_abs_error_sd = sd(a_minus_abs_error, na.rm = TRUE),

    a_plus_abs_error_mean = mean(a_plus_abs_error, na.rm = TRUE),
    a_plus_abs_error_sd = sd(a_plus_abs_error, na.rm = TRUE),

    a_minus_hat_mean = mean(a_minus_hat, na.rm = TRUE),
    a_plus_hat_mean = mean(a_plus_hat, na.rm = TRUE),
    a_minus_true_mean = mean(a_minus_true, na.rm = TRUE),
    a_plus_true_mean = mean(a_plus_true, na.rm = TRUE),

    .groups = "drop"
  )

pivotality_summary <- pivotality_df %>%
  group_by(model, design, n_train, Method) %>%
  summarise(
    ks_mean = mean(ks_to_marginal, na.rm = TRUE),
    ks_median = median(ks_to_marginal, na.rm = TRUE),
    ks_q90 = as.numeric(quantile(ks_to_marginal, 0.90, na.rm = TRUE)),
    ks_q95 = as.numeric(quantile(ks_to_marginal, 0.95, na.rm = TRUE)),
    ks_max = max(ks_to_marginal, na.rm = TRUE),
    .groups = "drop"
  )

pivotality_good_summary <- dplyr::bind_rows(lapply(ks_eps_grid, function(eps_ks) {

  pivotality_df %>%
    group_by(model, design, n_train, Method) %>%
    summarise(
      ks_eps = eps_ks,
      px_good_pivotality = mean(ks_to_marginal <= eps_ks, na.rm = TRUE),
      .groups = "drop"
    )

}))

# ============================================================
# Save CSVs
# ============================================================

readr::write_csv(
  estimation_df,
  file.path(out_dir, "assumption_estimation_raw.csv")
)

readr::write_csv(
  estimation_summary,
  file.path(out_dir, "assumption_estimation_summary.csv")
)

readr::write_csv(
  pivotality_df,
  file.path(out_dir, "score_pivotality_ks_raw.csv")
)

readr::write_csv(
  pivotality_summary,
  file.path(out_dir, "score_pivotality_ks_summary.csv")
)

readr::write_csv(
  pivotality_good_summary,
  file.path(out_dir, "score_pivotality_good_summary.csv")
)

cat("[saved] estimation summary:", file.path(out_dir, "assumption_estimation_summary.csv"), "\n")
cat("[saved] pivotality summary:", file.path(out_dir, "score_pivotality_ks_summary.csv"), "\n")
cat("[saved] pivotality good summary:", file.path(out_dir, "score_pivotality_good_summary.csv"), "\n")

# ============================================================
# Plots
# ============================================================

theme_diag <- function(base_size = 12) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "grey25", hjust = 0),
      axis.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey93", color = "grey60"),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      panel.grid.major.y = element_line(color = "grey88", linewidth = 0.3),
      panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25)
    )
}

method_cols <- c(
  "HCTI" = "#D55E00",
  "HCTI-asym" = "#CC79A7",
  "CQR-TI" = "#0072B2"
)

# ------------------------------------------------------------
# A2: mean L2 error
# ------------------------------------------------------------

p_mean <- ggplot(
  estimation_summary,
  aes(x = n_train, y = mean_l2_error_mean, group = 1)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.3) +
  facet_grid(design ~ model, labeller = label_both, scales = "free_y") +
  scale_x_continuous(breaks = n_train_vec) +
  labs(
    title = "Diagnostic for Assumption 6.1(A2): mean estimation",
    subtitle = "Estimated E_X[(mu_hat(X) - mu(X))^2]",
    x = "Training sample size",
    y = "Mean L2 error"
  ) +
  theme_diag()

ggsave(
  file.path(out_dir, "diag_A2_mean_L2_error.png"),
  p_mean,
  width = 13,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# A3: variance L1 error
# ------------------------------------------------------------

p_var <- ggplot(
  estimation_summary,
  aes(x = n_train, y = var_l1_error_mean, group = 1)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.3) +
  facet_grid(design ~ model, labeller = label_both, scales = "free_y") +
  scale_x_continuous(breaks = n_train_vec) +
  labs(
    title = "Diagnostic for Assumption 6.1(A3): variance estimation",
    subtitle = "Estimated E_X[|sigma_hat^2(X) - sigma^2(X)|]",
    x = "Training sample size",
    y = "Mean L1 error"
  ) +
  theme_diag()

ggsave(
  file.path(out_dir, "diag_A3_variance_L1_error.png"),
  p_var,
  width = 13,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# A6: tail-scale errors
# ------------------------------------------------------------

tail_long <- estimation_summary %>%
  select(
    model, design, n_train,
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
    )
  )

p_tail <- ggplot(
  tail_long,
  aes(x = n_train, y = abs_error, color = tail_scale, group = tail_scale)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.3) +
  facet_grid(design ~ model, labeller = label_both, scales = "free_y") +
  scale_x_continuous(breaks = n_train_vec) +
  labs(
    title = "Diagnostic for Assumption 6.2: asymmetric tail-scale estimation",
    subtitle = "Errors of a_minus and a_plus estimated from training standardized residuals",
    x = "Training sample size",
    y = "Absolute error"
  ) +
  theme_diag()

ggsave(
  file.path(out_dir, "diag_A6_tail_scale_error.png"),
  p_tail,
  width = 13,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# Score pivotality KS summary
# ------------------------------------------------------------

p_ks <- ggplot(
  pivotality_summary,
  aes(
    x = n_train,
    y = ks_q90,
    color = Method,
    group = Method
  )
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.4) +
  facet_grid(design ~ model, labeller = label_both, scales = "free_y") +
  scale_color_manual(values = method_cols) +
  scale_x_continuous(breaks = n_train_vec) +
  labs(
    title = "Empirical score pivotality diagnostic",
    subtitle = "90th percentile of sup_t |F_{n,x}(t) - F_n(t)| across x and Monte Carlo replications",
    x = "Training sample size",
    y = "KS distance to marginal score distribution"
  ) +
  theme_diag()

ggsave(
  file.path(out_dir, "diag_score_pivotality_KS_q90.png"),
  p_ks,
  width = 13,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# Pivotality-good proportion
# ------------------------------------------------------------

p_good <- ggplot(
  pivotality_good_summary,
  aes(
    x = n_train,
    y = px_good_pivotality,
    color = Method,
    group = Method
  )
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.4) +
  facet_grid(ks_eps ~ model + design, labeller = label_both) +
  scale_color_manual(values = method_cols) +
  scale_x_continuous(breaks = n_train_vec) +
  coord_cartesian(ylim = c(0, 1.03)) +
  labs(
    title = "Approximate pivotality-good proportion",
    subtitle = "Fraction of x values with KS distance below a chosen tolerance",
    x = "Training sample size",
    y = "Pivotality-good proportion"
  ) +
  theme_diag(base_size = 9.5)

ggsave(
  file.path(out_dir, "diag_pivotality_good_proportion.png"),
  p_good,
  width = 15,
  height = 10,
  dpi = 300
)

cat("[done] assumption diagnostics saved under:", out_dir, "\n")