source("R/packages.R")
source("R/sim/base_mean.R")
source("R/sim/data_generate.R")
source("R/sim/truth_content.R")
source("R/sim/fit_srti.R")
source("R/sim/fit_cqr.R")
source("R/sim/lambda_hoeffding.R")
source("R/sim/pti_utils.R")
source("R/sim/guo_young_ti.R")
source("R/sim/one_replication.R")

library(dplyr)
library(ggplot2)

set.seed(1)

generate_y_given_x <- function(model_id, x) {

  base <- base_mean(x)

  if (model_id == 1) {
    y <- base + rnorm(length(x), 0, 1)
  }

  if (model_id == 2) {
    y <- base + rt(length(x), df = 3)
  }

  if (model_id == 3) {
    y <- base + rnorm(length(x), 0, 1) * (1 + abs(x))
  }

  if (model_id == 4) {
    y <- base + rexp(length(x), rate = 1) - 1
  }

  if (model_id == 5) {
    sigma <- sqrt(0.75 + 0.5 * sin(2 * pi * x)^2)
    y <- base + rnorm(length(x), 0, sigma)
  }

  y
}

generate_eval_x <- function(n, design = "uniform") {

  p <- (seq_len(n) - 0.5) / n

  if (design == "uniform") {
    return(qunif(p, min = -2, max = 2))
  }

  if (design == "normal") {
    return(qnorm(p, mean = 0, sd = 1))
  }

  stop("Unknown design.")
}

models <- 1:5
n_train <- 500
n_cal <- 500
n_test <- 500
content <- 0.90
alpha <- 0.05
design_vec <- c("uniform", "normal")

out_dir <- "results/sim/models/plots/one_replication"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (design in design_vec) {
  design_out_dir <- file.path(out_dir, design)
  dir.create(design_out_dir, recursive = TRUE, showWarnings = FALSE)
  for (model_id in models) {

    cat("[START] design", design, "Model", model_id, "\n")

    train <- generate_data(model_id, n_train, design = design)
    cal   <- generate_data(model_id, n_cal, design = design)

    x_test <- generate_eval_x(n_test, design = design)
    y_test <- generate_y_given_x(model_id, x_test)

    x_train <- train$x
    y_train <- train$y

    x_cal <- cal$x
    y_cal <- cal$y

    true_mean <- base_mean(x_test)

    ############################################################
    # SR-TI
    ############################################################

    fit_mean_srti <- fit_mean_model_auto(x_train, y_train, model_id)
    fit_var_srti  <- fit_var_model_auto(x_train, y_train, fit_mean_srti, model_id)

    pred_cal_srti <- predict_mean_auto(fit_mean_srti, x_cal, model_id)
    var_cal_srti  <- predict_var_auto(fit_var_srti, x_cal, model_id)

    lambda_srti <- find_lambda_hat(
      mis = 1 - content,
      alpha = alpha,
      y = y_cal,
      pred = pred_cal_srti,
      variance = var_cal_srti
    )

    pred_test_srti <- predict_mean_auto(fit_mean_srti, x_test, model_id)
    var_test_srti  <- predict_var_auto(fit_var_srti, x_test, model_id)

    lower_srti <- pred_test_srti - lambda_srti * sqrt(pmax(var_test_srti, 1e-8))
    upper_srti <- pred_test_srti + lambda_srti * sqrt(pmax(var_test_srti, 1e-8))

    df_srti <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = lower_srti,
      upper = upper_srti,
      center = pred_test_srti,
      Method = "SR-TI"
    )

    ############################################################
    # ASR-TI
    ############################################################

    tau_asym <- (1 - content) / 2

    mu_train_asym  <- predict_mean_auto(fit_mean_srti, x_train, model_id)
    var_train_asym <- predict_var_auto(fit_var_srti, x_train, model_id)

    z_train_asym <- (y_train - mu_train_asym) / sqrt(pmax(var_train_asym, 1e-8))

    shape_asym <- find_asym_shape(
      z = z_train_asym,
      tau = tau_asym,
      eps = 1e-6
    )

    a_minus <- shape_asym["a_minus"]
    a_plus  <- shape_asym["a_plus"]

    z_cal_asym <- (y_cal - pred_cal_srti) / sqrt(pmax(var_cal_srti, 1e-8))

    score_asym <- asym_residual_score(
      z = z_cal_asym,
      a_minus = a_minus,
      a_plus = a_plus
    )

    q_asym <- find_score_cutoff(
      mis = 1 - content,
      alpha = alpha,
      score = score_asym
    )

    lower_asym <- pred_test_srti - q_asym * a_minus * sqrt(pmax(var_test_srti, 1e-8))
    upper_asym <- pred_test_srti + q_asym * a_plus  * sqrt(pmax(var_test_srti, 1e-8))

    df_asrti <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = lower_asym,
      upper = upper_asym,
      center = (lower_asym + upper_asym) / 2,
      Method = "ASR-TI"
    )

    ############################################################
    # CQR-TI
    ############################################################

    tau_lo <- (1 - content) / 2
    tau_hi <- 1 - tau_lo

    fit_qlo <- fit_quantile_model_auto(x_train, y_train, tau_lo, model_id)
    fit_qhi <- fit_quantile_model_auto(x_train, y_train, tau_hi, model_id)

    qlo_cal <- predict_quantile_auto(fit_qlo, x_cal, model_id)
    qhi_cal <- predict_quantile_auto(fit_qhi, x_cal, model_id)

    score_cqr <- pmax(qlo_cal - y_cal, y_cal - qhi_cal)

    lambda_cqr <- find_score_cutoff(
      mis = 1 - content,
      alpha = alpha,
      score = score_cqr
    )

    qlo_test <- predict_quantile_auto(fit_qlo, x_test, model_id)
    qhi_test <- predict_quantile_auto(fit_qhi, x_test, model_id)

    lower_cqr <- qlo_test - lambda_cqr
    upper_cqr <- qhi_test + lambda_cqr

    df_cqr <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = lower_cqr,
      upper = upper_cqr,
      center = (qlo_test + qhi_test) / 2,
      Method = "CQR-TI"
    )

    ############################################################
    # Parametric-TI
    ############################################################

    full_data <- bind_rows(train, cal)

    x_full <- full_data$x
    y_full <- full_data$y

    fit_pti <- classical_parametric_ti(
      x = x_full,
      y = y_full,
      x_new = x_test,
      content = content,
      alpha = alpha
    )

    df_pti <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = fit_pti$interval[, "lower"],
      upper = fit_pti$interval[, "upper"],
      center = fit_pti$fitted,
      Method = "Parametric-TI"
    )

    ############################################################
    # Guo-Young (2024) GY-TI
    ############################################################

    fit_gy <- gy_pointwise_ti(
      x = x_full,
      y = y_full,
      x_new = x_test,
      content = content,
      gamma = 1 - alpha,
      k_method = "appendix"
    )

    df_gy <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = fit_gy$interval[, "lower"],
      upper = fit_gy$interval[, "upper"],
      center = fit_gy$fitted,
      Method = "GY-TI"
    )

    ############################################################
    # Combine and plot
    ############################################################

    plot_df <- bind_rows(df_srti, df_asrti, df_cqr, df_pti, df_gy) %>%
    mutate(
      Method = factor(
        Method,
        levels = c(
          "SR-TI", "ASR-TI", "CQR-TI",
          "Parametric-TI", "GY-TI"
        )
      )
    )

    p <- ggplot(plot_df, aes(x = x)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.25) +
      geom_line(aes(y = center), linewidth = 0.8) +
      geom_line(aes(y = true_mean), linetype = "dashed", linewidth = 0.7) +
      geom_point(aes(y = y), size = 0.6, alpha = 0.35) +
      facet_wrap(~ Method, ncol = 1, scales = "free_y") +
      labs(
        title = paste0("One-replication intervals, Model ", model_id),
        subtitle = paste0(
          "Design = ", design,
          "n_train = ", n_train,
          ", n_cal = ", n_cal,
          ", content = ", content,
          ", alpha = ", alpha
        ),
        x = "x",
        y = "y"
      ) +
      theme_bw(base_size = 12)

    plot_file <- file.path(
      design_out_dir,
      paste0("one_replication_intervals_model", model_id, "_", design, ".png")
    )

    ggsave(
      filename = plot_file,
      plot = p,
      width = 8,
      height = 12,
      dpi = 300
    )

    cat("Saved plot to:", normalizePath(plot_file), "\n")
  }
}

cat("Done.\n")
