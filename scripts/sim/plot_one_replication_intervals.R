source("R/packages.R")
source("R/sim/base_mean.R")
source("R/sim/data_generate.R")
source("R/sim/truth_content.R")
source("R/sim/fit_hcti.R")
source("R/sim/fit_cqr.R")
source("R/sim/one_replication.R")
source("R/sim/lambda_hoeffding.R")
source("R/sim/pti_utils.R")

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
    # HCTI
    ############################################################

    fit_mean_hcti <- fit_mean_model_auto(x_train, y_train, model_id)
    fit_var_hcti  <- fit_var_model_auto(x_train, y_train, fit_mean_hcti, model_id)

    pred_cal_hcti <- predict_mean_auto(fit_mean_hcti, x_cal, model_id)
    var_cal_hcti  <- predict_var_auto(fit_var_hcti, x_cal, model_id)

    lambda_hcti <- find_lambda_hat(
      mis = 1 - content,
      alpha = alpha,
      y = y_cal,
      pred = pred_cal_hcti,
      variance = var_cal_hcti
    )

    pred_test_hcti <- predict_mean_auto(fit_mean_hcti, x_test, model_id)
    var_test_hcti  <- predict_var_auto(fit_var_hcti, x_test, model_id)

    lower_hcti <- pred_test_hcti - lambda_hcti * sqrt(pmax(var_test_hcti, 1e-8))
    upper_hcti <- pred_test_hcti + lambda_hcti * sqrt(pmax(var_test_hcti, 1e-8))

    df_hcti <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = lower_hcti,
      upper = upper_hcti,
      center = pred_test_hcti,
      Method = "HCTI"
    )

    ############################################################
    # HCTI-asym
    ############################################################

    tau_asym <- (1 - content) / 2

    mu_train_asym  <- predict_mean_auto(fit_mean_hcti, x_train, model_id)
    var_train_asym <- predict_var_auto(fit_var_hcti, x_train, model_id)

    z_train_asym <- (y_train - mu_train_asym) / sqrt(pmax(var_train_asym, 1e-8))

    shape_asym <- find_asym_shape(
      z = z_train_asym,
      tau = tau_asym,
      eps = 1e-6
    )

    a_minus <- shape_asym["a_minus"]
    a_plus  <- shape_asym["a_plus"]

    z_cal_asym <- (y_cal - pred_cal_hcti) / sqrt(pmax(var_cal_hcti, 1e-8))

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

    lower_asym <- pred_test_hcti - q_asym * a_minus * sqrt(pmax(var_test_hcti, 1e-8))
    upper_asym <- pred_test_hcti + q_asym * a_plus  * sqrt(pmax(var_test_hcti, 1e-8))

    df_hcti_asym <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = lower_asym,
      upper = upper_asym,
      center = (lower_asym + upper_asym) / 2,
      Method = "HCTI-asym"
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

    n_fit <- length(y_full)

    fit_mean_pti <- fit_mean_model(x_full, y_full)
    fit_var_pti  <- fit_var_model(x_full, y_full, fit_mean_pti)

    var_hat_full <- predict_var(fit_var_pti, x_full)

    y_std <- y_full / sqrt(pmax(var_hat_full, 1e-8))
    fit_std <- smooth.spline(x_full, y_std, cv = FALSE)
    mu_std <- as.numeric(predict(fit_std, x_full)$y)

    B_basis <- splines::bs(x_full, df = fit_std$df)
    D <- diff(diag(ncol(B_basis)), differences = 2)

    S_inv <- MASS::ginv(
      t(B_basis) %*% B_basis + fit_std$lambda * t(D) %*% D
    )

    S <- B_basis %*% S_inv %*% t(B_basis)
    R <- diag(n_fit) - S

    resid_std <- y_std - mu_std
    A <- t(R) %*% R

    est_var <- as.numeric(
      t(resid_std) %*% resid_std / sum(diag(A))
    )

    nu <- (sum(diag(A))^2) / sum(diag(A %*% A))

    B_test <- predict(
      splines::bs(x_full, df = fit_std$df),
      newx = x_test
    )

    L_test <- B_test %*% S_inv %*% t(B_basis)

    norm_lx_test <- apply(L_test, 1, function(v) sqrt(sum(v^2)))

    k_vec <- sapply(norm_lx_test, function(nlh) {
      find_k_factor(
        nu        = nu,
        norm_lx_h = nlh,
        content   = content,
        alpha     = alpha
      )
    })

    mu_std_test <- as.numeric(predict(fit_std, x_test)$y)
    var_test_pti <- predict_var(fit_var_pti, x_test)

    upper_pti <- (mu_std_test + sqrt(est_var) * k_vec) * sqrt(pmax(var_test_pti, 1e-8))
    lower_pti <- (mu_std_test - sqrt(est_var) * k_vec) * sqrt(pmax(var_test_pti, 1e-8))

    df_pti <- tibble(
      x = x_test,
      y = y_test,
      true_mean = true_mean,
      lower = lower_pti,
      upper = upper_pti,
      center = (lower_pti + upper_pti) / 2,
      Method = "Parametric-TI"
    )

    

    ############################################################
    # Combine and plot
    ############################################################

    plot_df <- bind_rows(df_hcti, df_hcti_asym, df_cqr, df_pti) %>%
    mutate(
      Method = factor(
        Method,
        levels = c("HCTI", "HCTI-asym", "CQR-TI", "Parametric-TI")
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
      height = 10,
      dpi = 300
    )

    cat("Saved plot to:", normalizePath(plot_file), "\n")
  }
}

cat("Done.\n")