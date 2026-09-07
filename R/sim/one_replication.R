# R/sim/one_replication.R
# --------------------------------------------------
# One replication for:
#   - SR-TI
#   - ASR-TI
#   - CQR-TI
#   - Parametric-TI
#   - GY-TI
#
# Output:
#   list(x, content, width, lambda_na)
# --------------------------------------------------


# --------------------------------------------------
# Parametric-TI
#
# Classical homoscedastic normal-regression tolerance interval:
#   E(Y | X=x) = beta_0 + beta_1 sin(2*pi*x),
#   epsilon ~ N(0, sigma^2).
#
# It uses residual df n-p, regression leverage h(x), and the closed-form
# normal-theory tolerance factor in pti_utils.R. It is not a smoothing
# spline, a heteroscedastic model, or the Guo-Young Equation (14) method.
#
# This implementation is for one-dimensional x only.
# --------------------------------------------------

one_replication_pti <- function(model_id,
                                n_train,
                                n_cal,
                                n_test = 1000,
                                content,
                                alpha,
                                seed = NULL,
                                design = "uniform",
                                heavy_tail_scale = c("original", "unit_variance")) {

  if (!is.null(seed)) set.seed(seed)
  heavy_tail_scale <- match.arg(heavy_tail_scale)

  stopifnot(
    model_id %in% 1:4,
    n_train >= 2,
    n_cal >= 2,
    n_test >= 1,
    content > 0, content < 1,
    alpha > 0, alpha < 1
  )

  n_fit <- n_train + n_cal

  data_fit  <- generate_data(
    model_id, n_fit, design = design, heavy_tail_scale = heavy_tail_scale
  )
  data_test <- generate_eval_data(model_id, n_test, design = design)

  x <- data_fit$x
  y <- data_fit$y

  x_test <- data_test$x

  classical_fit <- classical_parametric_ti(
    x = x,
    y = y,
    x_new = x_test,
    content = content,
    alpha = alpha
  )

  lower <- classical_fit$interval[, "lower"]
  upper <- classical_fit$interval[, "upper"]

  content_vec <- content_function(
    model_id, lower, upper, x_test, heavy_tail_scale = heavy_tail_scale
  )
  width_vec   <- upper - lower

  list(
    x = x_test,
    content = content_vec,
    width = width_vec,
    lambda_na = rep(0L, n_test)
  )
}


# --------------------------------------------------
# Guo-Young (2024) pointwise TI
#
# Homoscedastic nonparametric regression TI:
#   - Equation (11) for sigma_hat and nu
#   - Appendix Lemma A.1(3) for the fast approximate k(||ell_x||)
#
# No variance standardization or PAC calibration is applied.
# --------------------------------------------------

one_replication_gy <- function(model_id,
                               n_train,
                               n_cal,
                               n_test = 1000,
                               content,
                               alpha,
                               seed = NULL,
                               design = "uniform",
                               heavy_tail_scale = c("original", "unit_variance")) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  heavy_tail_scale <- match.arg(heavy_tail_scale)

  stopifnot(
    model_id %in% 1:4,
    n_train >= 2,
    n_cal >= 2,
    n_test >= 1,
    content > 0, content < 1,
    alpha > 0, alpha < 1
  )

  n_fit <- n_train + n_cal
  data_fit <- generate_data(
    model_id, n_fit, design = design, heavy_tail_scale = heavy_tail_scale
  )
  data_test <- generate_eval_data(model_id, n_test, design = design)

  x <- data_fit$x
  y <- data_fit$y
  x_test <- data_test$x

  gy_fit <- gy_pointwise_ti(
    x = x,
    y = y,
    x_new = x_test,
    content = content,
    gamma = 1 - alpha,
    k_method = "appendix"
  )

  lower <- gy_fit$interval[, "lower"]
  upper <- gy_fit$interval[, "upper"]

  list(
    x = x_test,
    content = content_function(
      model_id, lower, upper, x_test, heavy_tail_scale = heavy_tail_scale
    ),
    width = upper - lower,
    lambda_na = rep(0L, n_test)
  )
}


# --------------------------------------------------
# Our methods:
#   SR-TI, ASR-TI, CQR-TI
# --------------------------------------------------
one_replication_ours <- function(method,
                                 model_id,
                                 n_train,
                                 n_cal,
                                 n_test = 1000,
                                 content,
                                 alpha,
                                 seed = NULL,
                                 design = "uniform",
                                 cqr_basis_df = 8,
                                 cqr_basis_type = "cv_fixed_ns",
                                 cqr_df_grid = c(4, 6, 8, 10, 12),
                                 cqr_cv_folds = 5,
                                 heavy_tail_scale = c("original", "unit_variance"),
                                 calibration_rule = c(
                                   "hoeffding",
                                   "exact_binomial"
                                 )) {

  method <- match.arg(
    method,
    c("SR-TI", "ASR-TI", "CQR-TI", "NCQR-TI")
  )
  calibration_rule <- match.arg(calibration_rule)
  heavy_tail_scale <- match.arg(heavy_tail_scale)

  if (!is.null(seed)) set.seed(seed)

  stopifnot(
    model_id %in% 1:4,
    n_train >= 2,
    n_cal >= 2,
    n_test >= 1,
    content > 0, content < 1,
    alpha > 0, alpha < 1
  )

  mis <- 1 - content

  # --------------------------------------------------
  # Generate train / calibration / test data separately
  # --------------------------------------------------

  data_train <- generate_data(
    model_id, n_train, design = design, heavy_tail_scale = heavy_tail_scale
  )
  data_cal <- generate_data(
    model_id, n_cal, design = design, heavy_tail_scale = heavy_tail_scale
  )
  data_test <- generate_eval_data(model_id, n_test, design = design)

  extract_xy <- function(data, model_id) {
  list(
    x = data$x,
    y = data$y
  )
  }

  train <- extract_xy(data_train, model_id)
  cal   <- extract_xy(data_cal, model_id)
  test  <- extract_xy(data_test, model_id)

  train_x <- train$x
  train_y <- train$y

  cal_x <- cal$x
  cal_y <- cal$y

  test_x <- test$x
  test_y <- test$y   # not directly needed for true content, but kept for clarity

  content_vec   <- rep(NA_real_, n_test)
  width_vec     <- rep(NA_real_, n_test)
  lambda_na_vec <- rep(0L, n_test)

  # --------------------------------------------------
  # SR-TI
  # --------------------------------------------------

  if (method == "SR-TI") {

    # fit nuisance models on training data
    fit_mean <- fit_mean_model_auto(train_x, train_y, model_id)
    fit_var  <- fit_var_model_auto(train_x, train_y, fit_mean, model_id)

    # calibration scores
    mu_cal  <- predict_mean_auto(fit_mean, cal_x, model_id)
    var_cal <- predict_var_auto(fit_var, cal_x, model_id)

    lambda_hat <- find_lambda_hat(
      mis      = mis,
      alpha    = alpha,
      y        = cal_y,
      pred     = mu_cal,
      variance = var_cal,
      calibration_rule = calibration_rule
    )

    if (is.na(lambda_hat)) {
      lambda_na_vec[] <- 1L

      return(list(
        x = test_x,
        content = content_vec,
        width = width_vec,
        lambda_na = lambda_na_vec
      ))
    }

    # evaluate intervals on test points
    mu_test  <- predict_mean_auto(fit_mean, test_x, model_id)
    var_test <- predict_var_auto(fit_var, test_x, model_id)

    lower <- mu_test - lambda_hat * sqrt(pmax(var_test, 1e-8))
    upper <- mu_test + lambda_hat * sqrt(pmax(var_test, 1e-8))

    content_vec <- content_function(
      model_id, lower, upper, test_x, heavy_tail_scale = heavy_tail_scale
    )
    width_vec   <- upper - lower

    return(list(
      x = test_x,
      content = content_vec,
      width = width_vec,
      lambda_na = lambda_na_vec
    ))
  }

  # --------------------------------------------------
  # ASR-TI
  # --------------------------------------------------

  if (method == "ASR-TI") {

    tau_asym <- mis / 2

    # fit nuisance models on training data
    fit_mean <- fit_mean_model_auto(train_x, train_y, model_id)
    fit_var  <- fit_var_model_auto(train_x, train_y, fit_mean, model_id)

    # training standardized residuals for shape estimation
    mu_train  <- predict_mean_auto(fit_mean, train_x, model_id)
    var_train <- predict_var_auto(fit_var, train_x, model_id)

    z_train <- (train_y - mu_train) / sqrt(pmax(var_train, 1e-8))

    shape_hat <- find_asym_shape(
      z = z_train,
      tau = tau_asym,
      eps = 1e-6
    )

    a_minus <- shape_hat["a_minus"]
    a_plus  <- shape_hat["a_plus"]

    # calibration standardized residuals
    mu_cal  <- predict_mean_auto(fit_mean, cal_x, model_id)
    var_cal <- predict_var_auto(fit_var, cal_x, model_id)

    z_cal <- (cal_y - mu_cal) / sqrt(pmax(var_cal, 1e-8))

    score_cal <- asym_residual_score(
      z = z_cal,
      a_minus = a_minus,
      a_plus = a_plus
    )

    q_hat <- find_score_cutoff(
      mis = mis,
      alpha = alpha,
      score = score_cal,
      calibration_rule = calibration_rule
    )

    if (is.na(q_hat)) {
      lambda_na_vec[] <- 1L

      return(list(
        x = test_x,
        content = content_vec,
        width = width_vec,
        lambda_na = lambda_na_vec
      ))
    }

    # evaluate intervals on test points
    mu_test  <- predict_mean_auto(fit_mean, test_x, model_id)
    var_test <- predict_var_auto(fit_var, test_x, model_id)
    sd_test  <- sqrt(pmax(var_test, 1e-8))

    lower <- mu_test - q_hat * a_minus * sd_test
    upper <- mu_test + q_hat * a_plus  * sd_test

    content_vec <- content_function(
      model_id, lower, upper, test_x, heavy_tail_scale = heavy_tail_scale
    )
    width_vec   <- upper - lower

    return(list(
      x = test_x,
      content = content_vec,
      width = width_vec,
      lambda_na = lambda_na_vec
    ))
  }

  # --------------------------------------------------
  # CQR-TI
  # --------------------------------------------------

  if (method == "CQR-TI") {

    tau_lo <- mis / 2
    tau_hi <- 1 - mis / 2
    cqr_fold_id <- sample(
      rep(seq_len(cqr_cv_folds), length.out = length(train_x))
    )

    # fit quantile models on training data
    fit_qlo <- fit_quantile_model_auto(
      train_x, train_y, tau_lo, model_id, basis_df = cqr_basis_df,
      design = design, basis_type = cqr_basis_type,
      candidate_dfs = cqr_df_grid, fold_id = cqr_fold_id
    )
    fit_qhi <- fit_quantile_model_auto(
      train_x, train_y, tau_hi, model_id, basis_df = cqr_basis_df,
      design = design, basis_type = cqr_basis_type,
      candidate_dfs = cqr_df_grid, fold_id = cqr_fold_id
    )

    # calibration scores
    qlo_cal <- predict_quantile_auto(fit_qlo, cal_x, model_id)
    qhi_cal <- predict_quantile_auto(fit_qhi, cal_x, model_id)

    score_cal <- pmax(
      qlo_cal - cal_y,
      cal_y - qhi_cal
    )

    lambda_hat <- find_score_cutoff(
      mis = mis,
      alpha = alpha,
      score = score_cal,
      calibration_rule = calibration_rule
    )

    if (is.na(lambda_hat)) {
      lambda_na_vec[] <- 1L

      return(list(
        x = test_x,
        content = content_vec,
        width = width_vec,
        lambda_na = lambda_na_vec
      ))
    }

    # evaluate intervals on test points
    qlo_test <- predict_quantile_auto(fit_qlo, test_x, model_id)
    qhi_test <- predict_quantile_auto(fit_qhi, test_x, model_id)

    lower <- qlo_test - lambda_hat
    upper <- qhi_test + lambda_hat

    content_vec <- content_function(
      model_id, lower, upper, test_x, heavy_tail_scale = heavy_tail_scale
    )
    width_vec   <- upper - lower

    return(list(
      x = test_x,
      content = content_vec,
      width = width_vec,
      lambda_na = lambda_na_vec
    ))
  }


  # --------------------------------------------------
  # NCQR-TI
  # Tail-normalized CQR score
  # --------------------------------------------------

  if (method == "NCQR-TI") {

    tau_lo_inner <- mis / 2
    tau_hi_inner <- 1 - mis / 2

    # More stable outer quantiles than 0.01 and 0.99
    tau_lo_outer <- tau_lo_inner / 2
    tau_hi_outer <- 1 - tau_lo_inner / 2

    eps_scale <- 1e-6

    # fit quantile models on training data
    fit_qlo_outer <- fit_quantile_model_auto(train_x, train_y, tau_lo_outer, model_id, basis_df = cqr_basis_df, design = design, basis_type = cqr_basis_type)
    fit_qlo_inner <- fit_quantile_model_auto(train_x, train_y, tau_lo_inner, model_id, basis_df = cqr_basis_df, design = design, basis_type = cqr_basis_type)
    fit_qhi_inner <- fit_quantile_model_auto(train_x, train_y, tau_hi_inner, model_id, basis_df = cqr_basis_df, design = design, basis_type = cqr_basis_type)
    fit_qhi_outer <- fit_quantile_model_auto(train_x, train_y, tau_hi_outer, model_id, basis_df = cqr_basis_df, design = design, basis_type = cqr_basis_type)

    # calibration predictions
    qlo_outer_cal <- predict_quantile_auto(fit_qlo_outer, cal_x, model_id)
    qlo_inner_cal <- predict_quantile_auto(fit_qlo_inner, cal_x, model_id)
    qhi_inner_cal <- predict_quantile_auto(fit_qhi_inner, cal_x, model_id)
    qhi_outer_cal <- predict_quantile_auto(fit_qhi_outer, cal_x, model_id)

    s_minus_cal <- pmax(qlo_inner_cal - qlo_outer_cal, eps_scale)
    s_plus_cal  <- pmax(qhi_outer_cal - qhi_inner_cal, eps_scale)

    score_cal <- pmax(
      (qlo_inner_cal - cal_y) / s_minus_cal,
      (cal_y - qhi_inner_cal) / s_plus_cal
    )

    q_hat <- find_score_cutoff(
      mis = mis,
      alpha = alpha,
      score = score_cal,
      calibration_rule = calibration_rule
    )

    if (is.na(q_hat)) {
      lambda_na_vec[] <- 1L

      return(list(
        x = test_x,
        content = content_vec,
        width = width_vec,
        lambda_na = lambda_na_vec
      ))
    }

    # test predictions
    qlo_outer_test <- predict_quantile_auto(fit_qlo_outer, test_x, model_id)
    qlo_inner_test <- predict_quantile_auto(fit_qlo_inner, test_x, model_id)
    qhi_inner_test <- predict_quantile_auto(fit_qhi_inner, test_x, model_id)
    qhi_outer_test <- predict_quantile_auto(fit_qhi_outer, test_x, model_id)

    s_minus_test <- pmax(qlo_inner_test - qlo_outer_test, eps_scale)
    s_plus_test  <- pmax(qhi_outer_test - qhi_inner_test, eps_scale)

    lower <- qlo_inner_test - q_hat * s_minus_test
    upper <- qhi_inner_test + q_hat * s_plus_test

    content_vec <- content_function(
      model_id, lower, upper, test_x, heavy_tail_scale = heavy_tail_scale
    )
    width_vec   <- upper - lower

    return(list(
      x = test_x,
      content = content_vec,
      width = width_vec,
      lambda_na = lambda_na_vec
    ))
  }

  stop("Method branch did not return output: ", method, call. = FALSE)
  }
