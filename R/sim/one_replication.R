# R/sim/one_replication.R
# --------------------------------------------------
# One replication for:
#   - HCTI
#   - CQR-TI
#   - Parametric-TI
#
# Output:
#   list(content, width, lambda_na)
# --------------------------------------------------


# --------------------------------------------------
# Parametric TI
# This implementation is for one-dimensional x only.
# --------------------------------------------------

one_replication_pti <- function(model_id,
                                n_train,
                                n_cal,
                                n_test = 1000,
                                content,
                                alpha,
                                seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  stopifnot(
    model_id %in% 1:6,
    n_train >= 2,
    n_cal >= 2,
    n_test >= 1,
    content > 0, content < 1,
    alpha > 0, alpha < 1
  )

  if (model_id == 6) {
    return(list(
      content = rep(NA_real_, n_test),
      width = rep(NA_real_, n_test),
      lambda_na = rep(1L, n_test)
    ))
  }

  n_fit <- n_train + n_cal

  data_fit  <- generate_data(model_id, n_fit)
  data_test <- generate_data(model_id, n_test)

  x <- data_fit$x
  y <- data_fit$y

  x_test <- data_test$x

  # mean / variance fit using all fitting data
  fit_mean <- fit_mean_model(x, y)
  fit_var  <- fit_var_model(x, y, fit_mean)

  var_hat <- predict_var(fit_var, x)

  # standardization
  y_std <- y / sqrt(pmax(var_hat, 1e-8))
  fit_std <- smooth.spline(x, y_std, cv = FALSE)
  mu_std <- as.numeric(predict(fit_std, x)$y)

  # smoothing matrix S
  B <- splines::bs(x, df = fit_std$df)
  D <- diff(diag(ncol(B)), differences = 2)

  S_inv <- MASS::ginv(
    t(B) %*% B + fit_std$lambda * t(D) %*% D
  )

  S <- B %*% S_inv %*% t(B)
  R <- diag(n_fit) - S

  resid_std <- y_std - mu_std

  A <- t(R) %*% R

  est_var <- as.numeric(
    t(resid_std) %*% resid_std / sum(diag(A))
  )

  nu <- (sum(diag(A))^2) / sum(diag(A %*% A))

  # For test points, need smoother row l(x0), not just rows of S on training points.
  # Approximate by constructing basis at test points.
  B_test <- predict(
    splines::bs(x, df = fit_std$df),
    newx = x_test
  )

  L_test <- B_test %*% S_inv %*% t(B)

  norm_lx_test <- apply(L_test, 1, function(v) sqrt(sum(v^2)))

  k_vec <- sapply(norm_lx_test, function(nlh) {
    find_k_factor(
      nu         = nu,
      norm_lx_h  = nlh,
      content    = content,
      alpha      = alpha
    )
  })

  # Predict standardized mean and variance on test points
  mu_std_test <- as.numeric(predict(fit_std, x_test)$y)
  var_test <- predict_var(fit_var, x_test)

  upper <- (mu_std_test + sqrt(est_var) * k_vec) * sqrt(pmax(var_test, 1e-8))
  lower <- (mu_std_test - sqrt(est_var) * k_vec) * sqrt(pmax(var_test, 1e-8))

  content_vec <- content_function(model_id, lower, upper, x_test)
  width_vec   <- upper - lower

  list(
    content = content_vec,
    width = width_vec,
    lambda_na = rep(0L, n_test)
  )
}


# --------------------------------------------------
# Our methods:
#   HCTI and CQR-TI
# --------------------------------------------------
one_replication_ours <- function(method,
                                 model_id,
                                 n_train,
                                 n_cal,
                                 n_test = 1000,
                                 content,
                                 alpha,
                                 seed = NULL) {

  method <- match.arg(method, c("HCTI", "CQR-TI"))

  if (!is.null(seed)) set.seed(seed)

  stopifnot(
    model_id %in% 1:6,
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

  data_train <- generate_data(model_id, n_train)
  data_cal   <- generate_data(model_id, n_cal)
  data_test  <- generate_data(model_id, n_test)

  extract_xy <- function(data, model_id) {
    y <- data$y

    if (model_id == 6) {
      x_cols <- paste0("x", 1:20)
      x <- as.matrix(data[, x_cols, drop = FALSE])
    } else {
      x <- data$x
    }

    list(x = x, y = y)
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
  # HCTI
  # --------------------------------------------------

  if (method == "HCTI") {

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
      variance = var_cal
    )

    if (is.na(lambda_hat)) {
      lambda_na_vec[] <- 1L

      return(list(
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

    content_vec <- content_function(model_id, lower, upper, test_x)
    width_vec   <- upper - lower

    return(list(
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

    # fit quantile models on training data
    fit_qlo <- fit_quantile_model_auto(train_x, train_y, tau_lo, model_id)
    fit_qhi <- fit_quantile_model_auto(train_x, train_y, tau_hi, model_id)

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
      score = score_cal
    )

    if (is.na(lambda_hat)) {
      lambda_na_vec[] <- 1L

      return(list(
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

    content_vec <- content_function(model_id, lower, upper, test_x)
    width_vec   <- upper - lower

    return(list(
      content = content_vec,
      width = width_vec,
      lambda_na = lambda_na_vec
    ))
  }
}