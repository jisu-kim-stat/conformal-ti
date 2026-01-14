one_replication_gy <- function(model_id, n, content, alpha, seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  data <- generate_data(model_id, n)
  x <- data$x; y <- data$y

  # model fitting
  fit_mean <- fit_mean_model(x, y)
  fit_var  <- fit_var_model(x, y, fit_mean)

  mu_hat  <- predict_mean(fit_mean, x)
  var_hat <- predict_var(fit_var, x)

  # standardization
  y_std <- y / sqrt(var_hat)
  fit_std <- smooth.spline(x, y_std, cv = FALSE)
  mu_std <- predict(fit_std, x)$y

  # smoothing matrix S
  B <- splines::bs(x, df = fit_std$df)
  D <- diff(diag(ncol(B)), differences = 2)
  S_inv <- MASS::ginv(t(B) %*% B + fit_std$lambda * t(D) %*% D)
  S <- B %*% S_inv %*% t(B)
  R <- diag(n) - S

  # variance + df
  resid2 <- y_std - mu_std
  est_var <- as.numeric(t(resid2) %*% resid2 / sum(diag(t(R) %*% R)))
  nu <- (sum(diag(t(R) %*% R))^2) / sum(diag((t(R) %*% R)^2))

  # norm ||l_x h||
  norm_lx <- apply(S, 2, function(v) sqrt(sum(v^2)))

  # k-factor per x
  mis <- 1 - content

  k_vec <- sapply(norm_lx, function(nlh) {
    find_k_factor(
      nu        = nu,
      norm_lx_h = nlh,
      content   = content,
      alpha     = alpha
    )
  })



  # TI
  upper <- (mu_std + sqrt(est_var) * k_vec) * sqrt(var_hat)
  lower <- (mu_std - sqrt(est_var) * k_vec) * sqrt(var_hat)

  content <- content_function(model_id, lower, upper, x)
  width   <- upper - lower
  lambda_na <- rep(0L, n)

  list(content=content, width=width, lambda_na=lambda_na)
}
