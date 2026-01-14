# R/sim/one_replication.R
# Inputs:
#   n      : sample size
#   content: content level (target coverage proportion)
#   alpha  : confidence error level (confidence = 1 - alpha)

one_replication_ours <- function(model_id, n, content, alpha, seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  mis <- 1 - content  # miscoverage level (internal use)

  data <- generate_data(model_id, n)
  x <- data$x
  y <- data$y

  n_train <- floor(n / 2)
  n_cal   <- n - n_train

  idx <- sample.int(n)
  train_index <- idx[1:n_train]
  cal_index   <- idx[(n_train + 1):n]

  train_x <- x[train_index]; train_y <- y[train_index]
  cal_x   <- x[cal_index];   cal_y   <- y[cal_index]

  # mean / var fit
  fit_mean <- fit_mean_model(train_x, train_y)
  fit_var  <- fit_var_model(train_x, train_y, fit_mean)

  mu_cal  <- predict_mean(fit_mean, cal_x)
  var_cal <- predict_var(fit_var, cal_x)

  # NOTE:
  # find_lambda_hat(alpha = mis, delta = alpha)
  # → old naming preserved internally
  lambda_hat <- find_lambda_hat(
  mis      = 1 - content,
  alpha    = alpha,
  y        = cal_y,
  pred     = mu_cal,
  variance = var_cal,
  n        = length(cal_y)
  )


  content_vec   <- rep(NA_real_, n)
  width_vec     <- rep(NA_real_, n)
  lambda_na_vec <- rep(0L, n)

  if (is.na(lambda_hat)) {
    lambda_na_vec[cal_index] <- 1L
    return(list(content = content_vec, width = width_vec, lambda_na = lambda_na_vec))
  }

  # index-safe mapping
  for (k in seq_along(cal_index)) {
    j <- cal_index[k]
    x0 <- x[j]

    ti <- compute_ti_point(x0, fit_mean, fit_var, lambda_hat)

    content_vec[j] <- content_function(model_id, ti$lower, ti$upper, x0)
    width_vec[j]   <- ti$upper - ti$lower
  }

  list(content = content_vec, width = width_vec, lambda_na = lambda_na_vec)
}
