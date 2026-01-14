## experiments/real/happy/R/ours_1d.R

suppressPackageStartupMessages({
  library(dplyr)
  library(gbm)
})

source("experiments/real/happy/R/transforms.R")
source("experiments/real/happy/R/hoeffding.R")

# median impute + standardize (train 기준)
prep_x_1d <- function(x_tr) {
  med <- stats::median(x_tr, na.rm = TRUE)
  x_tr_imp <- ifelse(is.na(x_tr), med, x_tr)
  m <- mean(x_tr_imp)
  s <- stats::sd(x_tr_imp)
  if (!is.finite(s) || s <= 0) s <- 1.0
  list(med = med, mean = m, sd = s)
}
apply_x_1d <- function(x, prep) {
  x_imp <- ifelse(is.na(x), prep$med, x)
  (x_imp - prep$mean) / prep$sd
}

fit_gbm_reg <- function(x, y, seed) {
  set.seed(seed)
  df <- data.frame(x = x, y = y)
  gbm(
    y ~ x,
    data = df,
    distribution = "gaussian",
    n.trees = 2000,
    interaction.depth = 2,
    shrinkage = 0.02,
    bag.fraction = 0.7,
    train.fraction = 1.0,
    verbose = FALSE
  )
}

predict_gbm <- function(fit, x) {
  as.numeric(predict(fit, newdata = data.frame(x = x), n.trees = fit$n.trees))
}

run_ours_1d <- function(train_df, test_df,
                        alpha, delta,
                        n_sample, seed) {

  dfA <- train_df %>% dplyr::slice_sample(n = n_sample, replace = FALSE)
  xA <- dfA$mag_r
  yA <- dfA$z_spec
  zA <- tf(yA)

  # split 50/50
  set.seed(seed)
  n <- nrow(dfA)
  idx <- sample.int(n)
  n_tr <- floor(n/2)
  idx_tr <- idx[1:n_tr]
  idx_cal <- idx[(n_tr+1):n]

  x_tr_raw <- xA[idx_tr]; z_tr <- zA[idx_tr]
  x_cal_raw <- xA[idx_cal]; z_cal <- zA[idx_cal]

  # preprocess (impute+scale)
  prep <- prep_x_1d(x_tr_raw)
  x_tr <- apply_x_1d(x_tr_raw, prep)
  x_cal <- apply_x_1d(x_cal_raw, prep)

  # mean model
  mean_fit <- fit_gbm_reg(x_tr, z_tr, seed = seed)
  mu_tr <- predict_gbm(mean_fit, x_tr)

  # var model on residual^2
  res2_tr <- (z_tr - mu_tr)^2
  var_fit <- fit_gbm_reg(x_tr, res2_tr, seed = seed + 1)

  mu_cal <- predict_gbm(mean_fit, x_cal)
  var_cal <- pmax(predict_gbm(var_fit, x_cal), 1e-6)

  lam <- find_lambda_fast(alpha, delta, z_cal, mu_cal, var_cal)

  # test prediction
  x_te_raw <- test_df$mag_r
  y_te <- test_df$z_spec
  x_te <- apply_x_1d(x_te_raw, prep)

  mu_te <- predict_gbm(mean_fit, x_te)
  var_te <- pmax(predict_gbm(var_fit, x_te), 1e-6)

  lower <- itf(mu_te - lam * sqrt(var_te))
  upper <- itf(mu_te + lam * sqrt(var_te))

  content <- mean((y_te >= lower) & (y_te <= upper), na.rm = TRUE)
  mean_width <- mean(upper - lower, na.rm = TRUE)

  list(method = "Ours(1D mag_r)", lambda = lam, content = content, mean_width = mean_width)
}
