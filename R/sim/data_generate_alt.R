# R/sim/data_generate_alt.R
# Alternative DGPs for score-pivotality experiments

generate_data <- function(model_id, n, design = "uniform") {

  stopifnot(model_id %in% 1:6)
  design <- match.arg(design, c("uniform", "normal"))

  if (design == "uniform") {
    x <- runif(n, min = -2, max = 2)
  }

  if (design == "normal") {
    x <- rnorm(n, mean = 0, sd = 1)
  }

  base <- base_mean(x)

  # Model 1: homoscedastic Gaussian
  if (model_id == 1) {
    y <- base + rnorm(n, mean = 0, sd = 1)
  }

  # Model 2: symmetric heavy-tailed
  # Residual distribution is common across x, but tails are heavy.
  if (model_id == 2) {
    y <- base + rt(n, df = 3)
  }

  # Model 3: heteroscedastic symmetric Gaussian
  # Standardized residual is pivotal if sigma(x) is well estimated.
  if (model_id == 3) {
    sigma <- 1 + abs(x)
    y <- base + sigma * rnorm(n, mean = 0, sd = 1)
  }

  # Model 4: strongly skewed homoscedastic location model
  # Z = (Chi-square_2 - 2) / 2, mean 0 and variance 1, independent of X.
  # This is the positive setting for HCTI-asym.
  if (model_id == 4) {
    z <- (rchisq(n, df = 2) - 2) / 2
    y <- base + z
  }

  # Model 5: x-dependent skewness
  # Residual shape changes with x, so exact residual-score pivotality is violated.
  if (model_id == 5) {
    w <- plogis(3 * x)

    is_exp <- rbinom(n, size = 1, prob = w)

    z_norm <- rnorm(n, mean = 0, sd = 1)
    z_exp  <- rexp(n, rate = 1) - 1

    z <- ifelse(is_exp == 1, z_exp, z_norm)

    y <- base + z
  }

  # Model 6: centered asymmetric heteroscedastic split-normal model
  # Lower and upper tail scales vary differently with x.
  # The raw split-normal error is centered so that E[Y | X=x] = base_mean(x).
  # This is a CQR-friendly setting with x-dependent lower/upper quantile shape.
  if (model_id == 6) {
    z <- rnorm(n, mean = 0, sd = 1)

    sigma_minus <- 0.6 + 0.4 * as.numeric(x < 0) + 0.15 * abs(x)
    sigma_plus  <- 0.7 + 0.8 * as.numeric(x > 0) + 0.30 * abs(x)

    eps_raw <- ifelse(z < 0, sigma_minus * z, sigma_plus * z)

    mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
    eps <- eps_raw - mean_eps

    y <- base + eps
  }

  return(data.frame(x = x, y = y))
}


generate_eval_data <- function(model_id, n, design = "uniform") {

  stopifnot(model_id %in% 1:6)
  design <- match.arg(design, c("uniform", "normal"))

  p <- (seq_len(n) - 0.5) / n

  if (design == "uniform") {
    x <- qunif(p, min = -2, max = 2)
  }

  if (design == "normal") {
    x <- qnorm(p, mean = 0, sd = 1)
  }

  return(data.frame(
    x = x,
    y = rep(NA_real_, n)
  ))
}


generate_y_given_x <- function(model_id, x) {

  stopifnot(model_id %in% 1:6)

  n <- length(x)
  base <- base_mean(x)

  if (model_id == 1) {
    return(base + rnorm(n, mean = 0, sd = 1))
  }

  if (model_id == 2) {
    return(base + rt(n, df = 3))
  }

  if (model_id == 3) {
    sigma <- 1 + abs(x)
    return(base + sigma * rnorm(n, mean = 0, sd = 1))
  }

  if (model_id == 4) {
    z <- (rchisq(n, df = 2) - 2) / 2
    return(base + z)
  }

  if (model_id == 5) {
    w <- plogis(3 * x)

    is_exp <- rbinom(n, size = 1, prob = w)

    z_norm <- rnorm(n, mean = 0, sd = 1)
    z_exp  <- rexp(n, rate = 1) - 1

    z <- ifelse(is_exp == 1, z_exp, z_norm)

    return(base + z)
  }

  if (model_id == 6) {
    z <- rnorm(n, mean = 0, sd = 1)

    sigma_minus <- 0.6 + 0.4 * as.numeric(x < 0) + 0.15 * abs(x)
    sigma_plus  <- 0.7 + 0.8 * as.numeric(x > 0) + 0.30 * abs(x)

    eps_raw <- ifelse(z < 0, sigma_minus * z, sigma_plus * z)

    mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
    eps <- eps_raw - mean_eps

    return(base + eps)
  }

  stop("Unknown model_id.")
}