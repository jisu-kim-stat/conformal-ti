# R/sim/data_generate_alt.R
# Alternative DGPs for score-pivotality experiments

generate_data <- function(model_id, n, design = "uniform") {

  stopifnot(model_id %in% 1:6)
  design <- match.arg(design, c("uniform", "normal"))

  if (model_id %in% 1:5) {

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
    if (model_id == 2) {
      y <- base + rt(n, df = 3)
    }

    # Model 3: heteroscedastic symmetric Gaussian
    if (model_id == 3) {
      sigma <- 1 + abs(x)
      y <- base + sigma * rnorm(n, mean = 0, sd = 1)
    }

    # Model 4: homoscedastic skewed location-scale
    # Z = Exp(1) - 1, independent of X
    if (model_id == 4) {
      z <- rexp(n, rate = 1) - 1
      y <- base + z
    }

    # Model 5: X-dependent skewness, non-pivotal residual shape
    # Smooth mixture between Gaussian and shifted exponential
    if (model_id == 5) {
      w <- plogis(3 * x)

      is_exp <- rbinom(n, size = 1, prob = w)

      z_norm <- rnorm(n, mean = 0, sd = 1)
      z_exp  <- rexp(n, rate = 1) - 1

      z <- ifelse(is_exp == 1, z_exp, z_norm)

      y <- base + z
    }

    return(data.frame(x = x, y = y))
  }

  # Model 6: high-dimensional sparse Gaussian
  if (model_id == 6) {
    d <- 20

    x <- matrix(rnorm(n * d), nrow = n, ncol = d)
    beta <- c(0.5, 0.4, 0.3, 0.2, rep(0, d - 4))

    base <- as.vector(x %*% beta)
    y <- base + rnorm(n, mean = 0, sd = 1)

    colnames(x) <- paste0("x", seq_len(d))

    return(data.frame(x, y = y))
  }

  stop("Unknown model_id.")
}


generate_eval_data <- function(model_id, n, design = "uniform") {

  stopifnot(model_id %in% 1:6)
  design <- match.arg(design, c("uniform", "normal"))

  if (model_id %in% 1:5) {

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

  # For model 6, use ordinary random evaluation data.
  if (model_id == 6) {
    return(generate_data(model_id, n, design = design))
  }

  stop("Unknown model_id.")
}


generate_y_given_x <- function(model_id, x) {

  stopifnot(model_id %in% 1:5)

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
    z <- rexp(n, rate = 1) - 1
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

  stop("Unknown model_id.")
}