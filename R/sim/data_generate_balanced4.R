# Four-DGP simulation suite for the main paper.
# Models 1--3 retain the original location--scale settings. Model 4 has
# smoothly varying lower/upper tail scales and is not a location--scale model.

generate_data <- function(model_id, n, design = "uniform",
                          heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:4)
  design <- match.arg(design, c("uniform", "normal"))
  match.arg(heavy_tail_scale)

  x <- if (design == "uniform") runif(n, -2, 2) else rnorm(n)
  base <- base_mean(x)

  if (model_id == 1) {
    y <- base + rnorm(n)
  }

  if (model_id == 2) {
    y <- base + rt(n, df = 3) / sqrt(3)
  }

  if (model_id == 3) {
    y <- base + (1 + abs(x)) * rnorm(n)
  }

  if (model_id == 4) {
    p <- plogis(2 * x)
    sigma_minus <- 0.7 + 0.5 * p
    sigma_plus <- 1.3 - 0.5 * p
    z <- rnorm(n)
    eps_raw <- ifelse(z < 0, sigma_minus * z, sigma_plus * z)
    mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
    y <- base + eps_raw - mean_eps
  }

  data.frame(x = x, y = y)
}

generate_eval_data <- function(model_id, n, design = "uniform") {
  stopifnot(model_id %in% 1:4)
  design <- match.arg(design, c("uniform", "normal"))
  p <- (seq_len(n) - 0.5) / n
  x <- if (design == "uniform") qunif(p, -2, 2) else qnorm(p)
  data.frame(x = x, y = rep(NA_real_, n))
}

generate_y_given_x <- function(model_id, x,
                                heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:4)
  match.arg(heavy_tail_scale)
  base <- base_mean(x)
  n <- length(x)

  if (model_id == 1) return(base + rnorm(n))
  if (model_id == 2) return(base + rt(n, df = 3) / sqrt(3))
  if (model_id == 3) return(base + (1 + abs(x)) * rnorm(n))

  p <- plogis(2 * x)
  sigma_minus <- 0.7 + 0.5 * p
  sigma_plus <- 1.3 - 0.5 * p
  z <- rnorm(n)
  eps_raw <- ifelse(z < 0, sigma_minus * z, sigma_plus * z)
  mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
  base + eps_raw - mean_eps
}
