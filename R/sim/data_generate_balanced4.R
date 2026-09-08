# Five-DGP simulation suite for the main paper.
# Models 1--4 have an x-invariant standardized residual distribution; Model 5
# has x-dependent residual shape and is not a location--scale model.

generate_data <- function(model_id, n, design = "uniform",
                          heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:5)
  design <- match.arg(design, c("uniform", "normal"))
  match.arg(heavy_tail_scale)

  x <- if (design == "uniform") runif(n, -2, 2) else rnorm(n)
  base <- base_mean(x)

  if (model_id == 1) {
    y <- base + rnorm(n)
  }

  if (model_id == 2) {
    y <- base + rt(n, df = 3)
  }

  if (model_id == 3) {
    y <- base + (1 + abs(x)) * rnorm(n)
  }

  if (model_id == 4) {
    y <- base + (rchisq(n, df = 2) - 2) / 2
  }

  if (model_id == 5) {
    # Both mixture components have mean zero and variance one.  The changing
    # mixture weight therefore changes conditional skewness, not location or
    # scale alone.
    w <- plogis(3 * x)
    is_skewed <- runif(n) < w
    eps <- rnorm(n)
    eps[is_skewed] <- rexp(sum(is_skewed)) - 1
    y <- base + eps
  }

  data.frame(x = x, y = y)
}

generate_eval_data <- function(model_id, n, design = "uniform") {
  stopifnot(model_id %in% 1:5)
  design <- match.arg(design, c("uniform", "normal"))
  p <- (seq_len(n) - 0.5) / n
  x <- if (design == "uniform") qunif(p, -2, 2) else qnorm(p)
  data.frame(x = x, y = rep(NA_real_, n))
}

generate_y_given_x <- function(model_id, x,
                                heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:5)
  match.arg(heavy_tail_scale)
  base <- base_mean(x)
  n <- length(x)

  if (model_id == 1) return(base + rnorm(n))
  if (model_id == 2) return(base + rt(n, df = 3))
  if (model_id == 3) return(base + (1 + abs(x)) * rnorm(n))
  if (model_id == 4) return(base + (rchisq(n, df = 2) - 2) / 2)

  w <- plogis(3 * x)
  is_skewed <- runif(n) < w
  eps <- rnorm(n)
  eps[is_skewed] <- rexp(sum(is_skewed)) - 1
  base + eps
}
