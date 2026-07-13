# R/sim/data_generate.R
generate_data <- function(model_id, n) {

  if (model_id %in% 1:5) {
    x <- seq(-2, 2, length.out = n)
    base <- base_mean(x)

    if (model_id == 1) {
      y <- base + rnorm(n, 0, 1)
    }

    if (model_id == 2) {
      y <- base + rt(n, df = 3)
    }

    if (model_id == 3) {
      y <- base + rnorm(n, 0, 1) * (1 + abs(x))
    }

    if (model_id == 4) {
      y <- base + rexp(n, rate = 1) - 1
    }

    if (model_id == 5) {
      sigma <- sqrt(0.75 + 0.5 * sin(2 * pi * x)^2)
      y <- base + rnorm(n, 0, sigma)
    }

    return(data.frame(x = x, y = y))
  }

  if (model_id == 6) {
    d <- 20
    x <- matrix(rnorm(n * d), nrow = n, ncol = d)

    beta <- c(0.5, 0.4, 0.3, 0.2, rep(0, d - 4))

    base <- as.vector(x %*% beta)
    y <- base + rnorm(n, 0, 1)

    colnames(x) <- paste0("x", 1:d)

    return(data.frame(x, y = y))
  }
}