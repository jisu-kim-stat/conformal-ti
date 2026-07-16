# R/sim/truth_content_alt.R
# True conditional content for alternative DGPs

content_function <- function(model_id, lower, upper, x) {

  stopifnot(model_id %in% 1:6)

  if (model_id %in% 1:5) {

    base <- base_mean(x)

    # Model 1: homoscedastic Gaussian
    if (model_id == 1) {
      return(
        pnorm(upper - base, mean = 0, sd = 1) -
          pnorm(lower - base, mean = 0, sd = 1)
      )
    }

    # Model 2: symmetric heavy-tailed t_3
    if (model_id == 2) {
      return(
        pt(upper - base, df = 3) -
          pt(lower - base, df = 3)
      )
    }

    # Model 3: heteroscedastic Gaussian
    if (model_id == 3) {
      sigma <- 1 + abs(x)

      return(
        pnorm(upper - base, mean = 0, sd = sigma) -
          pnorm(lower - base, mean = 0, sd = sigma)
      )
    }

    # Model 4: shifted exponential Exp(1) - 1
    if (model_id == 4) {
      lo <- lower - base + 1
      hi <- upper - base + 1

      return(
        pexp(hi, rate = 1) -
          pexp(lo, rate = 1)
      )
    }

    # Model 5: smooth X-dependent skewness mixture
    if (model_id == 5) {
      w <- plogis(3 * x)

      content_norm <-
        pnorm(upper - base, mean = 0, sd = 1) -
        pnorm(lower - base, mean = 0, sd = 1)

      content_exp <-
        pexp(upper - base + 1, rate = 1) -
        pexp(lower - base + 1, rate = 1)

      return(
        (1 - w) * content_norm +
          w * content_exp
      )
    }
  }

  # Model 6: high-dimensional Gaussian
  if (model_id == 6) {
    x <- as.matrix(x)

    beta <- c(0.5, 0.4, 0.3, 0.2, rep(0, 16))
    base <- as.vector(x %*% beta)

    return(
      pnorm(upper - base, mean = 0, sd = 1) -
        pnorm(lower - base, mean = 0, sd = 1)
    )
  }

  stop("Unknown model_id.")
}