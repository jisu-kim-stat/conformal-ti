# R/sim/truth_content.R
content_function <- function(model_id, lower, upper, x) {

  if (model_id %in% 1:5) {

    base <- base_mean(x)

    if (model_id == 1) {
      # Y = base + N(0, 1)
      return(
        pnorm(upper - base, mean = 0, sd = 1) -
          pnorm(lower - base, mean = 0, sd = 1)
      )
    }

    if (model_id == 2) {
      # Y = base + t_3
      return(
        pt(upper - base, df = 3) -
          pt(lower - base, df = 3)
      )
    }

    if (model_id == 3) {
      # Y = base + (1 + |x|) * N(0, 1)
      sd_x <- 1 + abs(x)

      return(
        pnorm(upper - base, mean = 0, sd = sd_x) -
          pnorm(lower - base, mean = 0, sd = sd_x)
      )
    }

    if (model_id == 4) {
      # Y = base + Exp(1) - 1
      #
      # lower <= base + E - 1 <= upper
      # lower - base + 1 <= E <= upper - base + 1
      lo <- lower - base + 1
      hi <- upper - base + 1

      return(
        pexp(hi, rate = 1) - pexp(lo, rate = 1)
      )
    }

    if (model_id == 5) {
      # Y = base + sigma(x) * N(0, 1)
      # sigma^2(x) = 0.75 + 0.5 sin^2(2 pi x)
      sigma_x <- sqrt(0.75 + 0.5 * sin(2 * pi * x)^2)

      return(
        pnorm(upper - base, mean = 0, sd = sigma_x) -
          pnorm(lower - base, mean = 0, sd = sigma_x)
      )
    }
  }

  if (model_id == 6) {
    # X is n x 20 or 1 x 20.
    # Y = X beta + N(0, 1)

    x <- as.matrix(x)

    beta <- c(0.5, 0.4, 0.3, 0.2, rep(0, 16))

    base <- as.vector(x %*% beta)

    return(
      pnorm(upper - base, mean = 0, sd = 1) -
        pnorm(lower - base, mean = 0, sd = 1)
    )
  }

  stop("Unknown model.")
}