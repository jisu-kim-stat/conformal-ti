# Analytic conditional content for the five-DGP suite.

content_function <- function(model_id, lower, upper, x,
                             heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:5, length(lower) == length(upper), length(x) == length(lower))
  heavy_tail_scale <- match.arg(heavy_tail_scale)
  base <- base_mean(x)

  if (model_id == 1) return(pnorm(upper - base) - pnorm(lower - base))

  if (model_id == 2) {
    sigma <- hetero_scale(x)
    return(
      pt((upper - base) / sigma, df = 3) -
        pt((lower - base) / sigma, df = 3)
    )
  }

  if (model_id == 3) {
    sigma <- hetero_scale(x)
    return(pnorm(upper - base, sd = sigma) - pnorm(lower - base, sd = sigma))
  }

  if (model_id == 4) {
    sigma <- hetero_scale(x)
    residual_cdf <- function(z) pgamma(pmax(2 * z + 4, 0), shape = 4, rate = 1)
    return(
      residual_cdf((upper - base) / sigma) -
        residual_cdf((lower - base) / sigma)
    )
  }

  # Model 5 is a recentered two-piece Gaussian with x-dependent upper-tail
  # scale. Its distribution is continuous at the recentered split point.
  s_minus <- hetero_scale(x)
  s_plus <- s_minus * (1 + 0.8 * plogis(1.5 * x))
  center <- (s_plus - s_minus) / sqrt(2 * pi)
  residual_cdf <- function(z) {
    raw <- z + center
    ifelse(raw < 0, pnorm(raw / s_minus), pnorm(raw / s_plus))
  }
  residual_cdf(upper - base) - residual_cdf(lower - base)
}

true_conditional_quantile <- function(model_id, x, probability) {
  stopifnot(model_id %in% 1:5, probability > 0, probability < 1)
  base <- base_mean(x)

  if (model_id == 1) return(base + qnorm(probability))
  if (model_id == 2) return(base + hetero_scale(x) * qt(probability, df = 3))
  if (model_id == 3) return(base + hetero_scale(x) * qnorm(probability))
  if (model_id == 4) {
    return(base + hetero_scale(x) * (qgamma(probability, shape = 4, rate = 1) - 4) / 2)
  }

  s_minus <- hetero_scale(x)
  s_plus <- s_minus * (1 + 0.8 * plogis(1.5 * x))
  center <- (s_plus - s_minus) / sqrt(2 * pi)
  raw_quantile <- if (probability < 0.5) {
    s_minus * qnorm(probability)
  } else {
    s_plus * qnorm(probability)
  }
  base + raw_quantile - center
}
