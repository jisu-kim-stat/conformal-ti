# Analytic conditional content for the five-DGP suite.

content_function <- function(model_id, lower, upper, x,
                             heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:5, length(lower) == length(upper), length(x) == length(lower))
  heavy_tail_scale <- match.arg(heavy_tail_scale)
  base <- base_mean(x)

  if (model_id == 1) return(pnorm(upper - base) - pnorm(lower - base))

  if (model_id == 2) {
    return(
      pt(upper - base, df = 3) - pt(lower - base, df = 3)
    )
  }

  if (model_id == 3) {
    sigma <- 1 + abs(x)
    return(pnorm(upper - base, sd = sigma) - pnorm(lower - base, sd = sigma))
  }

  if (model_id == 4) {
    residual_cdf <- function(z) pchisq(pmax(2 * z + 2, 0), df = 2)
    return(residual_cdf(upper - base) - residual_cdf(lower - base))
  }

  # Model 5 mixes N(0,1) and Exp(1)-1.  Its component CDFs are both known.
  w <- plogis(3 * x)
  residual_cdf <- function(z) {
    exp_component <- ifelse(z < -1, 0, pexp(z + 1))
    (1 - w) * pnorm(z) + w * exp_component
  }
  residual_cdf(upper - base) - residual_cdf(lower - base)
}

true_conditional_quantile <- function(model_id, x, probability) {
  stopifnot(model_id %in% 1:5, probability > 0, probability < 1)
  base <- base_mean(x)

  if (model_id == 1) return(base + qnorm(probability))
  if (model_id == 2) return(base + qt(probability, df = 3))
  if (model_id == 3) return(base + (1 + abs(x)) * qnorm(probability))
  if (model_id == 4) return(base + (qchisq(probability, df = 2) - 2) / 2)

  w <- plogis(3 * x)
  residual_cdf <- function(z, weight) {
    exp_component <- ifelse(z < -1, 0, pexp(z + 1))
    (1 - weight) * pnorm(z) + weight * exp_component
  }
  base + vapply(w, function(weight) {
    uniroot(function(z) residual_cdf(z, weight) - probability,
            interval = c(-12, 20), tol = 1e-10)$root
  }, numeric(1))
}
