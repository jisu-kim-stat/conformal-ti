# Analytic conditional content for the balanced four-DGP suite.

content_split_normal <- function(lower, upper, base, sigma_minus, sigma_plus,
                                 mean_shift) {
  lo <- lower - base + mean_shift
  hi <- upper - base + mean_shift

  left <- pnorm(pmin(hi, 0) / sigma_minus) -
    pnorm(pmin(lo, 0) / sigma_minus)
  right <- pnorm(pmax(hi, 0) / sigma_plus) -
    pnorm(pmax(lo, 0) / sigma_plus)
  pmax(left, 0) + pmax(right, 0)
}

content_function <- function(model_id, lower, upper, x,
                             heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:4, length(lower) == length(upper), length(x) == length(lower))
  heavy_tail_scale <- match.arg(heavy_tail_scale)
  base <- base_mean(x)

  if (model_id == 1) return(pnorm(upper - base) - pnorm(lower - base))

  if (model_id == 2) {
    return(
      pt(sqrt(3) * (upper - base), df = 3) -
        pt(sqrt(3) * (lower - base), df = 3)
    )
  }

  if (model_id == 3) {
    sigma <- 1 + abs(x)
    return(pnorm(upper - base, sd = sigma) - pnorm(lower - base, sd = sigma))
  }

  p <- plogis(2 * x)
  sigma_minus <- 0.7 + 0.5 * p
  sigma_plus <- 1.3 - 0.5 * p
  mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)
  content_split_normal(lower, upper, base, sigma_minus, sigma_plus, mean_eps)
}
