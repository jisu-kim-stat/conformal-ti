# R/sim/truth_content_alt.R
# True conditional content functions for alternative DGPs

content_split_normal <- function(lower, upper, base, sigma_minus, sigma_plus,
                                 mean_shift = 0) {
  lo <- lower - base + mean_shift
  hi <- upper - base + mean_shift

  # Left side: eps_raw = sigma_minus * Z, Z < 0
  left_lo <- pmin(lo, 0)
  left_hi <- pmin(hi, 0)

  p_left <- pnorm(left_hi / sigma_minus) - pnorm(left_lo / sigma_minus)

  # Right side: eps_raw = sigma_plus * Z, Z >= 0
  right_lo <- pmax(lo, 0)
  right_hi <- pmax(hi, 0)

  p_right <- pnorm(right_hi / sigma_plus) - pnorm(right_lo / sigma_plus)

  pmax(p_left, 0) + pmax(p_right, 0)
}


content_function <- function(model_id, lower, upper, x) {

  stopifnot(model_id %in% 1:6)

  if (length(lower) != length(upper)) {
    stop("lower and upper must have the same length.")
  }

  if (length(x) != length(lower)) {
    stop("x, lower, and upper must have the same length.")
  }

  base <- base_mean(x)

  # Model 1: homoscedastic Gaussian
  # Y = mu(x) + Z, Z ~ N(0, 1)
  if (model_id == 1) {
    return(
      pnorm(upper - base, mean = 0, sd = 1) -
        pnorm(lower - base, mean = 0, sd = 1)
    )
  }

  # Model 2: symmetric heavy-tailed
  # Y = mu(x) + Z, Z ~ t_3
  if (model_id == 2) {
    return(
      pt(upper - base, df = 3) -
        pt(lower - base, df = 3)
    )
  }

  # Model 3: heteroscedastic symmetric Gaussian
  # Y = mu(x) + sigma(x) Z, sigma(x) = 1 + |x|
  if (model_id == 3) {
    sigma <- 1 + abs(x)

    return(
      pnorm(upper - base, mean = 0, sd = sigma) -
        pnorm(lower - base, mean = 0, sd = sigma)
    )
  }

  # Model 4: strongly skewed homoscedastic location model
  # Z = (Chi-square_2 - 2) / 2
  # If lower <= mu + Z <= upper, then
  # 2(lower - mu) + 2 <= Chi-square_2 <= 2(upper - mu) + 2.
  if (model_id == 4) {
    lo_w <- 2 * (lower - base) + 2
    hi_w <- 2 * (upper - base) + 2

    return(
      pchisq(hi_w, df = 2) -
        pchisq(lo_w, df = 2)
    )
  }

  # Model 5: X-dependent skewness
  # Z_X ~ (1 - w(x)) N(0,1) + w(x) {Exp(1) - 1}
  if (model_id == 5) {
    w <- plogis(3 * x)

    content_norm <-
      pnorm(upper - base, mean = 0, sd = 1) -
      pnorm(lower - base, mean = 0, sd = 1)

    # Exp(1)-1 <= z iff Exp(1) <= z + 1
    lo_exp <- lower - base + 1
    hi_exp <- upper - base + 1

    content_exp <- pexp(hi_exp, rate = 1) - pexp(lo_exp, rate = 1)

    return(
      (1 - w) * content_norm + w * content_exp
    )
  }

  # Model 6: centered asymmetric heteroscedastic split-normal model
  # eps = eps_raw - mean_eps, where
  # eps_raw = sigma_minus(x) Z for Z < 0,
  # eps_raw = sigma_plus(x) Z for Z >= 0.
  if (model_id == 6) {
    sigma_minus <- 0.6 + 0.4 * as.numeric(x < 0) + 0.15 * abs(x)
    sigma_plus  <- 0.7 + 0.8 * as.numeric(x > 0) + 0.30 * abs(x)

    mean_eps <- (sigma_plus - sigma_minus) / sqrt(2 * pi)

    return(
      content_split_normal(
        lower = lower,
        upper = upper,
        base = base,
        sigma_minus = sigma_minus,
        sigma_plus = sigma_plus,
        mean_shift = mean_eps
      )
    )
  }

  stop("Unknown model_id.")
}