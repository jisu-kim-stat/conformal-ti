# R/sim/pti_utils.R
# ------------------------------------------------------------
# Finite-dimensional homoscedastic normal-regression TI used by Parametric-TI.
#
# This is deliberately separate from Guo and Young's Proposition 3.1 /
# Equation (14), which is implemented in guo_young_ti.R by numerical
# integration and root finding.
# ------------------------------------------------------------

classical_parametric_design <- function(x,
                                        basis_df = 10L,
                                        boundary_knots = c(-4, 4)) {
  x <- as.numeric(x)

  if (any(!is.finite(x))) {
    stop("x must be finite.", call. = FALSE)
  }

  # Use a fixed finite-dimensional spline basis selected independently of the
  # DGP.  In particular, do not supply the true sin(2*pi*x) mean feature to
  # this benchmark.
  cbind(
    "(Intercept)" = 1,
    splines::bs(
      x,
      df = basis_df,
      degree = 3L,
      intercept = FALSE,
      Boundary.knots = boundary_knots
    )
  )
}


find_parametric_k_factor <- function(nu,
                                     norm_lx,
                                     content,
                                     alpha) {
  stopifnot(
    is.finite(nu), nu > 0,
    all(is.finite(norm_lx)), all(norm_lx >= 0),
    is.finite(content), content > 0, content < 1,
    is.finite(alpha), alpha > 0, alpha < 1
  )

  q_content <- stats::qchisq(
    content,
    df = 1,
    ncp = norm_lx^2
  )
  q_confidence <- stats::qchisq(alpha, df = nu)

  if (!is.finite(q_confidence) || q_confidence <= 0) {
    return(rep(NA_real_, length(norm_lx)))
  }

  sqrt(nu * q_content / q_confidence)
}


classical_parametric_ti <- function(x,
                                    y,
                                    x_new = x,
                                    content = 0.90,
                                    alpha = 0.05,
                                    basis_df = 10L,
                                    boundary_knots = c(-4, 4)) {
  x <- as.numeric(x)
  y <- as.numeric(y)
  x_new <- as.numeric(x_new)

  if (length(x) != length(y)) {
    stop("x and y must have the same length.", call. = FALSE)
  }
  if (any(!is.finite(y))) {
    stop("y must be finite.", call. = FALSE)
  }

  design <- classical_parametric_design(
    x, basis_df = basis_df, boundary_knots = boundary_knots
  )
  design_new <- classical_parametric_design(
    x_new, basis_df = basis_df, boundary_knots = boundary_knots
  )
  n <- nrow(design)
  p <- ncol(design)
  nu <- n - p

  if (nu <= 0) {
    stop("The classical regression fit has no residual degrees of freedom.",
         call. = FALSE)
  }
  if (qr(design)$rank < p) {
    stop("The classical regression design matrix is rank deficient.",
         call. = FALSE)
  }

  xtx_inverse <- solve(crossprod(design), diag(p))
  beta_hat <- drop(xtx_inverse %*% crossprod(design, y))
  fitted <- drop(design %*% beta_hat)
  residual <- y - fitted
  sigma2_hat <- sum(residual^2) / nu
  sigma_hat <- sqrt(sigma2_hat)

  predicted <- drop(design_new %*% beta_hat)
  leverage <- rowSums((design_new %*% xtx_inverse) * design_new)
  leverage <- pmax(leverage, 0)

  k <- find_parametric_k_factor(
    nu = nu,
    norm_lx = sqrt(leverage),
    content = content,
    alpha = alpha
  )

  half_width <- k * sigma_hat
  interval <- cbind(
    lower = predicted - half_width,
    upper = predicted + half_width
  )

  list(
    interval = interval,
    fitted = predicted,
    k = k,
    leverage = leverage,
    beta_hat = beta_hat,
    sigma2_hat = sigma2_hat,
    sigma_hat = sigma_hat,
    nu = nu,
    content = content,
    confidence = 1 - alpha
  )
}


# Backward-compatible alias for older plotting scripts.
find_k_factor <- function(nu, norm_lx_h, content, alpha) {
  find_parametric_k_factor(
    nu = nu,
    norm_lx = norm_lx_h,
    content = content,
    alpha = alpha
  )
}
