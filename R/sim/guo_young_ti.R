# R/sim/guo_young_ti.R
# ------------------------------------------------------------
# Guo and Young (2024), pointwise two-sided tolerance interval
#
# Model:
#   Y_i = f(x_i) + epsilon_i,
#   E(epsilon_i) = 0, Var(epsilon_i) = sigma^2.
#
# This file implements:
#   - Equation (11): sigma_hat^2 and Satterthwaite df nu
#   - Appendix Lemma A.1(3): fast approximate pointwise k-factor (default)
#   - Proposition 3.1 / Equation (14): integral k-factor (optional)
#
# The regression fit is R's GCV-selected cubic smoothing spline.  Its
# linear-smoother geometry is reconstructed from the matrices retained by
# smooth.spline(keep.stuff = TRUE), rather than approximated with an
# unrelated B-spline basis.
#
# Notation:
#   content = P in Guo and Young
#   gamma   = confidence level in Guo and Young
# ------------------------------------------------------------

.gy_unpack_symmetric_band <- function(values, dimension, bandwidth = 4L) {
  stopifnot(
    length(values) == bandwidth * dimension,
    dimension >= 1L,
    bandwidth >= 1L
  )

  out <- matrix(0, nrow = dimension, ncol = dimension)

  for (offset in 0:(bandwidth - 1L)) {
    block <- values[offset * dimension + seq_len(dimension)]

    if (offset == 0L) {
      diag(out) <- block
    } else if (offset < dimension) {
      rows <- seq_len(dimension - offset)
      cols <- rows + offset
      out[cbind(rows, cols)] <- block[rows]
      out[cbind(cols, rows)] <- block[rows]
    }
  }

  out
}


.gy_spline_basis <- function(spline_fit, x_new) {
  fit <- spline_fit$fit
  x_new <- as.numeric(x_new)

  scaled_x <- (x_new - fit$min) / fit$range
  inside <- scaled_x >= 0 & scaled_x <= 1

  basis <- matrix(0, nrow = length(x_new), ncol = fit$nk)

  if (any(inside)) {
    basis[inside, ] <- splines::splineDesign(
      knots = fit$knot,
      x = scaled_x[inside],
      ord = 4L,
      derivs = 0L,
      outer.ok = TRUE
    )
  }

  # Natural cubic smoothing splines extrapolate linearly.
  left <- scaled_x < 0
  if (any(left)) {
    basis_left <- splines::splineDesign(
      knots = fit$knot,
      x = 0,
      ord = 4L,
      derivs = 0L,
      outer.ok = TRUE
    )
    slope_left <- splines::splineDesign(
      knots = fit$knot,
      x = 0,
      ord = 4L,
      derivs = 1L,
      outer.ok = TRUE
    )

    basis[left, ] <- matrix(
      basis_left,
      nrow = sum(left),
      ncol = fit$nk,
      byrow = TRUE
    ) + matrix(
      slope_left,
      nrow = sum(left),
      ncol = fit$nk,
      byrow = TRUE
    ) * scaled_x[left]
  }

  right <- scaled_x > 1
  if (any(right)) {
    basis_right <- splines::splineDesign(
      knots = fit$knot,
      x = 1,
      ord = 4L,
      derivs = 0L,
      outer.ok = TRUE
    )
    slope_right <- splines::splineDesign(
      knots = fit$knot,
      x = 1,
      ord = 4L,
      derivs = 1L,
      outer.ok = TRUE
    )
    delta <- scaled_x[right] - 1

    basis[right, ] <- matrix(
      basis_right,
      nrow = sum(right),
      ncol = fit$nk,
      byrow = TRUE
    ) + matrix(
      slope_right,
      nrow = sum(right),
      ncol = fit$nk,
      byrow = TRUE
    ) * delta
  }

  basis
}


.gy_trace <- function(x) {
  sum(diag(x))
}


gy_fit_smoothing_spline <- function(x,
                                    y,
                                    spar = NULL,
                                    all_knots = FALSE,
                                    nknots = NULL,
                                    control_spar = list()) {
  x <- as.numeric(x)
  y <- as.numeric(y)

  if (length(x) != length(y)) {
    stop("x and y must have the same length.", call. = FALSE)
  }
  if (length(x) < 4L || length(unique(x)) < 4L) {
    stop("At least four observations and four unique x values are required.",
         call. = FALSE)
  }
  if (any(!is.finite(x)) || any(!is.finite(y))) {
    stop("x and y must be finite.", call. = FALSE)
  }

  fit_args <- list(
    x = x,
    y = y,
    cv = FALSE,
    all.knots = all_knots,
    keep.data = TRUE,
    keep.stuff = TRUE,
    control.spar = control_spar
  )
  if (!is.null(nknots)) {
    fit_args$nknots <- nknots
  }
  if (!is.null(spar)) {
    fit_args$spar <- spar
  }

  spline_fit <- do.call(stats::smooth.spline, fit_args)
  basis_train <- .gy_spline_basis(spline_fit, x)
  nk <- spline_fit$fit$nk

  cross_basis <- .gy_unpack_symmetric_band(
    spline_fit$auxM$XWX,
    dimension = nk
  )
  penalty <- .gy_unpack_symmetric_band(
    spline_fit$auxM$Sigma,
    dimension = nk
  )

  penalized_cross_basis <- cross_basis + spline_fit$lambda * penalty
  inverse_penalized_cross_basis <- solve(
    penalized_cross_basis,
    diag(nk)
  )

  # Use the coefficients returned by smooth.spline's banded solver for
  # fitted values. Re-solving the same system as a dense matrix can lose a
  # few decimal places for occasional, moderately ill-conditioned GCV fits,
  # even though the fitted spline itself is numerically stable.
  beta <- spline_fit$fit$coef
  fitted <- drop(basis_train %*% beta)

  reference_fitted <- stats::predict(spline_fit, x)$y
  fit_error <- max(abs(fitted - reference_fitted))
  fit_tolerance <- 1e-8 * (1 + max(abs(reference_fitted)))

  if (!is.finite(fit_error) || fit_error > fit_tolerance) {
    stop(
      "Failed to evaluate the retained smooth.spline coefficients; ",
      "maximum fitted-value discrepancy = ",
      signif(fit_error, 6),
      ".",
      call. = FALSE
    )
  }

  # If L = B K^{-1} B^T, the nonzero eigenvalues of L equal
  # those of H = K^{-1} B^T B.  This gives exact trace formulas
  # without forming an n by n smoother matrix.
  h_basis <- inverse_penalized_cross_basis %*% cross_basis
  h2 <- h_basis %*% h_basis
  h3 <- h2 %*% h_basis
  h4 <- h2 %*% h2

  tr_l <- .gy_trace(h_basis)
  tr_l2 <- .gy_trace(h2)
  tr_l3 <- .gy_trace(h3)
  tr_l4 <- .gy_trace(h4)

  # A = (I - L)^T(I - L); L is symmetric for this smoother.
  trace_a <- length(y) - 2 * tr_l + tr_l2
  trace_a2 <- length(y) - 4 * tr_l + 6 * tr_l2 - 4 * tr_l3 + tr_l4

  if (trace_a <= 0 || trace_a2 <= 0) {
    stop("The smoothing-spline residual matrix has invalid traces.",
         call. = FALSE)
  }

  residual <- y - fitted
  sigma2_hat <- sum(residual^2) / trace_a
  nu <- trace_a^2 / trace_a2

  list(
    spline_fit = spline_fit,
    basis_train = basis_train,
    cross_basis = cross_basis,
    penalty = penalty,
    inverse_penalized_cross_basis = inverse_penalized_cross_basis,
    beta = beta,
    fitted = fitted,
    residual = residual,
    sigma2_hat = sigma2_hat,
    sigma_hat = sqrt(sigma2_hat),
    nu = nu,
    trace_a = trace_a,
    trace_a2 = trace_a2,
    lambda = spline_fit$lambda,
    spar = spline_fit$spar,
    effective_df = tr_l,
    fit_error = fit_error
  )
}


gy_predict_smoother <- function(model, x_new) {
  basis_new <- .gy_spline_basis(model$spline_fit, x_new)
  fitted <- drop(basis_new %*% model$beta)

  # ell_x = b(x)^T K^{-1} B^T, so
  # ||ell_x||^2 = b(x)^T K^{-1} B^T B K^{-1} b(x).
  ell_quadratic <- model$inverse_penalized_cross_basis %*%
    model$cross_basis %*%
    model$inverse_penalized_cross_basis

  ell_norm2 <- rowSums((basis_new %*% ell_quadratic) * basis_new)

  list(
    fitted = fitted,
    ell_norm = sqrt(pmax(ell_norm2, 0)),
    basis = basis_new
  )
}


gy_appendix_k <- function(ell_norm,
                          nu,
                          content = 0.90,
                          gamma = 0.95) {
  if (any(!is.finite(ell_norm)) || any(ell_norm < 0)) {
    stop("ell_norm must be nonnegative and finite.", call. = FALSE)
  }
  if (!is.finite(nu) || nu <= 0) {
    stop("nu must be positive and finite.", call. = FALSE)
  }
  if (!is.finite(content) || content <= 0 || content >= 1) {
    stop("content must lie in (0, 1).", call. = FALSE)
  }
  if (!is.finite(gamma) || gamma <= 0 || gamma >= 1) {
    stop("gamma must be the confidence level in (0, 1).",
         call. = FALSE)
  }

  # Guo and Young (2024), Appendix Lemma A.1(3):
  #   k_2 ~= sqrt{nu * chi^2_{1;P}(||ell_x||^2) /
  #                      chi^2_{nu;1-gamma}}.
  numerator <- stats::qchisq(
    content,
    df = 1,
    ncp = ell_norm^2
  )
  denominator <- stats::qchisq(1 - gamma, df = nu)

  sqrt(nu * numerator / denominator)
}


# qchisq(content, df = 1, ncp = t^2) is the expensive part of
# Equation (14).  Cache it on a fine t grid.  Above t = 8,
# (t + qnorm(content))^2 is equal to the noncentral quantile to
# machine precision because the opposite normal tail is negligible.
.gy_qchisq_cache <- new.env(parent = emptyenv())


.gy_noncentral_chisq1_quantile <- function(t,
                                           content,
                                           use_cache = TRUE,
                                           grid_step = 5e-4) {
  t <- abs(as.numeric(t))

  if (!use_cache) {
    return(stats::qchisq(content, df = 1, ncp = t^2))
  }

  key <- paste0(
    formatC(content, digits = 15, format = "fg"),
    "_",
    formatC(grid_step, digits = 15, format = "fg")
  )

  if (!exists(key, envir = .gy_qchisq_cache, inherits = FALSE)) {
    t_grid <- seq(0, 8, by = grid_step)
    q_grid <- stats::qchisq(content, df = 1, ncp = t_grid^2)

    assign(
      key,
      # A monotone C1 interpolant avoids the thousands of small kinks
      # produced by linear interpolation. Those kinks can make
      # integrate() report a roundoff error at an otherwise valid k.
      stats::splinefun(t_grid, q_grid, method = "monoH.FC"),
      envir = .gy_qchisq_cache
    )
  }

  interpolate <- get(key, envir = .gy_qchisq_cache, inherits = FALSE)
  out <- numeric(length(t))
  small <- t <= 8

  out[small] <- interpolate(t[small])
  out[!small] <- (t[!small] + stats::qnorm(content))^2
  out
}


gy_equation14_probability <- function(k,
                                      ell_norm,
                                      nu,
                                      content,
                                      rel_tol = 1e-8,
                                      subdivisions = 300L,
                                      use_quantile_cache = TRUE) {
  if (!is.finite(k) || k <= 0) {
    stop("k must be positive and finite.", call. = FALSE)
  }
  if (!is.finite(ell_norm) || ell_norm < 0) {
    stop("ell_norm must be nonnegative and finite.", call. = FALSE)
  }
  if (!is.finite(nu) || nu <= 0) {
    stop("nu must be positive and finite.", call. = FALSE)
  }
  if (!is.finite(content) || content <= 0 || content >= 1) {
    stop("content must lie in (0, 1).", call. = FALSE)
  }

  if (ell_norm <= sqrt(.Machine$double.eps)) {
    threshold <- nu * stats::qchisq(content, df = 1) / k^2
    return(stats::pchisq(
      threshold,
      df = nu,
      lower.tail = FALSE
    ))
  }

  # Equation (14), after t = ||ell_x|| z:
  # sqrt(2/pi) int_0^infinity Pr{...} exp(-z^2/2) dz.
  integrand <- function(z) {
    q_noncentral <- .gy_noncentral_chisq1_quantile(
      t = ell_norm * z,
      content = content,
      use_cache = use_quantile_cache
    )
    threshold <- nu * q_noncentral / k^2

    sqrt(2 / pi) *
      stats::pchisq(threshold, df = nu, lower.tail = FALSE) *
      exp(-z^2 / 2)
  }

  stats::integrate(
    integrand,
    lower = 0,
    # The omitted half-normal probability above 12 is < 4e-33.
    upper = 12,
    rel.tol = rel_tol,
    subdivisions = subdivisions,
    stop.on.error = TRUE
  )$value
}


gy_two_sided_k <- function(ell_norm,
                           nu,
                           content = 0.90,
                           gamma = 0.95,
                           rel_tol = 1e-8,
                           root_tol = 1e-8,
                           subdivisions = 300L,
                           use_quantile_cache = TRUE) {
  if (!is.finite(gamma) || gamma <= 0 || gamma >= 1) {
    stop("gamma must be the confidence level in (0, 1).",
         call. = FALSE)
  }

  if (ell_norm <= sqrt(.Machine$double.eps)) {
    return(sqrt(
      nu * stats::qchisq(content, df = 1) /
        stats::qchisq(1 - gamma, df = nu)
    ))
  }

  objective <- function(k) {
    gy_equation14_probability(
      k = k,
      ell_norm = ell_norm,
      nu = nu,
      content = content,
      rel_tol = rel_tol,
      subdivisions = subdivisions,
      use_quantile_cache = use_quantile_cache
    ) - gamma
  }

  lower <- 1e-6
  upper <- 2

  while (objective(upper) < 0) {
    upper <- 2 * upper
    if (upper > 1e4) {
      stop("Could not bracket the Guo-Young Equation (14) k-factor.",
           call. = FALSE)
    }
  }

  stats::uniroot(
    objective,
    interval = c(lower, upper),
    tol = root_tol
  )$root
}


gy_pointwise_ti <- function(x,
                            y,
                            x_new = x,
                            content = 0.90,
                            gamma = 0.95,
                            spar = NULL,
                            all_knots = FALSE,
                            nknots = NULL,
                            control_spar = list(),
                            k_method = c("appendix", "equation14"),
                            k_cache_digits = 10L,
                            rel_tol = 1e-8,
                            root_tol = 1e-8,
                            subdivisions = 300L,
                            use_quantile_cache = TRUE) {
  k_method <- match.arg(k_method)

  model <- gy_fit_smoothing_spline(
    x = x,
    y = y,
    spar = spar,
    all_knots = all_knots,
    nknots = nknots,
    control_spar = control_spar
  )

  prediction <- gy_predict_smoother(model, x_new)

  if (k_method == "appendix") {
    k <- gy_appendix_k(
      ell_norm = prediction$ell_norm,
      nu = model$nu,
      content = content,
      gamma = gamma
    )
  } else {
    cache_key <- formatC(
      prediction$ell_norm,
      digits = k_cache_digits,
      format = "fg",
      flag = "#"
    )
    unique_key <- unique(cache_key)
    representative_norm <- vapply(
      unique_key,
      function(key) prediction$ell_norm[match(key, cache_key)],
      numeric(1)
    )

    unique_k <- vapply(
      representative_norm,
      gy_two_sided_k,
      numeric(1),
      nu = model$nu,
      content = content,
      gamma = gamma,
      rel_tol = rel_tol,
      root_tol = root_tol,
      subdivisions = subdivisions,
      use_quantile_cache = use_quantile_cache
    )
    k <- unname(unique_k[match(cache_key, unique_key)])
  }

  half_width <- k * model$sigma_hat
  interval <- cbind(
    lower = prediction$fitted - half_width,
    upper = prediction$fitted + half_width
  )

  list(
    interval = interval,
    fitted = prediction$fitted,
    ell_norm = prediction$ell_norm,
    k = k,
    sigma_hat = model$sigma_hat,
    sigma2_hat = model$sigma2_hat,
    nu = model$nu,
    content = content,
    gamma = gamma,
    k_method = k_method,
    model = model
  )
}
