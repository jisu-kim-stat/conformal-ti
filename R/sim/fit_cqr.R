fixed_cqr_knots <- function(design, basis_df) {
  design <- match.arg(design, c("uniform", "normal"))
  stopifnot(basis_df >= 4)

  # The knots and boundary knots are fixed before observing the training
  # sample.  This avoids the unstable extrapolation induced when each
  # replicate uses its observed min/max as B-spline boundary knots.
  if (design == "uniform") {
    boundary <- c(-2, 2)
    interior <- seq(-2, 2, length.out = basis_df + 1L)[-c(1L, basis_df + 1L)]
  } else {
    boundary <- c(-4, 4)
    probs <- seq(0.05, 0.95, length.out = basis_df - 1L)
    interior <- stats::qnorm(probs)
  }

  list(boundary = boundary, interior = interior)
}

pinball_loss <- function(y, fitted, tau) {
  residual <- y - fitted
  mean(ifelse(residual >= 0, tau * residual, (tau - 1) * residual))
}

fit_fixed_ns_quantile <- function(x, y, tau, basis_df, design) {
  knots <- fixed_cqr_knots(design, basis_df)
  quantreg::rq(
    y ~ splines::ns(
      x,
      knots = knots$interior,
      Boundary.knots = knots$boundary
    ),
    tau = tau,
    data = data.frame(x = x, y = y)
  )
}

select_quantile_spline_df <- function(x, y, tau, design, candidate_dfs,
                                      fold_id) {
  stopifnot(length(x) == length(y), length(y) == length(fold_id))
  candidate_dfs <- sort(unique(as.integer(candidate_dfs)))
  stopifnot(length(candidate_dfs) >= 1L, all(candidate_dfs >= 4L))

  cv_loss <- vapply(candidate_dfs, function(df) {
    fold_loss <- vapply(sort(unique(fold_id)), function(fold) {
      train_idx <- fold_id != fold
      valid_idx <- !train_idx
      fit <- fit_fixed_ns_quantile(
        x[train_idx], y[train_idx], tau = tau,
        basis_df = df, design = design
      )
      pred <- predict_quantile(fit, x[valid_idx])
      pinball_loss(y[valid_idx], pred, tau)
    }, numeric(1))
    mean(fold_loss)
  }, numeric(1))

  candidate_dfs[which.min(cv_loss)]
}

fit_rqss_quantile <- function(x, y, tau, lambda) {
  qss <- quantreg::qss
  fit <- quantreg::rqss(
    y ~ qss(x, lambda = lambda),
    tau = tau,
    data = data.frame(x = x, y = y)
  )
  attr(fit, "cqr_x_range") <- range(x)
  fit
}

select_quantile_rqss_lambda <- function(x, y, tau, candidate_lambdas,
                                        fold_id) {
  stopifnot(length(x) == length(y), length(y) == length(fold_id))
  candidate_lambdas <- sort(unique(as.numeric(candidate_lambdas)))
  stopifnot(length(candidate_lambdas) >= 1L, all(is.finite(candidate_lambdas)),
            all(candidate_lambdas > 0))

  cv_loss <- vapply(candidate_lambdas, function(lambda) {
    fold_loss <- vapply(sort(unique(fold_id)), function(fold) {
      train_idx <- fold_id != fold
      valid_idx <- !train_idx
      fit <- fit_rqss_quantile(x[train_idx], y[train_idx], tau, lambda)
      pinball_loss(y[valid_idx], predict_quantile(fit, x[valid_idx]), tau)
    }, numeric(1))
    mean(fold_loss)
  }, numeric(1))

  candidate_lambdas[which.min(cv_loss)]
}

fit_quantile_model <- function(x, y, tau, basis_df = 8,
                               design = c("uniform", "normal"),
                               basis_type = c("cv_fixed_ns", "cv_rqss", "fixed_ns", "legacy_bs"),
                               candidate_dfs = c(4, 6, 8, 10, 12),
                               candidate_lambdas = c(0.01, 0.03, 0.1, 0.3, 1, 3),
                               fold_id = NULL) {
  design <- match.arg(design)
  basis_type <- match.arg(basis_type)
  df <- data.frame(
    x = x,
    y = y
  )

  if (basis_type == "cv_rqss") {
    if (is.null(fold_id)) {
      fold_id <- sample(rep(seq_len(5L), length.out = length(y)))
    }
    lambda <- select_quantile_rqss_lambda(
      x, y, tau = tau, candidate_lambdas = candidate_lambdas, fold_id = fold_id
    )
    fit <- fit_rqss_quantile(x, y, tau = tau, lambda = lambda)
    attr(fit, "cqr_selected_lambda") <- lambda
    return(fit)
  }

  if (basis_type %in% c("cv_fixed_ns", "fixed_ns")) {
    if (basis_type == "cv_fixed_ns") {
      if (is.null(fold_id)) {
        fold_id <- sample(rep(seq_len(5L), length.out = length(y)))
      }
      basis_df <- select_quantile_spline_df(
        x, y, tau = tau, design = design,
        candidate_dfs = candidate_dfs, fold_id = fold_id
      )
    }
    fit <- fit_fixed_ns_quantile(x, y, tau, basis_df = basis_df, design = design)
    attr(fit, "cqr_selected_df") <- basis_df
    return(fit)
  }

  quantreg::rq(
    y ~ splines::bs(x, df = basis_df),
    tau = tau,
    data = df
  )
}

predict_quantile <- function(fit_quantile, xnew) {
  if (inherits(fit_quantile, "rqss")) {
    x_range <- attr(fit_quantile, "cqr_x_range")
    x_inside <- pmin(pmax(xnew, x_range[1]), x_range[2])
    fitted <- as.numeric(predict(fit_quantile, newdata = data.frame(x = x_inside)))

    # rqss does not extrapolate.  Holding the endpoint prediction constant
    # caused systematic undercoverage in the covariate tails, so extend the
    # fitted endpoint linearly using a short, in-support secant slope.
    span <- diff(x_range)
    step <- max(1e-6, 0.01 * span)
    left_base <- x_range[1]
    right_base <- x_range[2]
    left_next <- min(left_base + step, right_base)
    right_prev <- max(right_base - step, left_base)
    left_slope <- (
      as.numeric(predict(fit_quantile, newdata = data.frame(x = left_next))) -
        as.numeric(predict(fit_quantile, newdata = data.frame(x = left_base)))
    ) / (left_next - left_base)
    right_slope <- (
      as.numeric(predict(fit_quantile, newdata = data.frame(x = right_base))) -
        as.numeric(predict(fit_quantile, newdata = data.frame(x = right_prev)))
    ) / (right_base - right_prev)

    below <- xnew < left_base
    above <- xnew > right_base
    fitted[below] <- fitted[below] + left_slope * (xnew[below] - left_base)
    fitted[above] <- fitted[above] + right_slope * (xnew[above] - right_base)
    return(fitted)
  }
  as.numeric(
    predict(
      fit_quantile,
      newdata = data.frame(x = xnew)
    )
  )
}

order_quantile_endpoints <- function(qlo, qhi) {
  list(lower = pmin(qlo, qhi), upper = pmax(qlo, qhi))
}

fit_quantile_model_hd <- function(x, y, tau) {
  x <- as.data.frame(x)
  colnames(x) <- paste0("x", seq_len(ncol(x)))

  df <- data.frame(
    y = y,
    x
  )

  quantreg::rq(
    y ~ .,
    tau = tau,
    data = df
  )
}

predict_quantile_hd <- function(fit_quantile, xnew) {
  xnew <- as.data.frame(xnew)
  colnames(xnew) <- paste0("x", seq_len(ncol(xnew)))

  as.numeric(
    predict(
      fit_quantile,
      newdata = xnew
    )
  )
}

fit_quantile_model_auto <- function(x, y, tau, model_id, basis_df = 8,
                                    design = "uniform",
                                    basis_type = "cv_fixed_ns",
                                    candidate_dfs = c(4, 6, 8, 10, 12),
                                    candidate_lambdas = c(0.01, 0.03, 0.1, 0.3, 1, 3),
                                    fold_id = NULL) {
  fit_quantile_model(
    x, y, tau,
    basis_df = basis_df,
    design = design,
    basis_type = basis_type,
    candidate_dfs = candidate_dfs,
    candidate_lambdas = candidate_lambdas,
    fold_id = fold_id
  )
}

predict_quantile_auto <- function(fit_quantile, xnew, model_id) {
  predict_quantile(fit_quantile, xnew)
}
