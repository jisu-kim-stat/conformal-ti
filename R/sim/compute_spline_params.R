# R/sim/compute_spline_params.R
# ------------------------------------------------------------
# Backward-compatible wrapper around the exact smoothing-spline
# reconstruction in guo_young_ti.R.
#
# The old implementation switched to leverage-based trace approximations
# when n > threshold. That approximation is not Guo and Young's method,
# so threshold is retained only for call compatibility and is ignored.
# ------------------------------------------------------------

compute_spline_params <- function(x, y, spar = NULL, threshold = NULL) {
  model <- gy_fit_smoothing_spline(
    x = x,
    y = y,
    spar = spar
  )
  prediction <- gy_predict_smoother(model, x)

  list(
    f_hat = prediction$fitted,
    norm_lx = prediction$ell_norm,
    sigma_hat = model$sigma_hat,
    nu = model$nu,
    approx = FALSE,
    trace_a = model$trace_a,
    trace_a2 = model$trace_a2,
    fit_error = model$fit_error
  )
}
