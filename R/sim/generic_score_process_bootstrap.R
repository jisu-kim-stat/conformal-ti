# ============================================================
# Generic score-process bootstrap for simultaneous calibration
#
# Purpose
# -------
# This is an exploratory, score-agnostic implementation of the idea
#
#   inf_{x in Lambda} F_x(q)
#     >= F(q) - sup_{x in Lambda, t} |F_x(t) - F(t)|.
#
# It does NOT generate pseudo-responses Y*, assign artificial signs, or
# require a normal/symmetric error distribution.  It only uses scalar
# fitted scores S_eta_hat(X, Y).
#
# For continuous X, exact pointwise-x estimation is impossible without
# additional smoothness assumptions.  This implementation therefore uses
# a fixed, data-independent partition of Lambda and targets simultaneous
# validity across the resulting groups/bins:
#
#   D_G = max_g sup_t |F_g(t) - F(t)|.
#
# Recommended three-way split
# ---------------------------
#   D_tr  : fit eta_hat and define the score;
#   D_piv : estimate/bootstrap D_G from (X_i, S_i);
#   D_cal : calibrate the final score threshold q.
#
# D_piv and D_cal should be independent conditional on D_tr.
#
# Important
# ---------
# The bootstrap estimate of the uniform discrepancy is a simulation
# device, not a general finite-sample theorem.  The validity target is
# group-simultaneous unless stronger within-bin assumptions are supplied.
# ============================================================


# ------------------------------------------------------------
# 1. Input helpers
# ------------------------------------------------------------

.assert_probability <- function(x, name, open = FALSE) {
  ok <- length(x) == 1L && is.finite(x)
  if (open) {
    ok <- ok && x > 0 && x < 1
  } else {
    ok <- ok && x >= 0 && x <= 1
  }
  if (!ok) {
    stop(name, " must be a single probability", call. = FALSE)
  }
}


.check_numeric_scores <- function(score, name, min_length = 1L) {
  if (!is.numeric(score) || length(score) < min_length) {
    stop(name, " must be a numeric vector", call. = FALSE)
  }
  if (any(!is.finite(score))) {
    stop(name, " contains NA, NaN, or infinite values", call. = FALSE)
  }
  invisible(TRUE)
}


make_fixed_bins <- function(
    x,
    breaks,
    include_lowest = TRUE,
    right = TRUE,
    outside = c("error", "drop")) {

  outside <- match.arg(outside)

  if (!is.numeric(x) || any(!is.finite(x))) {
    stop("x must be a finite numeric vector", call. = FALSE)
  }
  if (
    !is.numeric(breaks) ||
      length(breaks) < 3L ||
      anyNA(breaks) ||
      any(is.infinite(breaks[-c(1L, length(breaks))])) ||
      is.unsorted(breaks, strictly = TRUE)
  ) {
    stop(
      paste0(
        "breaks must be a strictly increasing numeric vector of length >= 3; ",
        "only the two endpoints may be infinite"
      ),
      call. = FALSE
    )
  }

  group <- cut(
    x,
    breaks = breaks,
    include.lowest = include_lowest,
    right = right,
    ordered_result = TRUE
  )

  outside_index <- which(is.na(group))
  if (outside == "error" && length(outside_index) > 0L) {
    stop(
      length(outside_index),
      " x value(s) fall outside the fixed bin range [",
      min(breaks),
      ", ",
      max(breaks),
      "]",
      call. = FALSE
    )
  }

  list(
    # Keep empty levels.  Because the bins define the target region in
    # advance, silently dropping an empty bin would make Lambda data-driven.
    group = group,
    keep = !is.na(group),
    outside_index = outside_index,
    breaks = breaks
  )
}


# ------------------------------------------------------------
# 2. Group analogue of the uniform score-CDF discrepancy
# ------------------------------------------------------------

empirical_cdf_distance <- function(x, y) {
  .check_numeric_scores(x, "x")
  .check_numeric_scores(y, "y")

  x_sorted <- sort(x)
  y_sorted <- sort(y)
  evaluation_points <- sort(unique(c(x_sorted, y_sorted)))

  Fx <- findInterval(evaluation_points, x_sorted) / length(x_sorted)
  Fy <- findInterval(evaluation_points, y_sorted) / length(y_sorted)

  max(abs(Fx - Fy))
}


estimate_group_score_discrepancy <- function(
    score,
    group,
    min_group_size = 20L) {

  .check_numeric_scores(score, "score", min_length = 2L)

  if (length(group) != length(score)) {
    stop("group and score must have the same length", call. = FALSE)
  }
  if (anyNA(group)) {
    stop("group contains missing values", call. = FALSE)
  }
  if (
    length(min_group_size) != 1L ||
      !is.finite(min_group_size) ||
      min_group_size < 2L
  ) {
    stop("min_group_size must be at least 2", call. = FALSE)
  }

  group <- if (is.factor(group)) group else as.factor(group)
  group_scores <- split(score, group, drop = FALSE)
  group_counts <- lengths(group_scores)

  if (length(group_scores) < 2L) {
    stop("At least two nonempty groups are required", call. = FALSE)
  }
  if (any(group_counts < min_group_size)) {
    bad <- names(group_counts)[group_counts < min_group_size]
    stop(
      "The following fixed groups have fewer than min_group_size observations: ",
      paste(bad, collapse = ", "),
      ". Use fewer prespecified bins or a larger pivot sample.",
      call. = FALSE
    )
  }

  discrepancy_by_group <- vapply(
    group_scores,
    empirical_cdf_distance,
    y = score,
    FUN.VALUE = numeric(1L)
  )

  list(
    D_hat = max(discrepancy_by_group),
    discrepancy_by_group = discrepancy_by_group,
    group_counts = group_counts,
    group_scores = group_scores
  )
}


# ------------------------------------------------------------
# 3. Stratified nonparametric bootstrap of the score process
# ------------------------------------------------------------

bootstrap_group_score_discrepancy <- function(
    score,
    group,
    B = 1000L,
    min_group_size = 20L,
    seed = NULL,
    verbose = interactive()) {

  if (length(B) != 1L || !is.finite(B) || B < 2L) {
    stop("B must be an integer of at least 2", call. = FALSE)
  }
  B <- as.integer(B)

  observed <- estimate_group_score_discrepancy(
    score = score,
    group = group,
    min_group_size = min_group_size
  )

  if (!is.null(seed)) {
    set.seed(seed)
  }

  group_scores <- observed$group_scores
  group_counts <- lengths(group_scores)
  D_star <- numeric(B)

  report_every <- max(1L, floor(B / 10L))

  for (b in seq_len(B)) {
    bootstrap_scores <- Map(
      function(z, n) sample(z, size = n, replace = TRUE),
      group_scores,
      group_counts
    )
    pooled_bootstrap_score <- unlist(
      bootstrap_scores,
      use.names = FALSE
    )

    discrepancy_star <- vapply(
      bootstrap_scores,
      empirical_cdf_distance,
      y = pooled_bootstrap_score,
      FUN.VALUE = numeric(1L)
    )
    D_star[b] <- max(discrepancy_star)

    if (
      isTRUE(verbose) &&
        (b %% report_every == 0L || b == B)
    ) {
      message("Score-process bootstrap: ", b, "/", B)
    }
  }

  list(
    D_hat = observed$D_hat,
    D_star = D_star,
    discrepancy_by_group = observed$discrepancy_by_group,
    group_counts = observed$group_counts
  )
}


choose_discrepancy_buffer <- function(
    D_hat,
    D_star,
    zeta = 0.05,
    rule = c("percentile", "absolute", "basic", "plugin"),
    quantile_type = 8L) {

  rule <- match.arg(rule)
  .assert_probability(zeta, "zeta", open = TRUE)
  .check_numeric_scores(D_star, "D_star", min_length = 2L)

  if (length(D_hat) != 1L || !is.finite(D_hat) || D_hat < 0) {
    stop("D_hat must be a finite nonnegative number", call. = FALSE)
  }

  if (rule == "percentile") {
    rho_hat <- stats::quantile(
      D_star,
      probs = 1 - zeta,
      names = FALSE,
      type = quantile_type
    )
  } else if (rule == "absolute") {
    radius <- stats::quantile(
      abs(D_star - D_hat),
      probs = 1 - zeta,
      names = FALSE,
      type = quantile_type
    )
    rho_hat <- D_hat + radius
  } else if (rule == "basic") {
    lower_bootstrap_quantile <- stats::quantile(
      D_star,
      probs = zeta,
      names = FALSE,
      type = quantile_type
    )
    rho_hat <- 2 * D_hat - lower_bootstrap_quantile
  } else {
    rho_hat <- D_hat
  }

  min(1, max(0, as.numeric(rho_hat)))
}


# ------------------------------------------------------------
# 4. Distribution-free calibration of the score threshold
# ------------------------------------------------------------

calibrate_score_threshold <- function(
    calibration_score,
    target_content,
    alpha = 0.05,
    method = c("hoeffding", "binomial_rank", "none")) {

  method <- match.arg(method)
  .check_numeric_scores(
    calibration_score,
    "calibration_score",
    min_length = 2L
  )
  .assert_probability(target_content, "target_content")
  .assert_probability(alpha, "alpha", open = TRUE)

  n_cal <- length(calibration_score)
  sorted_score <- sort(calibration_score)

  if (target_content >= 1) {
    return(list(
      q = Inf,
      rank = n_cal + 1L,
      requested_content = target_content,
      empirical_quantile_level = Inf,
      method = method,
      feasible = FALSE,
      reason = "The buffered target content is at least one."
    ))
  }

  if (method == "hoeffding") {
    lambda <- sqrt(log(1 / alpha) / (2 * n_cal))
    empirical_level <- target_content + lambda
    rank <- ceiling(n_cal * empirical_level)
  } else if (method == "binomial_rank") {
    # Let U_(k) be the kth order statistic after the probability
    # integral transform.  Choose the smallest rank satisfying
    #   P{U_(k) >= target_content} >= 1 - alpha.
    # This is equivalent to
    #   P{Bin(n_cal, target_content) <= k - 1} >= 1 - alpha.
    rank <- stats::qbinom(
      p = 1 - alpha,
      size = n_cal,
      prob = target_content
    ) + 1L
    empirical_level <- rank / n_cal
  } else {
    empirical_level <- target_content
    rank <- ceiling(n_cal * empirical_level)
  }

  rank <- max(1L, as.integer(rank))

  if (!is.finite(empirical_level) || rank > n_cal) {
    return(list(
      q = Inf,
      rank = rank,
      requested_content = target_content,
      empirical_quantile_level = empirical_level,
      method = method,
      feasible = FALSE,
      reason = paste0(
        "The requested confidence/content combination needs order-statistic ",
        "rank ",
        rank,
        " but n_cal = ",
        n_cal,
        "."
      )
    ))
  }

  list(
    q = sorted_score[rank],
    rank = rank,
    requested_content = target_content,
    empirical_quantile_level = empirical_level,
    method = method,
    feasible = TRUE,
    reason = NA_character_
  )
}


.threshold_comparison <- function(
    calibration_score,
    target_content,
    alpha) {

  methods <- c("none", "hoeffding", "binomial_rank")
  fits <- lapply(
    methods,
    function(method) {
      calibrate_score_threshold(
        calibration_score = calibration_score,
        target_content = target_content,
        alpha = alpha,
        method = method
      )
    }
  )

  data.frame(
    method = methods,
    q = vapply(fits, function(z) z$q, numeric(1L)),
    rank = vapply(fits, function(z) z$rank, integer(1L)),
    empirical_quantile_level = vapply(
      fits,
      function(z) z$empirical_quantile_level,
      numeric(1L)
    ),
    feasible = vapply(fits, function(z) z$feasible, logical(1L)),
    stringsAsFactors = FALSE
  )
}


# ------------------------------------------------------------
# 5. Main calibration function
# ------------------------------------------------------------

score_process_bootstrap_calibrate <- function(
    pivot_x,
    pivot_score,
    calibration_score,
    bin_breaks,
    target_content = 0.90,
    confidence = 0.95,
    B = 1000L,
    zeta = 0.05,
    rho_rule = c("percentile", "absolute", "basic", "plugin"),
    calibration_method = c("hoeffding", "binomial_rank", "none"),
    min_group_size = 20L,
    seed = NULL,
    verbose = interactive()) {

  rho_rule <- match.arg(rho_rule)
  calibration_method <- match.arg(calibration_method)
  .assert_probability(target_content, "target_content", open = TRUE)
  .assert_probability(confidence, "confidence", open = TRUE)

  .check_numeric_scores(pivot_score, "pivot_score", min_length = 2L)
  if (length(pivot_x) != length(pivot_score)) {
    stop(
      "pivot_x and pivot_score must have the same length",
      call. = FALSE
    )
  }

  bin_info <- make_fixed_bins(
    x = pivot_x,
    breaks = bin_breaks,
    outside = "error"
  )
  pivot_group <- bin_info$group

  bootstrap_fit <- bootstrap_group_score_discrepancy(
    score = pivot_score,
    group = pivot_group,
    B = B,
    min_group_size = min_group_size,
    seed = seed,
    verbose = verbose
  )

  rho_hat <- choose_discrepancy_buffer(
    D_hat = bootstrap_fit$D_hat,
    D_star = bootstrap_fit$D_star,
    zeta = zeta,
    rule = rho_rule
  )

  buffered_content <- target_content + rho_hat
  alpha <- 1 - confidence

  selected_threshold <- calibrate_score_threshold(
    calibration_score = calibration_score,
    target_content = min(buffered_content, 1),
    alpha = alpha,
    method = calibration_method
  )

  threshold_comparison <- .threshold_comparison(
    calibration_score = calibration_score,
    target_content = min(buffered_content, 1),
    alpha = alpha
  )

  bin_summary <- data.frame(
    group = names(bootstrap_fit$group_counts),
    n_pivot = as.integer(bootstrap_fit$group_counts),
    discrepancy = as.numeric(
      bootstrap_fit$discrepancy_by_group[
        names(bootstrap_fit$group_counts)
      ]
    ),
    stringsAsFactors = FALSE
  )

  result <- list(
    q = selected_threshold$q,
    feasible = selected_threshold$feasible &&
      buffered_content < 1,
    reason = if (buffered_content >= 1) {
      paste0(
        "target_content + rho_hat = ",
        signif(buffered_content, 5),
        " >= 1"
      )
    } else {
      selected_threshold$reason
    },
    target_content = target_content,
    confidence = confidence,
    alpha = alpha,
    zeta = zeta,
    # If rho_hat were a valid (1-zeta) upper bound for D_G, the simple
    # union-bound bookkeeping would give 1 - alpha - zeta.
    # Here this number is descriptive because the bootstrap upper bound
    # itself is only being used as an exploratory approximation.
    nominal_bookkeeping_confidence = max(0, confidence - zeta),
    D_hat = bootstrap_fit$D_hat,
    rho_hat = rho_hat,
    rho_rule = rho_rule,
    buffered_content = buffered_content,
    calibration_method = calibration_method,
    selected_threshold = selected_threshold,
    threshold_comparison = threshold_comparison,
    D_star = bootstrap_fit$D_star,
    bin_summary = bin_summary,
    bin_breaks = bin_breaks,
    n_pivot = length(pivot_score),
    n_calibration = length(calibration_score),
    B = as.integer(B),
    call = match.call()
  )
  class(result) <- "score_process_bootstrap"
  result
}


print.score_process_bootstrap <- function(x, ...) {
  cat("Generic score-process bootstrap calibration\n")
  cat("  target content      :", x$target_content, "\n")
  cat("  observed D_hat      :", signif(x$D_hat, 5), "\n")
  cat("  bootstrap rho_hat   :", signif(x$rho_hat, 5), "\n")
  cat("  buffered content    :", signif(x$buffered_content, 5), "\n")
  cat("  calibration method  :", x$calibration_method, "\n")
  cat(
    "  alpha/zeta bookkeeping:",
    signif(x$nominal_bookkeeping_confidence, 5),
    "(heuristic)\n"
  )
  cat("  threshold q         :", signif(x$q, 7), "\n")
  cat("  finite/feasible     :", x$feasible, "\n")
  if (!x$feasible && !is.na(x$reason)) {
    cat("  reason              :", x$reason, "\n")
  }
  invisible(x)
}


plot.score_process_bootstrap <- function(
    x,
    col_bootstrap = "grey80",
    col_observed = "#2C7FB8",
    col_buffer = "#D95F0E",
    ...) {

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)

  graphics::par(mfrow = c(1, 2), mar = c(4.2, 4.2, 2.5, 1))

  graphics::hist(
    x$D_star,
    breaks = "FD",
    col = col_bootstrap,
    border = "white",
    xlab = expression(D^"*"),
    main = "Bootstrap discrepancy"
  )
  graphics::abline(
    v = x$D_hat,
    col = col_observed,
    lwd = 2
  )
  graphics::abline(
    v = x$rho_hat,
    col = col_buffer,
    lwd = 2,
    lty = 2
  )
  graphics::legend(
    "topright",
    legend = c("observed D", "chosen rho"),
    col = c(col_observed, col_buffer),
    lwd = 2,
    lty = c(1, 2),
    bty = "n"
  )

  bar_mid <- graphics::barplot(
    x$bin_summary$discrepancy,
    names.arg = x$bin_summary$group,
    las = 2,
    col = col_observed,
    border = NA,
    ylab = "Empirical CDF discrepancy",
    main = "Discrepancy by fixed bin"
  )
  graphics::text(
    x = bar_mid,
    y = x$bin_summary$discrepancy,
    labels = paste0("n=", x$bin_summary$n_pivot),
    pos = 3,
    cex = 0.75
  )

  invisible(x)
}


# ------------------------------------------------------------
# 6. Matched comparison of several scalar scores
# ------------------------------------------------------------

.as_named_score_list <- function(x, object_name) {
  if (is.data.frame(x)) {
    x <- as.list(x)
  } else if (is.matrix(x)) {
    x <- lapply(seq_len(ncol(x)), function(j) x[, j])
    names(x) <- colnames(x)
  }

  if (!is.list(x) || length(x) < 1L) {
    stop(
      object_name,
      " must be a named list, data frame, or numeric matrix",
      call. = FALSE
    )
  }

  if (is.null(names(x)) || any(names(x) == "")) {
    names(x) <- paste0("score_", seq_along(x))
  }

  for (name in names(x)) {
    .check_numeric_scores(x[[name]], paste0(object_name, "$", name))
  }
  x
}


compare_score_process_bootstraps <- function(
    pivot_x,
    pivot_scores,
    calibration_scores,
    bin_breaks,
    target_content = 0.90,
    confidence = 0.95,
    B = 1000L,
    zeta = 0.05,
    rho_rule = "percentile",
    calibration_method = "hoeffding",
    min_group_size = 20L,
    seed = NULL,
    verbose = interactive()) {

  pivot_scores <- .as_named_score_list(
    pivot_scores,
    "pivot_scores"
  )
  calibration_scores <- .as_named_score_list(
    calibration_scores,
    "calibration_scores"
  )

  method_names <- intersect(
    names(pivot_scores),
    names(calibration_scores)
  )
  if (length(method_names) == 0L) {
    stop(
      "pivot_scores and calibration_scores have no common method names",
      call. = FALSE
    )
  }

  fits <- setNames(vector("list", length(method_names)), method_names)

  for (method_name in method_names) {
    if (isTRUE(verbose)) {
      message("Calibrating score method: ", method_name)
    }

    # Reusing the same seed produces matched stratified resampling indices
    # across score methods, which reduces irrelevant Monte Carlo variation.
    fits[[method_name]] <- score_process_bootstrap_calibrate(
      pivot_x = pivot_x,
      pivot_score = pivot_scores[[method_name]],
      calibration_score = calibration_scores[[method_name]],
      bin_breaks = bin_breaks,
      target_content = target_content,
      confidence = confidence,
      B = B,
      zeta = zeta,
      rho_rule = rho_rule,
      calibration_method = calibration_method,
      min_group_size = min_group_size,
      seed = seed,
      verbose = verbose
    )
  }

  summary <- data.frame(
    method = method_names,
    D_hat = vapply(fits, function(z) z$D_hat, numeric(1L)),
    rho_hat = vapply(fits, function(z) z$rho_hat, numeric(1L)),
    buffered_content = vapply(
      fits,
      function(z) z$buffered_content,
      numeric(1L)
    ),
    q = vapply(fits, function(z) z$q, numeric(1L)),
    feasible = vapply(fits, function(z) z$feasible, logical(1L)),
    stringsAsFactors = FALSE
  )

  list(
    fits = fits,
    summary = summary,
    matched_bootstrap_seed = seed
  )
}


# ============================================================
# Minimal usage
# ============================================================
#
# 1. Fit each score model once on D_tr.
# 2. Evaluate the fitted scalar scores on independent D_piv and D_cal:
#
# pivot_scores <- list(
#   SR  = score_sr(x_piv,  y_piv),
#   ASR = score_asr(x_piv, y_piv),
#   CQR = score_cqr(x_piv, y_piv)
# )
#
# calibration_scores <- list(
#   SR  = score_sr(x_cal,  y_cal),
#   ASR = score_asr(x_cal, y_cal),
#   CQR = score_cqr(x_cal, y_cal)
# )
#
# Fixed breaks must be chosen without looking at D_piv/D_cal.  For example,
# if the design support is known to be [0, 1]:
#
# fixed_breaks <- seq(0, 1, length.out = 11)
#
# comparison <- compare_score_process_bootstraps(
#   pivot_x = x_piv,
#   pivot_scores = pivot_scores,
#   calibration_scores = calibration_scores,
#   bin_breaks = fixed_breaks,
#   target_content = 0.90,
#   confidence = 0.95,
#   B = 1000,
#   zeta = 0.05,
#   rho_rule = "percentile",
#   calibration_method = "hoeffding",
#   min_group_size = 20,
#   seed = 20260727
# )
#
# The defaults confidence = 0.95 and zeta = 0.05 are convenient for
# diagnostics, but correspond to 0.90 under alpha + zeta bookkeeping.
# For a nominal overall 0.95 with an equal split, use
# confidence = 0.975 and zeta = 0.025.  This is still not a theorem for
# the bootstrap approximation.
#
# comparison$summary
# comparison$fits$SR$threshold_comparison
# plot(comparison$fits$SR)
#
# Construct the final score set in the usual way:
#
#   T_hat(x) = {y : S_eta_hat(x, y) <= comparison$fits$SR$q}.
#
# If "hoeffding" is infeasible, inspect
# comparison$fits$SR$threshold_comparison.  Using method = "none" can be
# informative as an exploratory diagnostic, but it does not supply the
# calibration-sample PAC correction.


# ============================================================
# Standalone one-simulation run
# ============================================================
#
# Run in VSCode terminal:
#
#   Rscript generic_score_process_bootstrap.R
#
# This block is skipped when the file is loaded with source().

.is_direct_rscript_run <- function(filename) {
  command_arguments <- commandArgs(trailingOnly = FALSE)
  file_argument <- command_arguments[
    grepl("^--file=", command_arguments)
  ]

  if (length(file_argument) != 1L) {
    return(FALSE)
  }

  executed_file <- sub("^--file=", "", file_argument)
  identical(
    basename(executed_file),
    basename(filename)
  )
}


run_one_score_process_example <- function(
    seed = 1L,
    n_pivot = 5000L,
    n_calibration = 5000L,
    B = 500L) {

  set.seed(seed)

  # ----------------------------------------------------------
  # EXAMPLE ONLY
  #
  # The noise is centered and standardized chi-square, hence
  # asymmetric and nonnormal.  The scalar score below is pivotal:
  #
  #   S(x, y) = |(y - mu(x)) / sigma(x)|.
  #
  # In the paper simulation, replace only pivot_score and
  # calibration_score by the scores fitted on D_tr.
  # ----------------------------------------------------------

  mean_function <- function(x) {
    sin(pi * x) + 0.5 * x
  }

  scale_function <- function(x) {
    0.6 + 0.4 * abs(x)
  }

  generate_data <- function(n) {
    x <- stats::runif(n, min = -1, max = 1)
    z <- (
      stats::rchisq(n, df = 3) - 3
    ) / sqrt(6)
    y <- mean_function(x) + scale_function(x) * z

    data.frame(
      x = x,
      y = y
    )
  }

  fitted_score <- function(x, y) {
    abs(
      (y - mean_function(x)) /
        scale_function(x)
    )
  }

  pivot_data <- generate_data(n_pivot)
  calibration_data <- generate_data(n_calibration)

  pivot_score <- fitted_score(
    pivot_data$x,
    pivot_data$y
  )
  calibration_score <- fitted_score(
    calibration_data$x,
    calibration_data$y
  )

  fit <- score_process_bootstrap_calibrate(
    pivot_x = pivot_data$x,
    pivot_score = pivot_score,
    calibration_score = calibration_score,
    bin_breaks = seq(-1, 1, length.out = 6L),
    target_content = 0.90,
    confidence = 0.95,
    B = B,
    zeta = 0.05,
    rho_rule = "percentile",
    calibration_method = "hoeffding",
    min_group_size = 20L,
    seed = seed + 1L,
    verbose = TRUE
  )

  print(fit)
  print(fit$threshold_comparison)

  output_summary <- data.frame(
    seed = seed,
    n_pivot = n_pivot,
    n_calibration = n_calibration,
    B = B,
    target_content = fit$target_content,
    D_hat = fit$D_hat,
    rho_hat = fit$rho_hat,
    buffered_content = fit$buffered_content,
    calibration_method = fit$calibration_method,
    q = fit$q,
    feasible = fit$feasible,
    stringsAsFactors = FALSE
  )

  utils::write.csv(
    output_summary,
    file = "generic_score_bootstrap_one_sim_summary.csv",
    row.names = FALSE
  )

  grDevices::png(
    filename = "generic_score_bootstrap_one_sim.png",
    width = 1800,
    height = 800,
    res = 150
  )
  plot(fit)
  grDevices::dev.off()

  message(
    "Saved: generic_score_bootstrap_one_sim_summary.csv"
  )
  message(
    "Saved: generic_score_bootstrap_one_sim.png"
  )

  invisible(fit)
}


if (.is_direct_rscript_run("generic_score_process_bootstrap.R")) {
  run_one_score_process_example()
}
