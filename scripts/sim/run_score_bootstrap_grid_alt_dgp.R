# scripts/run_score_bootstrap_grid_alt_dgp.R
#
# Run from the project root:
#
#   Rscript scripts/run_score_bootstrap_grid_alt_dgp.R
#
# Purpose
# -------
# Compare the original score calibration and the exploratory
# score-process bootstrap buffer for:
#
#   SR-TI, ASR-TI, CQR-TI
#
# under the six alternative DGPs and the uniform/normal designs.
#
# IMPORTANT:
# The bootstrap discrepancy uses fixed X-bins and is therefore a
# groupwise proxy for the pointwise quantity
#
#   sup_x sup_t |F_x(t) - F(t)|.
#
# It is not claimed to provide exact pointwise-x simultaneous validity.
#
# Sample split
# ------------
# D_tr  : fit the score model
# D_cal : calibrate the score cutoff
# D_piv : independently estimate/bootstrap the score discrepancy
#
# D_piv is an additional sample with n_piv = n_cal.  The Original and
# Bootstrap variants share exactly the same D_tr and D_cal, so their
# paired width difference isolates the effect of the bootstrap buffer.


cat("Working directory:", getwd(), "\n")


# ============================================================
# 1. Existing project code
# ============================================================

source("R/packages.R")
source("R/sim/base_mean.R")
source("R/sim/data_generate_alt.R")
source("R/sim/truth_content_alt.R")
source("R/sim/fit_srti.R")
source("R/sim/fit_cqr.R")
source("R/sim/lambda_hoeffding.R")

suppressPackageStartupMessages({
  library(parallel)
  library(doParallel)
  library(doRNG)
  library(foreach)
  library(dplyr)
  library(tidyr)
  library(readr)
})


# ============================================================
# 2. User settings
# ============================================================

# TRUE: small pilot run over the complete model/design/n_cal grid.
# FALSE: paper-size Monte Carlo run.
quick_run <- FALSE

n_cal_vec <- c(200L, 500L, 1000L)
n_test <- 1000L
models <- 1:6
design_vec <- c("uniform", "normal")

content_level <- 0.90
alpha <- 0.05

# Five fixed bins are used because n_piv can be as small as 200.
n_boot_bins <- 5L
min_bin_size <- 10L

zeta <- 0.05
rho_rule <- "percentile"

if (quick_run) {
  M <- 10L
  B_boot <- 100L
} else {
  M <- 100L
  B_boot <- 500L
}

# Compare like with like:
# Original-Plain       vs Bootstrap-Plain
# Original-Binomial    vs Bootstrap-Binomial
# Original-Hoeffding   vs Bootstrap-Hoeffding
calibration_rules <- c("plain", "binomial", "hoeffding")

out_dir <- "results/sim/bootstrap_score_process"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

replicate_path <- file.path(
  out_dir,
  "score_bootstrap_by_rep_alt_dgp.csv"
)
pointwise_path <- file.path(
  out_dir,
  "score_bootstrap_pointwise_alt_dgp.csv"
)
summary_path <- file.path(
  out_dir,
  "score_bootstrap_summary_alt_dgp.csv"
)
width_path <- file.path(
  out_dir,
  "score_bootstrap_width_comparison_alt_dgp.csv"
)


# ============================================================
# 3. Generic score-process bootstrap helpers
# ============================================================

.safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) {
    return(NA_real_)
  }
  mean(x)
}


.safe_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) <= 1L) {
    return(NA_real_)
  }
  stats::sd(x)
}


.safe_min <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) {
    return(NA_real_)
  }
  min(x)
}


fixed_bootstrap_breaks <- function(design, n_bins) {
  if (design == "uniform") {
    return(seq(-2, 2, length.out = n_bins + 1L))
  }

  if (design == "normal") {
    # Equal-probability, data-independent bins.
    return(
      stats::qnorm(
        seq(0, 1, length.out = n_bins + 1L)
      )
    )
  }

  stop("Unknown design: ", design, call. = FALSE)
}


make_bootstrap_plan <- function(
    x,
    design,
    n_bins,
    min_group_size,
    B) {

  breaks <- fixed_bootstrap_breaks(
    design = design,
    n_bins = n_bins
  )

  group <- cut(
    x,
    breaks = breaks,
    include.lowest = TRUE,
    labels = FALSE
  )

  if (anyNA(group)) {
    stop(
      "Some pivot X values fall outside the fixed bin range.",
      call. = FALSE
    )
  }

  group <- factor(group, levels = seq_len(n_bins))
  group_indices <- split(
    seq_along(group),
    group,
    drop = FALSE
  )
  group_counts <- lengths(group_indices)

  if (any(group_counts < min_group_size)) {
    return(list(
      feasible = FALSE,
      reason = paste0(
        "Fixed-bin count below min_group_size: ",
        paste(group_counts, collapse = ",")
      ),
      group = group,
      group_counts = group_counts,
      resample_indices = NULL,
      breaks = breaks,
      B = B
    ))
  }

  # The same bootstrap indices are reused for SR, ASR, and CQR.
  # This makes their bootstrap Monte Carlo noise matched.
  resample_indices <- lapply(
    group_counts,
    function(n_g) {
      matrix(
        sample.int(
          n = n_g,
          size = B * n_g,
          replace = TRUE
        ),
        nrow = B,
        ncol = n_g
      )
    }
  )

  list(
    feasible = TRUE,
    reason = NA_character_,
    group = group,
    group_counts = group_counts,
    resample_indices = resample_indices,
    breaks = breaks,
    B = B
  )
}


empirical_cdf_distance_to_pooled <- function(
    group_score,
    pooled_sorted) {

  group_sorted <- sort(group_score)
  evaluation_points <- sort(
    unique(c(group_sorted, pooled_sorted))
  )

  F_group <- findInterval(
    evaluation_points,
    group_sorted
  ) / length(group_sorted)

  F_pooled <- findInterval(
    evaluation_points,
    pooled_sorted
  ) / length(pooled_sorted)

  max(abs(F_group - F_pooled))
}


score_discrepancy <- function(group_scores) {
  pooled_sorted <- sort(
    unlist(group_scores, use.names = FALSE)
  )

  discrepancy <- vapply(
    group_scores,
    empirical_cdf_distance_to_pooled,
    pooled_sorted = pooled_sorted,
    FUN.VALUE = numeric(1L)
  )

  list(
    D = max(discrepancy),
    by_group = discrepancy
  )
}


bootstrap_score_buffer <- function(
    score,
    plan,
    zeta = 0.05,
    rho_rule = c("percentile", "absolute", "basic", "plugin")) {

  rho_rule <- match.arg(rho_rule)

  if (length(score) != length(plan$group)) {
    stop(
      "score and pivot X must have the same length.",
      call. = FALSE
    )
  }
  if (any(!is.finite(score))) {
    stop("Pivot score contains non-finite values.", call. = FALSE)
  }

  if (!plan$feasible) {
    return(list(
      feasible = FALSE,
      reason = plan$reason,
      D_hat = NA_real_,
      rho_hat = NA_real_,
      D_star = rep(NA_real_, plan$B),
      discrepancy_by_group = rep(
        NA_real_,
        length(plan$group_counts)
      )
    ))
  }

  score_by_group <- split(
    score,
    plan$group,
    drop = FALSE
  )

  observed <- score_discrepancy(score_by_group)
  D_star <- numeric(plan$B)

  for (b in seq_len(plan$B)) {
    bootstrap_group_scores <- Map(
      function(group_score, index_matrix) {
        group_score[index_matrix[b, ]]
      },
      score_by_group,
      plan$resample_indices
    )

    D_star[b] <- score_discrepancy(
      bootstrap_group_scores
    )$D
  }

  if (rho_rule == "percentile") {
    rho_hat <- stats::quantile(
      D_star,
      probs = 1 - zeta,
      names = FALSE,
      type = 8
    )
  } else if (rho_rule == "absolute") {
    radius <- stats::quantile(
      abs(D_star - observed$D),
      probs = 1 - zeta,
      names = FALSE,
      type = 8
    )
    rho_hat <- observed$D + radius
  } else if (rho_rule == "basic") {
    lower_quantile <- stats::quantile(
      D_star,
      probs = zeta,
      names = FALSE,
      type = 8
    )
    rho_hat <- 2 * observed$D - lower_quantile
  } else {
    rho_hat <- observed$D
  }

  list(
    feasible = TRUE,
    reason = NA_character_,
    D_hat = observed$D,
    rho_hat = min(1, max(0, as.numeric(rho_hat))),
    D_star = D_star,
    discrepancy_by_group = observed$by_group
  )
}


# ============================================================
# 4. Score cutoff rules
# ============================================================

score_cutoff_by_rule <- function(
    score,
    requested_content,
    alpha,
    rule = c("plain", "binomial", "hoeffding")) {

  rule <- match.arg(rule)
  n <- length(score)

  if (
    !is.finite(requested_content) ||
      requested_content >= 1
  ) {
    return(list(
      q = NA_real_,
      rank = n + 1L,
      empirical_level = NA_real_,
      feasible = FALSE
    ))
  }

  if (rule == "plain") {
    empirical_level <- requested_content
    rank <- ceiling(n * empirical_level)
  } else if (rule == "binomial") {
    rank <- stats::qbinom(
      p = 1 - alpha,
      size = n,
      prob = requested_content
    ) + 1L
    empirical_level <- rank / n
  } else {
    # Match the existing project function find_score_cutoff().
    lambda_hoef <- sqrt(
      log(2 / alpha) / (2 * n)
    )
    empirical_level <- requested_content + lambda_hoef
    rank <- ceiling(n * empirical_level)
  }

  rank <- as.integer(max(1L, rank))

  if (
    !is.finite(empirical_level) ||
      empirical_level > 1 ||
      rank > n
  ) {
    return(list(
      q = NA_real_,
      rank = rank,
      empirical_level = empirical_level,
      feasible = FALSE
    ))
  }

  list(
    q = sort(score)[rank],
    rank = rank,
    empirical_level = empirical_level,
    feasible = TRUE
  )
}


# ============================================================
# 5. Fit the three existing score constructions
# ============================================================

fit_score_components <- function(
    train,
    pivot,
    calibration,
    x_eval,
    model_id,
    content) {

  mis <- 1 - content

  # ----------------------------------------------------------
  # Shared mean/variance fit for SR and ASR
  # ----------------------------------------------------------

  fit_mean <- fit_mean_model_auto(
    train$x,
    train$y,
    model_id
  )
  fit_var <- fit_var_model_auto(
    train$x,
    train$y,
    fit_mean,
    model_id
  )

  mu_train <- predict_mean_auto(
    fit_mean,
    train$x,
    model_id
  )
  mu_pivot <- predict_mean_auto(
    fit_mean,
    pivot$x,
    model_id
  )
  mu_cal <- predict_mean_auto(
    fit_mean,
    calibration$x,
    model_id
  )
  mu_eval <- predict_mean_auto(
    fit_mean,
    x_eval,
    model_id
  )

  sd_train <- sqrt(
    pmax(
      predict_var_auto(
        fit_var,
        train$x,
        model_id
      ),
      1e-8
    )
  )
  sd_pivot <- sqrt(
    pmax(
      predict_var_auto(
        fit_var,
        pivot$x,
        model_id
      ),
      1e-8
    )
  )
  sd_cal <- sqrt(
    pmax(
      predict_var_auto(
        fit_var,
        calibration$x,
        model_id
      ),
      1e-8
    )
  )
  sd_eval <- sqrt(
    pmax(
      predict_var_auto(
        fit_var,
        x_eval,
        model_id
      ),
      1e-8
    )
  )

  z_train <- (train$y - mu_train) / sd_train
  z_pivot <- (pivot$y - mu_pivot) / sd_pivot
  z_cal <- (calibration$y - mu_cal) / sd_cal

  sr <- list(
    method = "SR-TI",
    pivot_score = abs(z_pivot),
    calibration_score = abs(z_cal),
    mu_eval = mu_eval,
    sd_eval = sd_eval
  )

  # ----------------------------------------------------------
  # ASR score: same training shape estimate as one_replication.R
  # ----------------------------------------------------------

  shape_hat <- find_asym_shape(
    z = z_train,
    tau = mis / 2,
    eps = 1e-6
  )
  a_minus <- unname(shape_hat["a_minus"])
  a_plus <- unname(shape_hat["a_plus"])

  asr <- list(
    method = "ASR-TI",
    pivot_score = asym_residual_score(
      z = z_pivot,
      a_minus = a_minus,
      a_plus = a_plus
    ),
    calibration_score = asym_residual_score(
      z = z_cal,
      a_minus = a_minus,
      a_plus = a_plus
    ),
    mu_eval = mu_eval,
    sd_eval = sd_eval,
    a_minus = a_minus,
    a_plus = a_plus
  )

  # ----------------------------------------------------------
  # CQR score: same construction as one_replication.R
  # ----------------------------------------------------------

  tau_lo <- mis / 2
  tau_hi <- 1 - mis / 2

  fit_qlo <- fit_quantile_model_auto(
    train$x,
    train$y,
    tau_lo,
    model_id
  )
  fit_qhi <- fit_quantile_model_auto(
    train$x,
    train$y,
    tau_hi,
    model_id
  )

  qlo_pivot <- predict_quantile_auto(
    fit_qlo,
    pivot$x,
    model_id
  )
  qhi_pivot <- predict_quantile_auto(
    fit_qhi,
    pivot$x,
    model_id
  )
  qlo_cal <- predict_quantile_auto(
    fit_qlo,
    calibration$x,
    model_id
  )
  qhi_cal <- predict_quantile_auto(
    fit_qhi,
    calibration$x,
    model_id
  )
  qlo_eval <- predict_quantile_auto(
    fit_qlo,
    x_eval,
    model_id
  )
  qhi_eval <- predict_quantile_auto(
    fit_qhi,
    x_eval,
    model_id
  )

  cqr <- list(
    method = "CQR-TI",
    pivot_score = pmax(
      qlo_pivot - pivot$y,
      pivot$y - qhi_pivot
    ),
    calibration_score = pmax(
      qlo_cal - calibration$y,
      calibration$y - qhi_cal
    ),
    qlo_eval = qlo_eval,
    qhi_eval = qhi_eval
  )

  list(
    "SR-TI" = sr,
    "ASR-TI" = asr,
    "CQR-TI" = cqr
  )
}


interval_from_component <- function(component, q) {
  if (!is.finite(q)) {
    n_eval <- if (!is.null(component$mu_eval)) {
      length(component$mu_eval)
    } else {
      length(component$qlo_eval)
    }

    return(list(
      lower = rep(NA_real_, n_eval),
      upper = rep(NA_real_, n_eval)
    ))
  }

  if (component$method == "SR-TI") {
    return(list(
      lower = component$mu_eval -
        q * component$sd_eval,
      upper = component$mu_eval +
        q * component$sd_eval
    ))
  }

  if (component$method == "ASR-TI") {
    return(list(
      lower = component$mu_eval -
        q * component$a_minus * component$sd_eval,
      upper = component$mu_eval +
        q * component$a_plus * component$sd_eval
    ))
  }

  if (component$method == "CQR-TI") {
    return(list(
      lower = component$qlo_eval - q,
      upper = component$qhi_eval + q
    ))
  }

  stop(
    "Unknown score component: ",
    component$method,
    call. = FALSE
  )
}


# ============================================================
# 6. One matched Monte Carlo replication
# ============================================================

one_replication_score_bootstrap <- function(
    rep_id,
    model_id,
    n_train,
    n_piv,
    n_cal,
    n_test,
    content,
    alpha,
    design,
    n_boot_bins,
    min_bin_size,
    B_boot,
    zeta,
    rho_rule,
    calibration_rules) {

  # Match the original one_replication.R baseline data:
  # train, calibration, and deterministic evaluation grid are generated
  # first.  The additional pivot sample is generated afterwards.
  set.seed(rep_id)

  train <- generate_data(
    model_id,
    n_train,
    design = design
  )
  calibration <- generate_data(
    model_id,
    n_cal,
    design = design
  )
  evaluation <- generate_eval_data(
    model_id,
    n_test,
    design = design
  )
  pivot <- generate_data(
    model_id,
    n_piv,
    design = design
  )

  bootstrap_plan <- make_bootstrap_plan(
    x = pivot$x,
    design = design,
    n_bins = n_boot_bins,
    min_group_size = min_bin_size,
    B = B_boot
  )

  components <- fit_score_components(
    train = train,
    pivot = pivot,
    calibration = calibration,
    x_eval = evaluation$x,
    model_id = model_id,
    content = content
  )

  output <- vector(
    "list",
    length(components) *
      length(calibration_rules) *
      2L
  )
  output_index <- 1L

  for (method_name in names(components)) {
    component <- components[[method_name]]

    bootstrap_fit <- bootstrap_score_buffer(
      score = component$pivot_score,
      plan = bootstrap_plan,
      zeta = zeta,
      rho_rule = rho_rule
    )

    for (calibration_rule in calibration_rules) {
      for (buffer_name in c("Original", "Bootstrap")) {
        rho_used <- if (buffer_name == "Original") {
          0
        } else {
          bootstrap_fit$rho_hat
        }

        requested_content <- content + rho_used

        cutoff <- score_cutoff_by_rule(
          score = component$calibration_score,
          requested_content = requested_content,
          alpha = alpha,
          rule = calibration_rule
        )

        feasible <- isTRUE(bootstrap_fit$feasible) ||
          buffer_name == "Original"
        feasible <- feasible && cutoff$feasible

        if (!feasible) {
          cutoff$q <- NA_real_
        }

        interval <- interval_from_component(
          component = component,
          q = cutoff$q
        )

        conditional_content <- content_function(
          model_id = model_id,
          lower = interval$lower,
          upper = interval$upper,
          x = evaluation$x
        )

        output[[output_index]] <- data.frame(
          rep = rep_id,
          x = evaluation$x,
          content = conditional_content,
          width = interval$upper - interval$lower,
          Method = method_name,
          Buffer = buffer_name,
          Calibration = calibration_rule,
          Variant = paste(
            buffer_name,
            calibration_rule,
            sep = "-"
          ),
          q = cutoff$q,
          rank = cutoff$rank,
          empirical_level = cutoff$empirical_level,
          D_hat = bootstrap_fit$D_hat,
          rho_hat = bootstrap_fit$rho_hat,
          rho_used = rho_used,
          requested_content = requested_content,
          feasible = feasible,
          stringsAsFactors = FALSE
        )

        output_index <- output_index + 1L
      }
    }
  }

  dplyr::bind_rows(output)
}


# ============================================================
# 7. Summaries for one setting
# ============================================================

run_one_setting_score_bootstrap <- function(
    model_id,
    n_train,
    n_piv,
    n_cal,
    n_test,
    M,
    content,
    alpha,
    design,
    n_boot_bins,
    min_bin_size,
    B_boot,
    zeta,
    rho_rule,
    calibration_rules) {

  rep_long <- foreach(
    b = seq_len(M),
    .combine = dplyr::bind_rows,
    .packages = c(
      "dplyr",
      "mgcv",
      "quantreg",
      "splines"
    ),
    .export = c(
      ".safe_mean",
      ".safe_sd",
      ".safe_min",
      "fixed_bootstrap_breaks",
      "make_bootstrap_plan",
      "empirical_cdf_distance_to_pooled",
      "score_discrepancy",
      "bootstrap_score_buffer",
      "score_cutoff_by_rule",
      "fit_score_components",
      "interval_from_component",
      "one_replication_score_bootstrap",
      "base_mean",
      "generate_data",
      "generate_eval_data",
      "content_function",
      "content_split_normal",
      "fit_mean_model",
      "fit_var_model",
      "predict_mean",
      "predict_var",
      "fit_mean_model_auto",
      "fit_var_model_auto",
      "predict_mean_auto",
      "predict_var_auto",
      "fit_quantile_model",
      "predict_quantile",
      "fit_quantile_model_auto",
      "predict_quantile_auto",
      "find_asym_shape",
      "asym_residual_score"
    )
  ) %dorng% {
    tryCatch(
      one_replication_score_bootstrap(
        rep_id = b,
        model_id = model_id,
        n_train = n_train,
        n_piv = n_piv,
        n_cal = n_cal,
        n_test = n_test,
        content = content,
        alpha = alpha,
        design = design,
        n_boot_bins = n_boot_bins,
        min_bin_size = min_bin_size,
        B_boot = B_boot,
        zeta = zeta,
        rho_rule = rho_rule,
        calibration_rules = calibration_rules
      ),
      error = function(e) {
        stop(
          paste0(
            "\n[BOOTSTRAP ERROR]\n",
            "model: ", model_id, "\n",
            "design: ", design, "\n",
            "n_train: ", n_train, "\n",
            "n_piv: ", n_piv, "\n",
            "n_cal: ", n_cal, "\n",
            "rep: ", b, "\n",
            "message: ", conditionMessage(e)
          ),
          call. = FALSE
        )
      }
    )
  }

  by_rep <- rep_long %>%
    dplyr::group_by(
      rep,
      Method,
      Buffer,
      Calibration,
      Variant
    ) %>%
    dplyr::summarise(
      feasible = all(feasible),
      q = dplyr::first(q),
      rank = dplyr::first(rank),
      empirical_level = dplyr::first(empirical_level),
      D_hat = dplyr::first(D_hat),
      rho_hat = dplyr::first(rho_hat),
      rho_used = dplyr::first(rho_used),
      requested_content = dplyr::first(
        requested_content
      ),
      marginal_content = .safe_mean(content),
      min_content = .safe_min(content),
      average_width = .safe_mean(width),
      marginal_success = ifelse(
        is.finite(marginal_content),
        as.integer(
          marginal_content >= .env$content
        ),
        NA_integer_
      ),
      simultaneous_success = ifelse(
        is.finite(min_content),
        as.integer(
          min_content >= .env$content
        ),
        NA_integer_
      ),
      .groups = "drop"
    )

  pointwise <- rep_long %>%
    dplyr::group_by(
      x,
      Method,
      Buffer,
      Calibration,
      Variant
    ) %>%
    dplyr::summarise(
      n_rep_available = sum(is.finite(content)),
      mean_content = .safe_mean(content),
      pointwise_success = .safe_mean(
        as.numeric(
          .data$content >= .env$content
        )
      ),
      mean_width = .safe_mean(width),
      .groups = "drop"
    )

  setting_summary <- by_rep %>%
    dplyr::group_by(
      Method,
      Buffer,
      Calibration,
      Variant
    ) %>%
    dplyr::summarise(
      n_rep = dplyr::n(),
      feasible_rate = mean(feasible),
      average_width_mean = .safe_mean(average_width),
      average_width_sd = .safe_sd(average_width),
      marginal_content_mean = .safe_mean(
        marginal_content
      ),
      min_content_mean = .safe_mean(min_content),
      marginal_pac_success = .safe_mean(
        marginal_success
      ),
      simultaneous_success = .safe_mean(
        simultaneous_success
      ),
      q_mean = .safe_mean(q),
      D_hat_mean = .safe_mean(D_hat),
      rho_hat_mean = .safe_mean(rho_hat),
      requested_content_mean = .safe_mean(
        requested_content
      ),
      .groups = "drop"
    )

  original_rep <- by_rep %>%
    dplyr::filter(Buffer == "Original") %>%
    dplyr::select(
      rep,
      Method,
      Calibration,
      original_feasible = feasible,
      original_width = average_width,
      original_min_content = min_content
    )

  bootstrap_rep <- by_rep %>%
    dplyr::filter(Buffer == "Bootstrap") %>%
    dplyr::select(
      rep,
      Method,
      Calibration,
      bootstrap_feasible = feasible,
      bootstrap_width = average_width,
      bootstrap_min_content = min_content,
      D_hat,
      rho_hat
    )

  width_comparison <- dplyr::inner_join(
    original_rep,
    bootstrap_rep,
    by = c("rep", "Method", "Calibration")
  ) %>%
    dplyr::mutate(
      width_difference =
        bootstrap_width - original_width,
      width_ratio =
        bootstrap_width / original_width,
      min_content_difference =
        bootstrap_min_content -
        original_min_content
    )

  metadata <- list(
    model = model_id,
    design = design,
    n_train = n_train,
    n_piv = n_piv,
    n_cal = n_cal,
    n_test = n_test,
    M = M,
    B_boot = B_boot,
    n_boot_bins = n_boot_bins,
    content_target = content,
    alpha = alpha,
    zeta = zeta
  )

  add_metadata <- function(data) {
    for (name in names(metadata)) {
      data[[name]] <- metadata[[name]]
    }
    data
  }

  list(
    by_rep = add_metadata(by_rep),
    pointwise = add_metadata(pointwise),
    summary = add_metadata(setting_summary),
    width_comparison = add_metadata(
      width_comparison
    )
  )
}


# ============================================================
# 8. Full grid
# ============================================================

main <- function() {
  n_cores <- max(
    1L,
    parallel::detectCores() - 1L
  )

  cl <- parallel::makeCluster(n_cores)
  on.exit(
    parallel::stopCluster(cl),
    add = TRUE
  )

  doParallel::registerDoParallel(cl)
  doRNG::registerDoRNG(123)

  parallel::clusterEvalQ(cl, {
    src <- function(path) {
      source(path, local = .GlobalEnv)
    }

    src("R/packages.R")
    src("R/sim/base_mean.R")
    src("R/sim/data_generate_alt.R")
    src("R/sim/truth_content_alt.R")
    src("R/sim/fit_srti.R")
    src("R/sim/fit_cqr.R")
    src("R/sim/lambda_hoeffding.R")

    NULL
  })

  all_by_rep <- list()
  all_pointwise <- list()
  all_summary <- list()
  all_width <- list()

  result_index <- 1L

  cat(
    "Run mode:",
    ifelse(quick_run, "QUICK PILOT", "FULL"),
    "| M:", M,
    "| B_boot:", B_boot,
    "| cores:", n_cores,
    "\n"
  )

  for (design in design_vec) {
    for (model_id in models) {
      for (n_cal in n_cal_vec) {
        n_train <- n_cal
        n_piv <- n_cal

        cat(
          "[START]",
          "design:", design,
          "model:", model_id,
          "n_train:", n_train,
          "n_piv:", n_piv,
          "n_cal:", n_cal,
          "n_test:", n_test,
          "\n"
        )

        setting_result <- run_one_setting_score_bootstrap(
          model_id = model_id,
          n_train = n_train,
          n_piv = n_piv,
          n_cal = n_cal,
          n_test = n_test,
          M = M,
          content = content_level,
          alpha = alpha,
          design = design,
          n_boot_bins = n_boot_bins,
          min_bin_size = min_bin_size,
          B_boot = B_boot,
          zeta = zeta,
          rho_rule = rho_rule,
          calibration_rules = calibration_rules
        )

        all_by_rep[[result_index]] <-
          setting_result$by_rep
        all_pointwise[[result_index]] <-
          setting_result$pointwise
        all_summary[[result_index]] <-
          setting_result$summary
        all_width[[result_index]] <-
          setting_result$width_comparison

        # Checkpoint after every setting.
        readr::write_csv(
          dplyr::bind_rows(all_by_rep),
          replicate_path
        )
        readr::write_csv(
          dplyr::bind_rows(all_pointwise),
          pointwise_path
        )
        readr::write_csv(
          dplyr::bind_rows(all_summary),
          summary_path
        )
        readr::write_csv(
          dplyr::bind_rows(all_width),
          width_path
        )

        result_index <- result_index + 1L

        cat(
          "[DONE]",
          "design:", design,
          "model:", model_id,
          "n_cal:", n_cal,
          "\n"
        )
      }
    }
  }

  final_summary <- dplyr::bind_rows(all_summary)
  final_width <- dplyr::bind_rows(all_width)

  cat("\nSaved replication results to:", replicate_path, "\n")
  cat("Saved pointwise results to:", pointwise_path, "\n")
  cat("Saved setting summary to:", summary_path, "\n")
  cat("Saved paired width results to:", width_path, "\n\n")

  cat("Width comparison preview:\n")
  print(
    final_width %>%
      dplyr::group_by(
        model,
        design,
        n_cal,
        Method,
        Calibration
      ) %>%
      dplyr::summarise(
        feasible_rate = mean(
          bootstrap_feasible
        ),
        original_width = .safe_mean(
          original_width
        ),
        bootstrap_width = .safe_mean(
          bootstrap_width
        ),
        width_ratio = .safe_mean(width_ratio),
        rho_hat = .safe_mean(rho_hat),
        .groups = "drop"
      ) %>%
      dplyr::arrange(
        design,
        model,
        n_cal,
        Method,
        Calibration
      ),
    n = 100
  )

  invisible(
    list(
      summary = final_summary,
      width = final_width
    )
  )
}


main()
