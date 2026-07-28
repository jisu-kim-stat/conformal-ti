# ============================================================
# Oracle check of uniform score pivotality
#
# Run:
#   Rscript scripts/check_srti_uniform_pivotality.R
#
# This script checks, for Models 1--6:
#
#   1. whether the oracle score discrepancy decreases with n;
#   2. whether the bootstrap eta_hat upper-bounds the oracle
#      discrepancy;
#   3. how the behavior differs between pivotal and
#      non-pivotal DGPs.
# ============================================================


source("R/sim/base_mean.R")
source("R/sim/data_generate_alt.R")
source("R/sim/fit_srti_scoreboot.R")


# ------------------------------------------------------------
# 1. Simulation settings
# ------------------------------------------------------------

model_ids <- 1:6

design <- "uniform"

n_values <- c(
  200L,
  500L,
  1000L,
  2000L
)

M <- 50L

B <- 200L
n_grid <- 81L
n_conditional <- 500L

rho_boot <- 0.05
t_grid_length <- 201L

base_seed <- 20260723L

output_directory <-
  "results/sim/srti_score_pivotality"

dir.create(
  output_directory,
  recursive = TRUE,
  showWarnings = FALSE
)


# ------------------------------------------------------------
# 2. Oracle conditional score discrepancy
#
# The fitted score is held fixed:
#
#   S_n(x,y)
#     = |y - mu_hat(x)| / sigma_hat(x).
#
# Conditional Y draws come from the known simulation DGP.
# ------------------------------------------------------------

estimate_oracle_score_delta <- function(
    score_fit,
    model_id,
    x_grid,
    n_conditional = 500L,
    design_weights = NULL,
    t_grid_length = 201L
) {
  prediction <- score_fit$predict(
    x_grid
  )

  n_grid <- length(x_grid)

  score_matrix <- matrix(
    NA_real_,
    nrow = n_grid,
    ncol = n_conditional
  )

  for (j in seq_len(n_grid)) {
    x_repeated <- rep(
      x_grid[j],
      n_conditional
    )

    y_conditional <- generate_y_given_x(
      model_id = model_id,
      x = x_repeated
    )

    score_matrix[j, ] <-
      abs(
        y_conditional -
          prediction$mu[j]
      ) /
      prediction$sigma[j]
  }

  approx_score_discrepancy(
    score_matrix = score_matrix,
    design_weights = design_weights,
    t_grid_length = t_grid_length
  )
}


# ------------------------------------------------------------
# 3. One replication
# ------------------------------------------------------------

run_one_pivotality_replication <- function(
    model_id,
    n,
    replication,
    design,
    B,
    n_grid,
    n_conditional,
    rho_boot,
    t_grid_length,
    base_seed
) {
  seed_r <-
    base_seed +
    1000000L * model_id +
    1000L * n +
    replication

  set.seed(seed_r)

  data_fit <- generate_data(
    model_id = model_id,
    n = n,
    design = design
  )

  evaluation_data <- generate_eval_data(
    model_id = model_id,
    n = n_grid,
    design = design
  )

  x_grid <- evaluation_data$x

  # Equal weights are appropriate because generate_eval_data()
  # uses equally spaced design quantiles.
  design_weights <- rep(
    1 / length(x_grid),
    length(x_grid)
  )

  # ----------------------------------------------------------
  # Training-only fit
  #
  # We use the full generated dataset as the training sample
  # here because the purpose is to study Delta_n conditional
  # on the fitted score.
  # ----------------------------------------------------------

  eta_result <- estimate_scoreboot_eta(
    x_train = data_fit$x,
    y_train = data_fit$y,
    x_grid = x_grid,
    B = B,
    n_conditional = n_conditional,
    rho_boot = rho_boot,
    design_weights = design_weights,
    t_grid_length = t_grid_length,
    min_scale = 0.05,
    seed = seed_r + 100000L,
    verbose = FALSE
  )

  set.seed(
    seed_r + 200000L
  )

  oracle_delta <- estimate_oracle_score_delta(
    score_fit = eta_result$original_fit,
    model_id = model_id,
    x_grid = x_grid,
    n_conditional = n_conditional,
    design_weights = design_weights,
    t_grid_length = t_grid_length
  )

  data.frame(
    model = model_id,
    design = design,
    n = n,
    replication = replication,

    oracle_delta = oracle_delta,
    eta_hat = eta_result$eta_hat,

    eta_covers_oracle = as.integer(
      oracle_delta <= eta_result$eta_hat
    ),

    eta_minus_delta =
      eta_result$eta_hat -
      oracle_delta,

    bootstrap_delta_mean = mean(
      eta_result$bootstrap_delta
    ),

    bootstrap_delta_median =
      stats::median(
        eta_result$bootstrap_delta
      ),

    bootstrap_delta_q95 =
      as.numeric(
        stats::quantile(
          eta_result$bootstrap_delta,
          probs = 0.95,
          type = 1
        )
      ),

    mean_df =
      eta_result$original_fit$mean_df,

    scale_df =
      eta_result$original_fit$scale_df
  )
}


# ------------------------------------------------------------
# 4. Run simulation
# ------------------------------------------------------------

all_results <- list()
result_index <- 1L

for (model_id in model_ids) {
  for (n in n_values) {
    cat(
      sprintf(
        "\n[START] model = %d, n = %d\n",
        model_id,
        n
      )
    )

    setting_results <- vector(
      "list",
      M
    )

    for (r in seq_len(M)) {
      setting_results[[r]] <-
        run_one_pivotality_replication(
          model_id = model_id,
          n = n,
          replication = r,
          design = design,
          B = B,
          n_grid = n_grid,
          n_conditional = n_conditional,
          rho_boot = rho_boot,
          t_grid_length = t_grid_length,
          base_seed = base_seed
        )

      current <- do.call(
        rbind,
        setting_results[seq_len(r)]
      )

      cat(
        sprintf(
          paste0(
            "\rreplication %d/%d | ",
            "mean oracle delta = %.4f | ",
            "mean eta = %.4f | ",
            "coverage = %.3f"
          ),
          r,
          M,
          mean(current$oracle_delta),
          mean(current$eta_hat),
          mean(current$eta_covers_oracle)
        )
      )

      flush.console()
    }

    cat("\n")

    setting_df <- do.call(
      rbind,
      setting_results
    )

    all_results[[result_index]] <-
      setting_df

    result_index <- result_index + 1L

    saveRDS(
      setting_df,
      file = file.path(
        output_directory,
        sprintf(
          "model%d_n%d.rds",
          model_id,
          n
        )
      )
    )
  }
}


# ------------------------------------------------------------
# 5. Combine raw results
# ------------------------------------------------------------

results_df <- do.call(
  rbind,
  all_results
)

utils::write.csv(
  results_df,
  file = file.path(
    output_directory,
    "uniform_pivotality_raw.csv"
  ),
  row.names = FALSE
)


# ------------------------------------------------------------
# 6. Summarize by model and n
# ------------------------------------------------------------

split_results <- split(
  results_df,
  list(
    results_df$model,
    results_df$n
  ),
  drop = TRUE
)

summary_list <- lapply(
  split_results,
  function(setting_df) {
    data.frame(
      model = setting_df$model[1L],
      design = setting_df$design[1L],
      n = setting_df$n[1L],
      M = nrow(setting_df),

      mean_oracle_delta =
        mean(setting_df$oracle_delta),

      median_oracle_delta =
        stats::median(
          setting_df$oracle_delta
        ),

      oracle_delta_q90 =
        as.numeric(
          stats::quantile(
            setting_df$oracle_delta,
            probs = 0.90
          )
        ),

      oracle_delta_q95 =
        as.numeric(
          stats::quantile(
            setting_df$oracle_delta,
            probs = 0.95
          )
        ),

      mean_eta_hat =
        mean(setting_df$eta_hat),

      median_eta_hat =
        stats::median(
          setting_df$eta_hat
        ),

      eta_coverage =
        mean(
          setting_df$eta_covers_oracle
        ),

      mean_eta_minus_delta =
        mean(
          setting_df$eta_minus_delta
        )
    )
  }
)

summary_df <- do.call(
  rbind,
  summary_list
)

summary_df <- summary_df[
  order(
    summary_df$model,
    summary_df$n
  ),
]

utils::write.csv(
  summary_df,
  file = file.path(
    output_directory,
    "uniform_pivotality_summary.csv"
  ),
  row.names = FALSE
)

print(
  summary_df,
  row.names = FALSE
)


# ------------------------------------------------------------
# 7. Plot mean oracle discrepancy against n
# ------------------------------------------------------------

figure_directory <- "fig/sim"

dir.create(
  figure_directory,
  recursive = TRUE,
  showWarnings = FALSE
)

grDevices::png(
  filename = file.path(
    figure_directory,
    "srti_oracle_uniform_pivotality.png"
  ),
  width = 1400,
  height = 1000,
  res = 150
)

graphics::plot(
  range(n_values),
  range(
    summary_df$mean_oracle_delta,
    finite = TRUE
  ),
  type = "n",
  log = "x",
  xlab = "Training sample size n",
  ylab = "Mean oracle score discrepancy",
  main =
    "Uniform Score-Pivotality Diagnostic"
)

for (model_id in model_ids) {
  model_summary <- summary_df[
    summary_df$model == model_id,
  ]

  graphics::lines(
    model_summary$n,
    model_summary$mean_oracle_delta,
    type = "b",
    pch = model_id,
    lty = model_id
  )
}

graphics::legend(
  "topright",
  legend = paste(
    "Model",
    model_ids
  ),
  pch = model_ids,
  lty = model_ids,
  bty = "n"
)

grDevices::dev.off()


# ------------------------------------------------------------
# 8. Plot bootstrap eta coverage
# ------------------------------------------------------------

grDevices::png(
  filename = file.path(
    figure_directory,
    "srti_scoreboot_eta_coverage.png"
  ),
  width = 1400,
  height = 1000,
  res = 150
)

graphics::plot(
  range(n_values),
  c(0, 1),
  type = "n",
  log = "x",
  xlab = "Training sample size n",
  ylab =
    "P(oracle discrepancy <= bootstrap eta)",
  main =
    "Bootstrap Slack Coverage Diagnostic"
)

graphics::abline(
  h = 1 - rho_boot,
  lty = 2
)

for (model_id in model_ids) {
  model_summary <- summary_df[
    summary_df$model == model_id,
  ]

  graphics::lines(
    model_summary$n,
    model_summary$eta_coverage,
    type = "b",
    pch = model_id,
    lty = model_id
  )
}

graphics::legend(
  "bottomright",
  legend = paste(
    "Model",
    model_ids
  ),
  pch = model_ids,
  lty = model_ids,
  bty = "n"
)

grDevices::dev.off()


cat("\nSaved results:\n")
cat(
  file.path(
    output_directory,
    "uniform_pivotality_raw.csv"
  ),
  "\n"
)

cat(
  file.path(
    output_directory,
    "uniform_pivotality_summary.csv"
  ),
  "\n"
)

cat("\nSaved figures:\n")
cat(
  "fig/sim/srti_oracle_uniform_pivotality.png\n"
)
cat(
  "fig/sim/srti_scoreboot_eta_coverage.png\n"
)