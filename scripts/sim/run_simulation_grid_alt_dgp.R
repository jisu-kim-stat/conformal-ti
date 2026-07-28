# scripts/run_simulation_grid_alt_dgp.R
# ---------------------------
# Always run from project root
# ---------------------------
cat("Working directory:", getwd(), "\n")

# ---- packages ----
source("R/packages.R")

# ---- core ----
source("R/sim/base_mean.R")

# Alternative DGPs
source("R/sim/data_generate_alt.R")
source("R/sim/truth_content_alt.R")

# ---- methods ----
source("R/sim/fit_srti.R")
source("R/sim/fit_cqr.R")
source("R/sim/lambda_hoeffding.R")
source("R/sim/pti_utils.R")
source("R/sim/guo_young_ti.R")
source("R/sim/one_replication.R")

# ---- run setting ----
source("R/sim/run_one_setting.R")
source("R/utils/save_results.R")

suppressPackageStartupMessages({
  library(parallel)
  library(doParallel)
  library(doRNG)
  library(foreach)
  library(dplyr)
})

# ---------------------------
# Simulation setup
# ---------------------------
n_cal_vec <- c(200, 500, 1000)
n_test <- 1000
models <- 1:6

content_level <- 0.90
alpha <- 0.05
epsilon_grid <- c(0, 0.01, 0.02, 0.03, 0.05)

M <- 1000

design_vec <- c("uniform", "normal")

format_elapsed <- function(seconds) {
  seconds <- max(0, as.numeric(seconds))
  days <- floor(seconds / 86400)
  hours <- floor((seconds %% 86400) / 3600)
  minutes <- floor((seconds %% 3600) / 60)
  secs <- floor(seconds %% 60)

  if (days > 0) {
    sprintf("%dd %02d:%02d:%02d", days, hours, minutes, secs)
  } else {
    sprintf("%02d:%02d:%02d", hours, minutes, secs)
  }
}

# ---------------------------
# Parallel
# ---------------------------
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
registerDoRNG(123)

clusterEvalQ(cl, {
  src <- function(p) source(p, local = .GlobalEnv)

  src("R/packages.R")
  src("R/sim/base_mean.R")

  # Alternative DGPs
  src("R/sim/data_generate_alt.R")
  src("R/sim/truth_content_alt.R")

  src("R/sim/fit_srti.R")
  src("R/sim/fit_cqr.R")
  src("R/sim/lambda_hoeffding.R")
  src("R/sim/pti_utils.R")
  src("R/sim/guo_young_ti.R")
  src("R/sim/one_replication.R")
  src("R/sim/run_one_setting.R")

  NULL
})

# ---------------------------
# Run grid
# ---------------------------
all_pointwise <- list()
all_marginal  <- list()
all_px_good   <- list()

total_settings <- length(design_vec) * length(models) * length(n_cal_vec)
setting_index <- 0L
simulation_started_at <- Sys.time()

cat(
  "[SIMULATION START]",
  format(simulation_started_at, "%Y-%m-%d %H:%M:%S"),
  "| settings:", total_settings,
  "| workers:", n_cores,
  "\n"
)

for (design in design_vec) {
  for (model_id in models) {
    for (n_cal in n_cal_vec) {

      n_train <- n_cal
      setting_index <- setting_index + 1L
      setting_started_at <- Sys.time()

      cat(
        "\n[SETTING START]", paste0(setting_index, "/", total_settings),
        "| design:", design,
        "model:", model_id,
        "n_train:", n_train,
        "n_cal:", n_cal,
        "n_test:", n_test,
        "| time:", format(setting_started_at, "%Y-%m-%d %H:%M:%S"),
        "\n"
      )
      flush.console()

      res_one <- run_one_setting(
        model_id = model_id,
        n_train  = n_train,
        n_cal    = n_cal,
        n_test   = n_test,
        M        = M,
        content  = content_level,
        alpha    = alpha,
        epsilon_grid = epsilon_grid,
        design = design,
        n_bins = 50
      )

      key <- paste0(
        "Design_", design,
        "_Model_", model_id,
        "_ntrain_", n_train,
        "_ncal_", n_cal,
        "_ntest_", n_test
      )

      all_pointwise[[key]] <- res_one$pointwise
      all_marginal[[key]]  <- res_one$marginal
      all_px_good[[key]]   <- res_one$px_good

      setting_elapsed <- as.numeric(
        difftime(Sys.time(), setting_started_at, units = "secs")
      )
      total_elapsed <- as.numeric(
        difftime(Sys.time(), simulation_started_at, units = "secs")
      )
      average_setting <- total_elapsed / setting_index
      remaining_seconds <- average_setting *
        (total_settings - setting_index)
      estimated_finish <- Sys.time() + remaining_seconds

      cat(
        "[SETTING DONE] ", setting_index, "/", total_settings,
        " | setting ", format_elapsed(setting_elapsed),
        " | total ", format_elapsed(total_elapsed),
        " | remaining ~", format_elapsed(remaining_seconds),
        " | ETA ", format(estimated_finish, "%Y-%m-%d %H:%M:%S"),
        "\n",
        sep = ""
      )
      flush.console()
    }
  }
}

stopCluster(cl)

# ---------------------------
# Save
# ---------------------------
pointwise_df <- dplyr::bind_rows(all_pointwise)
marginal_df  <- dplyr::bind_rows(all_marginal)
px_good_df   <- dplyr::bind_rows(all_px_good)

pointwise_path <- "results/sim/models/pointwise_success_5methods_alt_dgp_design_uniform_normal.csv"
marginal_path  <- "results/sim/models/marginal_pac_5methods_alt_dgp_design_uniform_normal.csv"
px_good_path   <- "results/sim/models/px_good_proportion_5methods_alt_dgp_design_uniform_normal.csv"

readr::write_csv(pointwise_df, pointwise_path)
readr::write_csv(marginal_df, marginal_path)
readr::write_csv(px_good_df, px_good_path)

cat("Saved pointwise to:", pointwise_path, "\n")
cat("Saved marginal to:", marginal_path, "\n")
cat("Saved PX-good to:", px_good_path, "\n")
cat(
  "[SIMULATION DONE] total elapsed:",
  format_elapsed(difftime(
    Sys.time(),
    simulation_started_at,
    units = "secs"
  )),
  "\n"
)
