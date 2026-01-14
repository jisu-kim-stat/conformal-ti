# scripts/run_simulation_grid.R
# ---------------------------
# Always run from project root
# ---------------------------
cat("Working directory:", getwd(), "\n")

# ---- packages ----
source("R/packages.R")

# ---- core ----
source("R/sim/base_mean.R")
source("R/sim/data_generate.R")
source("R/sim/truth_content.R")
source("R/sim/fit_mean_var.R")

## ---- ours ----
source("R/sim/lambda_hoeffding.R")
source("R/sim/intervals_ours.R")
source("R/sim/one_replication_ours.R")

## ---- gy ----
source("R/sim/gy_utils.R")
source("R/sim/one_replication_gy.R")

## ---- run setting ----
source("R/sim/run_one_setting.R")     # <- (x별 coverage/mean_width/na_proportion 요약을 반환)
source("R/utils/save_results.R")      # <- save_pointwise_csv()


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
sample_sizes <- c(250, 500, 1000)
models <- 1:6

c <- 0.90          # content level
alpha <- 0.05      # confidence error
M <- 50            # number of replications

# ---------------------------
# Parallel
# ---------------------------
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
registerDoRNG(123)


clusterEvalQ(cl, {
  src <- function(p) source(p, local = .GlobalEnv)

  src('R/packages.R')

  src('R/sim/base_mean.R')
  src('R/sim/data_generate.R')
  src('R/sim/truth_content.R')
  src('R/sim/lambda_hoeffding.R')
  src('R/sim/fit_mean_var.R')
  src('R/sim/intervals_ours.R')

  src('R/sim/one_replication_ours.R')
  src('R/sim/one_replication_gy.R')
  src('R/sim/run_one_setting.R')
  src('R/sim/gy_utils.R')

  NULL
})



# ---------------------------
# Run (grid)
# ---------------------------
all_df <- list()

all_df <- list()

models <- 1:6   # data_generate.R에 정의된 모델 개수로

for (model_id in models) {
  for (n in sample_sizes) {
    cat("[START] model:", model_id, "n:", n, "\n")

    df_one <- run_one_setting(
      model_id = model_id,
      n        = n,
      M        = M,
      c        = c,
      alpha    = alpha
    )

    key <- paste0("Model_", model_id, "_n_", n)
    all_df[[key]] <- df_one
  }
}


stopCluster(cl)

# ---------------------------
# Save
# ---------------------------
pointwise_df <- bind_rows(all_df) %>%
  dplyr::select(x, coverage, mean_width, na_proportion, model, n, Method)

out_path <- "results/sim/models/pointwise_df_ours_vs_gy.csv"
save_pointwise_csv(pointwise_df, out_path)

cat("Saved to:", out_path, "\n")