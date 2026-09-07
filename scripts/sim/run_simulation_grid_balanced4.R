# Main four-DGP simulation suite.
# Run from the project root:
# Rscript scripts/sim/run_simulation_grid_balanced4.R --reps=1000 --cores=4

source("R/packages.R")
source("R/sim/base_mean.R")
source("R/sim/data_generate_balanced4.R")
source("R/sim/truth_content_balanced4.R")
source("R/sim/fit_srti.R")
source("R/sim/fit_cqr.R")
source("R/sim/lambda_hoeffding.R")
source("R/sim/pti_utils.R")
source("R/sim/guo_young_ti.R")
source("R/sim/one_replication.R")
source("R/sim/run_one_setting.R")

suppressPackageStartupMessages({
  library(parallel)
  library(doParallel)
  library(doRNG)
  library(dplyr)
  library(readr)
})

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    pieces <- strsplit(substring(arg, 3), "=", fixed = TRUE)[[1]]
    out[[pieces[1]]] <- if (length(pieces) == 2L) pieces[2] else TRUE
  }
  out
}

parse_num_vec <- function(x, default) {
  if (is.null(x)) return(default)
  as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
M <- as.integer(if (is.null(args$reps)) 1000 else args$reps)
detected_cores <- suppressWarnings(detectCores())
default_cores <- if (is.na(detected_cores)) 1L else max(1L, detected_cores - 1L)
n_cores <- as.integer(if (is.null(args$cores)) default_cores else args$cores)
n_cal_vec <- parse_num_vec(args$ncal, c(200, 500, 1000))
models <- parse_num_vec(args$models, 1:4)
design_vec <- if (is.null(args$designs)) c("normal", "uniform") else strsplit(args$designs, ",", fixed = TRUE)[[1]]
methods <- if (is.null(args$methods)) {
  c("SR-TI", "ASR-TI", "CQR-TI", "Parametric-TI", "GY-TI")
} else {
  strsplit(args$methods, ",", fixed = TRUE)[[1]]
}
cqr_basis_df <- as.integer(if (is.null(args$cqr_basis_df)) 8 else args$cqr_basis_df)
cqr_basis_type <- if (is.null(args$cqr_basis_type)) "cv_fixed_ns" else args$cqr_basis_type
cqr_df_grid <- if (is.null(args$cqr_df_grid)) c(4, 6, 8, 10, 12) else parse_num_vec(args$cqr_df_grid, integer())
cqr_cv_folds <- as.integer(if (is.null(args$cqr_cv_folds)) 5 else args$cqr_cv_folds)
tag <- if (is.null(args$tag)) paste0("M", M) else args$tag

stopifnot(M >= 1L, n_cores >= 1L, all(n_cal_vec >= 2L), cqr_basis_df >= 4L)
stopifnot(all(models %in% 1:4))
stopifnot(all(design_vec %in% c("normal", "uniform")))
stopifnot(all(methods %in% c("SR-TI", "ASR-TI", "CQR-TI", "Parametric-TI", "GY-TI")))
stopifnot(cqr_basis_type %in% c("cv_fixed_ns", "fixed_ns", "legacy_bs"))
stopifnot(all(cqr_df_grid >= 4L), cqr_cv_folds >= 2L)

content_level <- 0.90
alpha <- 0.05
epsilon_grid <- c(0, 0.01, 0.02, 0.03, 0.05)
n_test <- 1000

cl <- makeCluster(n_cores)
registerDoParallel(cl)
registerDoRNG(20260907)

clusterEvalQ(cl, {
  src <- function(path) source(path, local = .GlobalEnv)
  src("R/packages.R")
  src("R/sim/base_mean.R")
  src("R/sim/data_generate_balanced4.R")
  src("R/sim/truth_content_balanced4.R")
  src("R/sim/fit_srti.R")
  src("R/sim/fit_cqr.R")
  src("R/sim/lambda_hoeffding.R")
  src("R/sim/pti_utils.R")
  src("R/sim/guo_young_ti.R")
  src("R/sim/one_replication.R")
  src("R/sim/run_one_setting.R")
  NULL
})

all_pointwise <- list()
all_marginal <- list()
all_px_good <- list()
setting <- 0L

for (design in design_vec) for (model_id in models) for (n_cal in n_cal_vec) {
  setting <- setting + 1L
  message(sprintf(
    "[%d/%d] model=%d design=%s n_train=n_cal=%d",
    setting, length(design_vec) * length(models) * length(n_cal_vec),
    model_id, design, n_cal
  ))

  result <- run_one_setting(
    model_id = model_id,
    n_train = n_cal,
    n_cal = n_cal,
    n_test = n_test,
    M = M,
    content = content_level,
    alpha = alpha,
    epsilon_grid = epsilon_grid,
    design = design,
    n_bins = 50,
    cqr_basis_df = cqr_basis_df,
    cqr_basis_type = cqr_basis_type,
    cqr_df_grid = cqr_df_grid,
    cqr_cv_folds = cqr_cv_folds,
    methods = methods,
    calibration_rule = "hoeffding"
  )

  result$pointwise$simulation_suite <- "balanced4"
  result$marginal$simulation_suite <- "balanced4"
  result$px_good$simulation_suite <- "balanced4"
  all_pointwise[[setting]] <- result$pointwise
  all_marginal[[setting]] <- result$marginal
  all_px_good[[setting]] <- result$px_good
}

stopCluster(cl)

out_dir <- "results/sim/balanced4"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(bind_rows(all_pointwise), file.path(out_dir, paste0("pointwise_", tag, ".csv")))
write_csv(bind_rows(all_marginal), file.path(out_dir, paste0("marginal_", tag, ".csv")))
write_csv(bind_rows(all_px_good), file.path(out_dir, paste0("px_good_", tag, ".csv")))

metadata <- tibble(
  model = models,
  description = c(
    "Homoscedastic Gaussian",
    "Unit-variance heavy-tailed t3",
    "Heteroscedastic Gaussian location-scale",
    "Smooth x-dependent endpoint asymmetry (non-location-scale)"
  )[models],
  cqr_basis_df = cqr_basis_df,
  cqr_basis_type = cqr_basis_type,
  cqr_df_grid = paste(cqr_df_grid, collapse = ","),
  cqr_cv_folds = cqr_cv_folds,
  methods = paste(methods, collapse = ","),
  M = M,
  calibration_rule = "hoeffding"
)
write_csv(metadata, file.path(out_dir, paste0("metadata_", tag, ".csv")))

message("Saved balanced-four-DGP results to: ", out_dir)
