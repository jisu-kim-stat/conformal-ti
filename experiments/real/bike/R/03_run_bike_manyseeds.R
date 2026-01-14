# scripts/03_run_bike_manyseeds.R
# Multi-seed experiment for Bike data

library(data.table)
library(dplyr)
library(tidyr)
library(purrr)
library(gam)
library(MASS)
library(splines)
library(tibble)

# ---- source functions ----
source("experiments/real/bike/R/10_scores_lambda.R")
source("experiments/real/bike/R/20_intervals_ours.R")
source("experiments/real/bike/R/30_intervals_gy.R")
source("experiments/real/bike/R/40_intervals_normal.R")
source("experiments/real/bike/R/50_eval_conditional.R")
source("experiments/real/bike/R/01_data_bike.R")
source("experiments/real/bike/R/90_one_run.R")

# ---- load data ----
bike <- load_bike_data()

# ---- experiment config ----
seeds <- 1:50
nbins <- 20
alpha <- 0.10
delta <- 0.05

# ---- run experiments ----
all_runs <- lapply(seeds, function(s){
  cat("Running seed", s, "\n")
  one_run_bike_plot(
    seed_split = s,
    bike_df = bike,
    alpha = alpha,
    delta = delta,
    nbins = nbins
  )
})

# ---- aggregate conditional summaries ----
cond_all <- bind_rows(lapply(all_runs, function(res){
  res$cond_summary_seed %>% mutate(seed = res$seed)
}))

summary_tbl <- cond_all %>%
  group_by(method) %>%
  summarise(
    mean_prop_bins_ge90 = mean(prop_bins_ge90, na.rm = TRUE),
    sd_prop_bins_ge90   = sd(prop_bins_ge90, na.rm = TRUE),
    mean_min_bin        = mean(min_bin_content, na.rm = TRUE),
    pass_rate_all_bins  = mean(pass_all_bins, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_tbl)

# ---- save results ----
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)
write.csv(summary_tbl,
          "results/tables/bike_summary_manyseeds.csv",
          row.names = FALSE)

cat("Saved summary to results/tables/bike_summary_manyseeds.csv\n")
