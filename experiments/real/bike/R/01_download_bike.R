library(dplyr); library(tidyr); library(ggplot2)
library(data.table); library(gam); library(MASS); library(splines); library(tibble)

source("experiments/real/bike/R/10_scores_lambda.R")
source("experiments/real/bike/R/20_intervals_ours.R")
source("experiments/real/bike/R/30_intervals_gy.R")
source("experiments/real/bike/R/40_intervals_normal.R")
source("experiments/real/bike/R/50_eval_conditional.R")
source("experiments/real/bike/R/01_data_bike.R")
source("experiments/real/bike/R/90_one_run.R")

bike <- load_bike_data()

res1 <- one_run_bike_plot(seed_split=1, bike_df=bike, nbins=20)
print(res1$overall)
print(res1$cond_summary_seed)
