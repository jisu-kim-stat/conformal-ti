## experiments/real/happy/scripts/run_compare_1d.R

source("experiments/real/happy/R/compare_1d.R")

res <- run_many_seeds_1d()

print(res$summary %>% mutate(across(where(is.numeric), ~round(.x, 4))))

dir.create("experiments/real/happy/results/tables", recursive = TRUE, showWarnings = FALSE)
readr::write_csv(res$df, "experiments/real/happy/results/tables/results_happy_1d.csv")
readr::write_csv(res$summary, "experiments/real/happy/results/tables/summary_happy_1d.csv")

cat("\n[saved]\n")
cat(" - experiments/real/happy/results/tables/results_happy_1d.csv\n")
cat(" - experiments/real/happy/results/tables/summary_happy_1d.csv\n")
R