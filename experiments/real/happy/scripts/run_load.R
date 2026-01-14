
source("experiments/real/happy/R/config.R")
source("experiments/real/happy/R/load_data.R")
set.seed(SEED)


dat <- load_happy_train_test()
train <- dat$train
test  <- dat$test

cat("\n[head train]\n"); print(head(train))
cat("\n[head test]\n");  print(head(test))

dir.create(DATA_PROC_DIR, showWarnings = FALSE, recursive = TRUE)
readr::write_csv(train, file.path(DATA_PROC_DIR, "happy_train_A.csv"))
readr::write_csv(test,  file.path(DATA_PROC_DIR, "happy_test_B.csv"))
cat("\n[saved] processed csv written to:", DATA_PROC_DIR, "\n")