## experiments/real/happy/R/compare_1d.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(readr)
})

source("experiments/real/happy/R/config.R")
source("experiments/real/happy/R/load_data.R")

source("experiments/real/happy/R/ours_1d.R")
source("experiments/real/happy/R/gy_1d.R")

prep_train_seed <- function(train_df, n_sample, seed) {
  df <- train_df %>%
    dplyr::select(mag_r, z_spec) %>%
    dplyr::filter(is.finite(mag_r), is.finite(z_spec))

  if (nrow(df) < n_sample) {
    stop("Not enough rows after filtering: n=", nrow(df), " < n_sample=", n_sample)
  }

  set.seed(seed)
  idx <- sample.int(nrow(df), size = n_sample, replace = FALSE)
  df[idx, , drop = FALSE]
}


prep_test_clean <- function(test_df) {
  test_df %>%
    dplyr::select(mag_r, z_spec) %>%
    dplyr::filter(is.finite(mag_r), is.finite(z_spec))
}

run_many_seeds_1d <- function(seeds = SEEDS,
                              n_sample = N_SAMPLE,
                              alpha = ALPHA,
                              delta = DELTA) {

  dat <- load_happy_train_test()
  train0 <- dat$train
  test0  <- dat$test

  test <- prep_test_clean(test0)

  rows <- list()

  for (seed in seeds) {

    # 공정 비교: seed별로 A에서 동일한 표본 n_sample 고정
    train_seed <- prep_train_seed(train0, n_sample = n_sample, seed = seed)
    set.seed(seed)
    idx <- sample.int(nrow(train_seed))
    n_tr <- floor(nrow(train_seed)/2)

    df_tr <- train_seed[idx[1:n_tr], , drop = FALSE]
    df_cal <- train_seed[idx[(n_tr+1):nrow(train_seed)], , drop = FALSE]

    train_x <- df_tr$mag_r
    train_y <- df_tr$z_spec
    cal_x   <- df_cal$mag_r
    cal_y   <- df_cal$z_spec

    test_x <- test$mag_r
    test_y <- test$z_spec

    # -------------------------
    # (1) Ours: split(half/half)
    # -------------------------
    r1 <- run_ours_1d(
      train = train_seed,
      test  = test,
      alpha = alpha,
      delta = delta,
      n_sample = n_sample,
      seed = seed
    )
    r1$seed <- seed
    rows[[length(rows) + 1]] <- r1

    # -------------------------
    # (2) GY: no-split (use all train_seed)
    # -------------------------
    r2 <- run_gy_1d(
      train_x = train_x,
      train_y = train_y,
      cal_x   = cal_x,
      cal_y   = cal_y,
      test_x  = test_x,
      test_y  = test_y,
      alpha   = alpha,
      delta   = delta,
      df_mean = 12,
      df_var  = 12,
      df_std  = 12
    )
    r2 <- tibble::tibble(method="GY(1D mag_r)",
    content = r2$content, mean_width = r2$mean_width)

    r2$seed <- seed
    rows[[length(rows) + 1]] <- r2

    cat("[seed]", seed, 
    "ours=", round(r1$content,4), 
    "GY=", round(r2$content,4), "\n")
  }

  df <- dplyr::bind_rows(lapply(rows, tibble::as_tibble))

  summary <- df %>%
    dplyr::group_by(method) %>%
    dplyr::summarise(
      n_rep = sum(is.finite(content) & is.finite(mean_width)),
      n_total = dplyr::n(),
      content_mean = mean(content, na.rm = TRUE),
      content_sd = sd(content, na.rm = TRUE),
      content_median = median(content, na.rm = TRUE),
      width_mean = mean(mean_width, na.rm = TRUE),
      width_sd = sd(mean_width, na.rm = TRUE),
      width_median = median(mean_width, na.rm = TRUE),
      .groups = "drop"
    )

  list(df = df, summary = summary)
}
