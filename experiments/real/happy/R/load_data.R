## experiments/real/happy/R/01_load_data.R

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

source("experiments/real/happy/R/config.R")

load_happy_one <- function(path, cols = HAPPY_COLS) {
  if (!file.exists(path)) stop("File not found: ", path)

  # 혹시 모를 'extra' 열 대비(13번째 토큰이 들어오는 경우)
  cols_with_extra <- c(cols, "extra")

  df <- read_table(
    file = path,
    col_names = cols_with_extra,
    comment = "#",
    col_types = cols(
      id     = col_double(),
      mag_r  = col_double(),
      u_g    = col_double(),
      g_r    = col_double(),
      r_i    = col_double(),
      i_z    = col_double(),
      z_spec = col_double(),
      feat1  = col_double(),
      feat2  = col_double(),
      feat3  = col_double(),
      feat4  = col_double(),
      feat5  = col_double(),
      extra  = col_character()
    ),
    progress = FALSE
  )

  # extra가 "실제로 값이 있는" 경우는 데이터 포맷이 다른 것이므로 에러
  if (!all(is.na(df$extra) | df$extra == "")) {
    bad <- df %>% filter(!(is.na(extra) | extra == "")) %>% head(5)
    stop("Unexpected extra token(s) detected in file: ", path,
         "\nExample rows with extra:\n", paste(capture.output(print(bad)), collapse = "\n"))
  }

  df <- df %>% dplyr::select(all_of(cols))
  df
}

standardize_xy <- function(df, trim_q = TRIM_Q) {
  df2 <- df %>%
    dplyr::select(mag_r, z_spec) %>%
    dplyr::filter(is.finite(mag_r), is.finite(z_spec))

  if (!is.null(trim_q)) {
    qx <- quantile(df2$mag_r, probs = trim_q, na.rm = TRUE)
    df2 <- df2 %>% dplyr::filter(mag_r >= qx[1], mag_r <= qx[2])
  }
  df2
}


happy_processed_paths <- function() {
  list(
    train = file.path("data/processed/happy", "happy_A.csv"),
    test  = file.path("data/processed/happy", "happy_B.csv")
  )
}

load_happy_train_test <- function(force_rebuild = FALSE) {

  pp <- happy_processed_paths()

  if (!force_rebuild && file.exists(pp$train) && file.exists(pp$test)) {
    tr <- readr::read_csv(pp$train, show_col_types = FALSE, progress = FALSE)
    te <- readr::read_csv(pp$test,  show_col_types = FALSE, progress = FALSE)

    return(list(train = tr, test = te, source = "processed_csv"))
  }

  # ---- 없으면 raw에서 만들기 (네 기존 로직 그대로) ----
  tr_raw <- load_happy_one("data/raw/happy/happy_A")
  te_raw <- load_happy_one("data/raw/happy/happy_B")

  dir.create("data/processed/happy", recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(tr_raw, pp$train)
  readr::write_csv(te_raw, pp$test)

  list(train = tr_raw, test = te_raw, source = "raw_then_saved")
}
