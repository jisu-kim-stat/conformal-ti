# R/sim/run_one_setting.R

summarize_pointwise <- function(df_long, method) {

    out <- df_long %>%
      dplyr::group_by(x) %>%
      dplyr::summarise(
        mean_content      = mean(content, na.rm = TRUE),
        pointwise_success = mean(content >= content_level, na.rm = TRUE),
        mean_width        = mean(width, na.rm = TRUE),
        na_proportion     = mean(lambda_na == 1),
        .groups = "drop"
      )

    out$model   <- model_id
    out$n_train <- n_train
    out$n_cal   <- n_cal
    out$n_test  <- n_test
    out$Method  <- method

    out
  }

summarize_marginal <- function(df_long, method) {

  by_rep <- df_long %>%
    dplyr::group_by(rep) %>%
    dplyr::summarise(
      marginal_content = mean(content, na.rm = TRUE),
      average_width    = mean(width, na.rm = TRUE),
      na_proportion    = mean(lambda_na == 1),
      .groups = "drop"
    )

  out <- by_rep %>%
    dplyr::summarise(
      marginal_content_mean = mean(marginal_content, na.rm = TRUE),
      marginal_content_sd   = sd(marginal_content, na.rm = TRUE),
      marginal_pac_success  = mean(marginal_content >= content_level, na.rm = TRUE),

      average_width_mean = mean(average_width, na.rm = TRUE),
      average_width_sd   = sd(average_width, na.rm = TRUE),

      na_proportion = mean(na_proportion, na.rm = TRUE),
      .groups = "drop"
    )

  out$model   <- model_id
  out$n_train <- n_train
  out$n_cal   <- n_cal
  out$n_test  <- n_test
  out$Method  <- method

  out
}

summarize_px_good <- function(pointwise_df, method) {

  out <- pointwise_df %>%
    dplyr::summarise(
      px_good_proportion = mean(pointwise_success >= 1 - alpha_conf, na.rm = TRUE),
      min_pointwise_success = min(pointwise_success, na.rm = TRUE),
      q25_pointwise_success = quantile(pointwise_success, 0.25, na.rm = TRUE),
      median_pointwise_success = median(pointwise_success, na.rm = TRUE),
      q75_pointwise_success = quantile(pointwise_success, 0.75, na.rm = TRUE),
      max_pointwise_success = max(pointwise_success, na.rm = TRUE),
      .groups = "drop"
    )

  out$model   <- model_id
  out$n_train <- n_train
  out$n_cal   <- n_cal
  out$n_test  <- n_test
  out$Method  <- method

  out
}

run_one_setting <- function(model_id,
                            n_train,
                            n_cal,
                            n_test = 1000,
                            M,
                            content,
                            alpha) {

  # x-axis for pointwise summary / plotting
  # model 6 is high-dimensional, so x is only an index.
  if (model_id == 6) {
    x_grid <- seq_len(n_test)
  } else {
    x_grid <- seq(-2, 2, length.out = n_test)
  }

  # local aliases to avoid name collisions inside foreach
  content_level <- content
  alpha_conf    <- alpha

  run_method_long <- function(method) {

    foreach(
      b = 1:M,
      .combine  = dplyr::bind_rows,
      .packages = c(
        "dplyr",
        "mgcv",
        "glmnet",
        "quantreg",
        "splines",
        "MASS"
      ),
      .export = c(
        "model_id",
        "n_train",
        "n_cal",
        "n_test",
        "content_level",
        "alpha_conf",
        "x_grid",

        # replication functions
        "one_replication_ours",
        "one_replication_pti",

        # data + truth
        "base_mean",
        "generate_data",
        "content_function",

        # HCTI fitting
        "fit_mean_model",
        "fit_var_model",
        "predict_mean",
        "predict_var",
        "fit_mean_model_hd",
        "fit_var_model_hd",
        "predict_mean_hd",
        "predict_var_hd",
        "fit_mean_model_auto",
        "fit_var_model_auto",
        "predict_mean_auto",
        "predict_var_auto",

        # CQR-TI fitting
        "fit_quantile_model",
        "predict_quantile",
        "fit_quantile_model_hd",
        "predict_quantile_hd",
        "fit_quantile_model_auto",
        "predict_quantile_auto",

        # calibration / PTI utilities
        "find_lambda_hat",
        "find_score_cutoff",
        "find_k_factor"
      )
    ) %dorng% {

      tryCatch({

        stopifnot(is.numeric(model_id), length(model_id) == 1)
        stopifnot(is.numeric(n_train), length(n_train) == 1)
        stopifnot(is.numeric(n_cal), length(n_cal) == 1)
        stopifnot(is.numeric(n_test), length(n_test) == 1)
        stopifnot(is.numeric(content_level), length(content_level) == 1)
        stopifnot(is.numeric(alpha_conf), length(alpha_conf) == 1)

        if (method %in% c("HCTI", "CQR-TI")) {

          r <- one_replication_ours(
            method   = method,
            model_id = model_id,
            n_train  = n_train,
            n_cal    = n_cal,
            n_test   = n_test,
            content  = content_level,
            alpha    = alpha_conf,
            seed     = b
          )

        } else if (method == "Parametric-TI") {

          r <- one_replication_pti(
            model_id = model_id,
            n_train  = n_train,
            n_cal    = n_cal,
            n_test   = n_test,
            content  = content_level,
            alpha    = alpha_conf,
            seed     = b
          )

        } else {
          stop("Unknown method: ", method)
        }

        dplyr::tibble(
          rep       = b,
          x         = x_grid,
          content   = r$content,
          width     = r$width,
          lambda_na = r$lambda_na
        )

      }, error = function(e) {

        msg <- paste0(
          "\n[ERROR DETAILS]\n",
          "method: ", method, "\n",
          "model_id: ", model_id, " (", paste(class(model_id), collapse = ","), ")\n",
          "n_train: ", n_train, " (", paste(class(n_train), collapse = ","), ")\n",
          "n_cal: ", n_cal, " (", paste(class(n_cal), collapse = ","), ")\n",
          "n_test: ", n_test, " (", paste(class(n_test), collapse = ","), ")\n",
          "content_level: ", content_level, " (", paste(class(content_level), collapse = ","), ")\n",
          "alpha_conf: ", alpha_conf, " (", paste(class(alpha_conf), collapse = ","), ")\n",
          "rep: ", b, "\n",
          "message: ", conditionMessage(e), "\n"
        )

        stop(msg, call. = FALSE)
      })
    }
  }

  # --------------------------------------------------
  # 1. Pointwise summary
  # For each x:
  #   pointwise_success(x)
  #   = P_D{ conditional content at x >= C }
  # estimated by Monte Carlo over replications.
  # --------------------------------------------------
  summarize_pointwise <- function(df_long, method) {

    out <- df_long %>%
      dplyr::group_by(x) %>%
      dplyr::summarise(
        mean_content      = mean(content, na.rm = TRUE),
        pointwise_success = mean(content >= content_level, na.rm = TRUE),
        mean_width        = mean(width, na.rm = TRUE),
        na_proportion     = mean(lambda_na == 1),
        .groups = "drop"
      )

    out$model   <- model_id
    out$n_train <- n_train
    out$n_cal   <- n_cal
    out$n_test  <- n_test
    out$Method  <- method

    out
  }

  # --------------------------------------------------
  # 2. Marginal summary
  # For each replication:
  #   marginal_content_b = average content over test x's.
  # Then:
  #   marginal_pac_success
  #   = P_D{ marginal_content_b >= C }.
  # --------------------------------------------------
  summarize_marginal <- function(df_long, method) {

    by_rep <- df_long %>%
      dplyr::group_by(rep) %>%
      dplyr::summarise(
        marginal_content = mean(content, na.rm = TRUE),
        average_width    = mean(width, na.rm = TRUE),
        na_proportion    = mean(lambda_na == 1),
        .groups = "drop"
      )

    out <- by_rep %>%
      dplyr::summarise(
        marginal_content_mean = mean(marginal_content, na.rm = TRUE),
        marginal_content_sd   = sd(marginal_content, na.rm = TRUE),
        marginal_pac_success  = mean(marginal_content >= content_level, na.rm = TRUE),

        average_width_mean = mean(average_width, na.rm = TRUE),
        average_width_sd   = sd(average_width, na.rm = TRUE),

        na_proportion = mean(na_proportion, na.rm = TRUE)
      )

    out$model   <- model_id
    out$n_train <- n_train
    out$n_cal   <- n_cal
    out$n_test  <- n_test
    out$Method  <- method

    out
  }

  # --------------------------------------------------
  # 3. PX-good proportion
  # This estimates:
  #   P_X{x : P_D[content(x;D) >= C] >= 1 - alpha}.
  #
  # For model 6 this is not interpreted as a pointwise curve,
  # but we still compute it numerically. In plots, exclude model 6.
  # --------------------------------------------------
  summarize_px_good <- function(pointwise_df, method) {

    out <- pointwise_df %>%
      dplyr::summarise(
        px_good_proportion = mean(pointwise_success >= 1 - alpha_conf, na.rm = TRUE),

        min_pointwise_success    = min(pointwise_success, na.rm = TRUE),
        q25_pointwise_success    = as.numeric(quantile(pointwise_success, 0.25, na.rm = TRUE)),
        median_pointwise_success = median(pointwise_success, na.rm = TRUE),
        q75_pointwise_success    = as.numeric(quantile(pointwise_success, 0.75, na.rm = TRUE)),
        max_pointwise_success    = max(pointwise_success, na.rm = TRUE),

        mean_pointwise_success = mean(pointwise_success, na.rm = TRUE),
        mean_content           = mean(mean_content, na.rm = TRUE),
        mean_width             = mean(mean_width, na.rm = TRUE),
        na_proportion          = mean(na_proportion, na.rm = TRUE)
      )

    out$model   <- model_id
    out$n_train <- n_train
    out$n_cal   <- n_cal
    out$n_test  <- n_test
    out$Method  <- method

    out
  }

  # --------------------------------------------------
  # Run methods
  # --------------------------------------------------
  long_hcti <- run_method_long("HCTI")
  long_cqr  <- run_method_long("CQR-TI")

  long_list <- list(
    "HCTI"   = long_hcti,
    "CQR-TI" = long_cqr
  )

  # Current PTI implementation is 1D smoothing-spline based.
  # Exclude it for model 6 unless a high-dimensional PTI is implemented.
  if (model_id != 6) {
    long_pti <- run_method_long("Parametric-TI")
    long_list[["Parametric-TI"]] <- long_pti
  }

  # --------------------------------------------------
  # Summaries
  # --------------------------------------------------
  pointwise_df <- dplyr::bind_rows(
    lapply(names(long_list), function(method) {
      summarize_pointwise(long_list[[method]], method)
    })
  ) %>%
    dplyr::select(
      x,
      mean_content,
      pointwise_success,
      mean_width,
      na_proportion,
      model,
      n_train,
      n_cal,
      n_test,
      Method
    )

  marginal_df <- dplyr::bind_rows(
    lapply(names(long_list), function(method) {
      summarize_marginal(long_list[[method]], method)
    })
  ) %>%
    dplyr::select(
      marginal_content_mean,
      marginal_content_sd,
      marginal_pac_success,
      average_width_mean,
      average_width_sd,
      na_proportion,
      model,
      n_train,
      n_cal,
      n_test,
      Method
    )

  px_good_df <- dplyr::bind_rows(
    lapply(names(long_list), function(method) {
      pw_method <- pointwise_df %>%
        dplyr::filter(Method == method)

      summarize_px_good(pw_method, method)
    })
  ) %>%
    dplyr::select(
      px_good_proportion,
      min_pointwise_success,
      q25_pointwise_success,
      median_pointwise_success,
      q75_pointwise_success,
      max_pointwise_success,
      mean_pointwise_success,
      mean_content,
      mean_width,
      na_proportion,
      model,
      n_train,
      n_cal,
      n_test,
      Method
    )

  list(
    pointwise = pointwise_df,
    marginal  = marginal_df,
    px_good   = px_good_df
  )
}

