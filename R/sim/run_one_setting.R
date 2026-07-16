# R/sim/run_one_setting.R

run_one_setting <- function(model_id,
                            n_train,
                            n_cal,
                            n_test = 1000,
                            M,
                            content,
                            alpha,
                            epsilon_grid = c(0, 0.01, 0.02, 0.03, 0.05),
                            design = "grid",
                            n_bins = 50) {

  content_level <- content
  alpha_conf <- alpha
  rep_ids <- seq_len(M)

  make_bins <- function(x, design, n_bins = 50) {

    if (design == "normal") {
      breaks <- stats::qnorm(seq(0.001, 0.999, length.out = n_bins + 1))
    } else {
      breaks <- seq(
        min(x, na.rm = TRUE),
        max(x, na.rm = TRUE),
        length.out = n_bins + 1
      )
    }

    cut(
      x,
      breaks = breaks,
      include.lowest = TRUE,
      labels = FALSE
    )
  }

  summarize_pointwise <- function(df_long, method) {

    out <- dplyr::bind_rows(lapply(epsilon_grid, function(eps) {

      df_long %>%
        dplyr::group_by(x) %>%
        dplyr::summarise(
          epsilon = eps,
          x_bin = NA_integer_,
          bin_count = dplyr::n(),
          mean_content = mean(content, na.rm = TRUE),
          pointwise_success = mean(content >= content_level - eps, na.rm = TRUE),
          mean_width = mean(width, na.rm = TRUE),
          na_proportion = mean(lambda_na == 1),
          .groups = "drop"
        )

    }))

    out$model <- model_id
    out$design <- design
    out$n_train <- n_train
    out$n_cal <- n_cal
    out$n_test <- n_test
    out$Method <- method

    out
  }

  summarize_marginal <- function(df_long, method) {

    by_rep <- df_long %>%
      dplyr::group_by(rep) %>%
      dplyr::summarise(
        marginal_content = mean(content, na.rm = TRUE),
        average_width = mean(width, na.rm = TRUE),
        na_proportion = mean(lambda_na == 1),
        .groups = "drop"
      )

    out <- by_rep %>%
      dplyr::summarise(
        marginal_content_mean = mean(marginal_content, na.rm = TRUE),
        marginal_content_sd = sd(marginal_content, na.rm = TRUE),
        marginal_pac_success = mean(marginal_content >= content_level, na.rm = TRUE),
        average_width_mean = mean(average_width, na.rm = TRUE),
        average_width_sd = sd(average_width, na.rm = TRUE),
        na_proportion = mean(na_proportion, na.rm = TRUE),
        .groups = "drop"
      )

    out$model <- model_id
    out$design <- design
    out$n_train <- n_train
    out$n_cal <- n_cal
    out$n_test <- n_test
    out$Method <- method

    out
  }

  summarize_px_good <- function(pointwise_df, method) {

    out <- pointwise_df %>%
      dplyr::group_by(epsilon) %>%
      dplyr::summarise(
        px_good_proportion = weighted.mean(
          as.numeric(pointwise_success >= 1 - alpha_conf),
          w = bin_count,
          na.rm = TRUE
        ),

        min_pointwise_success = min(pointwise_success, na.rm = TRUE),
        q25_pointwise_success = as.numeric(quantile(pointwise_success, 0.25, na.rm = TRUE)),
        median_pointwise_success = median(pointwise_success, na.rm = TRUE),
        q75_pointwise_success = as.numeric(quantile(pointwise_success, 0.75, na.rm = TRUE)),
        max_pointwise_success = max(pointwise_success, na.rm = TRUE),

        mean_pointwise_success = weighted.mean(pointwise_success, w = bin_count, na.rm = TRUE),
        mean_content = weighted.mean(mean_content, w = bin_count, na.rm = TRUE),
        mean_width = weighted.mean(mean_width, w = bin_count, na.rm = TRUE),
        na_proportion = weighted.mean(na_proportion, w = bin_count, na.rm = TRUE),
        .groups = "drop"
      )

    out$model <- model_id
    out$design <- design
    out$n_train <- n_train
    out$n_cal <- n_cal
    out$n_test <- n_test
    out$Method <- method

    out
  }

  run_method_long <- function(method) {

    foreach(
      b = rep_ids,
      .combine = dplyr::bind_rows,
      .packages = c("dplyr", "mgcv", "glmnet", "quantreg", "splines", "MASS"),
      .export = c(
        "model_id", "n_train", "n_cal", "n_test",
        "content_level", "alpha_conf", "design",
        "one_replication_ours", "one_replication_pti",
        "base_mean", "generate_data", "content_function", "generate_eval_data",
        "fit_mean_model", "fit_var_model", "predict_mean", "predict_var",
        "fit_mean_model_hd", "fit_var_model_hd", "predict_mean_hd", "predict_var_hd",
        "fit_mean_model_auto", "fit_var_model_auto", "predict_mean_auto", "predict_var_auto",
        "fit_quantile_model", "predict_quantile", "fit_quantile_model_hd", "predict_quantile_hd",
        "fit_quantile_model_auto", "predict_quantile_auto",
        "find_lambda_hat", "find_score_cutoff", "find_k_factor", "find_asym_shape", "asym_residual_score"
      )
    ) %dorng% {

      tryCatch({

        if (method %in% c("HCTI", "HCTI-asym", "CQR-TI")) {

          r <- one_replication_ours(
            method = method,
            model_id = model_id,
            n_train = n_train,
            n_cal = n_cal,
            n_test = n_test,
            content = content_level,
            alpha = alpha_conf,
            seed = b,
            design = design
          )

        } else if (method == "Parametric-TI") {

          r <- one_replication_pti(
            model_id = model_id,
            n_train = n_train,
            n_cal = n_cal,
            n_test = n_test,
            content = content_level,
            alpha = alpha_conf,
            seed = b,
            design = design
          )

        } else {
          stop("Unknown method: ", method)
        }

        if (is.null(r$x)) {
          stop("Replication output does not contain x.")
        }

        x_out <- r$x

        if (is.matrix(x_out) || is.data.frame(x_out)) {
          x_out <- seq_len(length(r$content))
        }

        dplyr::tibble(
          rep       = b,
          x         = as.numeric(x_out),
          content   = as.numeric(r$content),
          width     = as.numeric(r$width),
          lambda_na = as.integer(r$lambda_na)
        )

      }, error = function(e) {

        msg <- paste0(
          "\n[ERROR DETAILS]\n",
          "method: ", method, "\n",
          "model_id: ", model_id, "\n",
          "design: ", design, "\n",
          "n_train: ", n_train, "\n",
          "n_cal: ", n_cal, "\n",
          "n_test: ", n_test, "\n",
          "rep: ", b, "\n",
          "message: ", conditionMessage(e), "\n"
        )

        stop(msg, call. = FALSE)
      })
    }
  }

  long_hcti <- run_method_long("HCTI")
  long_hcti_asym <- run_method_long("HCTI-asym")
  long_cqr  <- run_method_long("CQR-TI")

  long_list <- list(
    "HCTI" = long_hcti,
    "HCTI-asym" = long_hcti_asym,
    "CQR-TI" = long_cqr
  )

  if (model_id != 6) {
    long_pti <- run_method_long("Parametric-TI")
    long_list[["Parametric-TI"]] <- long_pti
  }

  pointwise_df <- dplyr::bind_rows(
    lapply(names(long_list), function(method) {
      summarize_pointwise(long_list[[method]], method)
    })
  ) %>%
    dplyr::select(
      x, x_bin, bin_count, epsilon,
      mean_content, pointwise_success, mean_width, na_proportion,
      model, design, n_train, n_cal, n_test, Method
    )

  marginal_df <- dplyr::bind_rows(
    lapply(names(long_list), function(method) {
      summarize_marginal(long_list[[method]], method)
    })
  ) %>%
    dplyr::select(
      marginal_content_mean, marginal_content_sd, marginal_pac_success,
      average_width_mean, average_width_sd, na_proportion,
      model, design, n_train, n_cal, n_test, Method
    )

  px_good_df <- dplyr::bind_rows(
    lapply(names(long_list), function(method) {
      pw_method <- pointwise_df %>%
        dplyr::filter(Method == method)

      summarize_px_good(pw_method, method)
    })
  ) %>%
    dplyr::select(
      epsilon, px_good_proportion,
      min_pointwise_success, q25_pointwise_success,
      median_pointwise_success, q75_pointwise_success,
      max_pointwise_success,
      mean_pointwise_success, mean_content, mean_width, na_proportion,
      model, design, n_train, n_cal, n_test, Method
    )

  list(
    pointwise = pointwise_df,
    marginal = marginal_df,
    px_good = px_good_df
  )
}