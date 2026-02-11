run_one_setting <- function(model_id, n, M, content, alpha) {
  x_grid <- seq(0, 10, length.out = n)

  # local aliases to avoid name collisions
  content_level    <- content
  alpha_conf <- alpha

  run_method_long <- function(method) {
    foreach(
      b = 1:M,
      .combine  = dplyr::bind_rows,
      .packages = c("dplyr"),
      .export   = c('model_id', 'n', 'content_level', 'alpha_conf', 'x_grid',
       'one_replication_ours', 'one_replication_gy')
    ) %dorng% {

      tryCatch({

        # ---- quick type guards (if this fails, it's argument passing) ----
        stopifnot(is.numeric(model_id), length(model_id) == 1)
        stopifnot(is.numeric(n), length(n) == 1)
        stopifnot(is.numeric(content_level), length(content_level) == 1)
        stopifnot(is.numeric(alpha_conf), length(alpha_conf) == 1)

        if (method == "HCTI") {
          r <- one_replication_ours(
            model_id = model_id,
            n        = n,
            content  = content_level,
            alpha    = alpha_conf,
            seed     = b
          )
        } else if (method == "Parametric TI") {
          r <- one_replication_gy(
            model_id = model_id,
            n        = n,
            content  = content_level,
            alpha = alpha_conf,
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

        # --- expose the real error (critical for foreach debugging) ---
        msg <- paste0(
          "\n[ERROR DETAILS]\n",
          "method: ", method, "\n",
          "model_id: ", model_id, " (", paste(class(model_id), collapse=","), ")\n",
          "n: ", n, " (", paste(class(n), collapse=","), ")\n",
          "content_level: ", content_level, " (", paste(class(content_level), collapse=","), ")\n",
          "alpha_conf: ", alpha_conf, " (", paste(class(alpha_conf), collapse=","), ")\n",
          "rep: ", b, "\n",
          "message: ", conditionMessage(e), "\n"
        )
        stop(msg, call. = FALSE)
      })
    }
  }

  summarize_pointwise <- function(df_long, method) {
    out <- df_long %>%
      dplyr::group_by(x) %>%
      dplyr::summarise(
        coverage      = mean(content >= content_level, na.rm = TRUE),
        mean_width    = mean(width, na.rm = TRUE),
        na_proportion = mean(lambda_na == 1),
        .groups = "drop"
      )

    out$model  <- model_id
    out$n      <- n
    out$Method <- method
    out
  }

  long_ours <- run_method_long("HCTI")
  long_gy   <- run_method_long("Parametric TI")

  dplyr::bind_rows(
    summarize_pointwise(long_ours, "HCTI"),
    summarize_pointwise(long_gy, "Parametric TI")
  ) %>%
    dplyr::select(x, coverage, mean_width, na_proportion, model, n, Method)
}
