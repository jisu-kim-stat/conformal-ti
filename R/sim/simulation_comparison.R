# ============================================================
# Simulation Study: 4-Method Comparison
#
# Methods:
#   1. Parametric TI (homo)     – Guo & Young (2024), 등분산
#   2. Parametric TI (hetero)   – Guo & Young (2024), 이분산 추정
#   3. HCTI (symmetric)         – Hoeffding-adjusted, symmetric
#   4. HCTI (nonsymmetric)      – Hoeffding-adjusted, nonsymmetric
#
# Models:
#   1. N(0, 2)          – 등분산, 정규
#   2. t(3)             – 등분산, heavy-tail
#   3. N(0, 2x)         – 이분산, 정규
#   4. N(0, σ(x))       – 이분산, bump형
#   5. N(0, σ(x))       – 이분산, piecewise
#   6. χ²(3) - 3        – 등분산, skewed
# ============================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(foreach)
library(doParallel)
library(doRNG)
library(MASS)

# ============================================================
# 0. 공통 설정
# ============================================================
CONTENT <- 0.90
ALPHA   <- 0.05
M       <- 500          # Monte Carlo 반복
N_VEC   <- c(100, 200)  # 샘플 사이즈
MODELS  <- 1:6


# ============================================================
# 1. 데이터 생성
# ============================================================
base_mean <- function(x) 3 * cos(x) - 5 * (x / 15)^2

generate_data <- function(model_id, n) {
  x    <- seq(0, 10, length.out = n)
  base <- base_mean(x)

  noise <- switch(as.character(model_id),
    "1" = rnorm(n, 0, sqrt(2)),
    "2" = rt(n, df = 3),
    "3" = rnorm(n, 0, sqrt(2 * pmax(x, 0.01))),
    "4" = rnorm(n, 0, 0.3 + 3 * exp(-(x - 5)^2 / (2 * 1.5^2))),
    "5" = ifelse(x < 5,
                 rnorm(n, 0, 1 + 0.5 * x),
                 rnorm(n, 0, 2 - 0.3 * (x - 5))),
    "6" = rchisq(n, df = 3) - 3,
    stop("Unknown model_id")
  )
  data.frame(x = x, y = base + noise)
}


# ============================================================
# 2. True content 계산
# ============================================================
content_function <- function(model_id, lower, upper, x) {
  y0 <- base_mean(x)

  switch(as.character(model_id),
    "1" = pnorm(upper - y0, 0, sqrt(2))    - pnorm(lower - y0, 0, sqrt(2)),
    "2" = pt(upper - y0, df = 3) - pt(lower - y0, df = 3),
    "3" = pnorm(upper - y0, 0, sqrt(2 * x)) - pnorm(lower - y0, 0, sqrt(2 * x)),
    "4" = {
      sd_x <- 0.3 + 3 * exp(-(x - 5)^2 / (2 * 1.5^2))
      pnorm(upper - y0, 0, sd_x) - pnorm(lower - y0, 0, sd_x)
    },
    "5" = {
      sd_x <- ifelse(x < 5, 1 + 0.5 * x, 2 - 0.3 * (x - 5))
      pnorm(upper - y0, 0, sd_x) - pnorm(lower - y0, 0, sd_x)
    },
    "6" = pchisq(upper - y0 + 3, df = 3) - pchisq(lower - y0 + 3, df = 3),
    stop("Unknown model_id")
  )
}


# ============================================================
# 3. 모델 fitting 함수
# ============================================================
fit_mean_model <- function(x, y) {
  smooth.spline(x, y, cv = FALSE)
}

fit_var_model <- function(x, y, fit_mean) {
  res       <- y - as.numeric(predict(fit_mean, x)$y)
  log_res2  <- log(res^2 + 1e-6)
  gam::gam(log_res2 ~ gam::s(x), data = data.frame(x = x, log_res2 = log_res2))
}

predict_mean <- function(fit_mean, xnew) {
  as.numeric(predict(fit_mean, xnew)$y)
}

predict_var <- function(fit_var, xnew) {
  as.numeric(exp(predict(fit_var, newdata = data.frame(x = xnew))))
}


# ============================================================
# 4+5. Spline params 통합 계산 (n 기준 자동 분기)
#   n <= 200 : 단위벡터 반복 → 정확한 S
#   n >  200 : fit$lev 근사  → 빠른 계산
# ============================================================
source("/home/jisukim/conformal-ti/R/sim/compute_spline_params.R")


# ============================================================
# 6. Pointwise k-factor (Guo & Young 근사식 사용)
# ============================================================
build_k_cache <- function(n_vec = c(50, 100, 200, 500),
                           n_rep = 100,
                           content = CONTENT,
                           alpha   = ALPHA) {
  cache <- list()

  for (n in n_vec) {
    cat(sprintf("n=%d 캐시 생성 중...\n", n))

    params <- replicate(n_rep, {
      dat  <- generate_data(1, n)
      fit  <- fit_mean_model(dat$x, dat$y)
      info <- compute_spline_params(dat$x, dat$y, fit$spar)
      c(nu     = info$nu,
        nl_min = min(info$norm_lx),
        nl_max = max(info$norm_lx))
    })

    nu_med <- median(params["nu", ])
    nl_min <- min(params["nl_min", ]) * 0.8
    nl_max <- max(params["nl_max", ]) * 1.2

    cat(sprintf("  nu=%.1f, nl=[%.3f, %.3f]\n", nu_med, nl_min, nl_max))

    nl_grid <- seq(nl_min, nl_max, length.out = 30)

    # nl 하나씩 직접 적분
    k_values <- sapply(nl_grid, function(nl) {
      upper_t <- max(6 * nl, 2.0)
      k_grid  <- seq(0.5, 15, by = 0.1)

      probs <- sapply(k_grid, function(k) {
        val <- tryCatch(
          integrate(function(t) {
            ncp  <- pmin(t^2, 1e4)
            q    <- qchisq(content, df = 1, ncp = ncp)
            prob <- pchisq((nu_med * q) / k^2, df = nu_med,
                           lower.tail = FALSE)
            prob <- ifelse(is.nan(prob) | is.na(prob), 0, prob)
            exp(-t^2 / (2 * nl^2)) * prob
          }, 0, upper_t, subdivisions = 200, rel.tol = 1e-6)$value,
          error = function(e) NA
        )
        if (is.na(val)) return(0)
        (2 / (sqrt(2 * pi) * nl)) * val
      })
      k_grid[which.min(abs(probs - (1 - alpha)))]
    })

    cache[[as.character(n)]] <- list(
      nu      = nu_med,
      nl_grid = nl_grid,
      k_vals  = k_values
    )

    cat(sprintf("  k=[%.3f, %.3f]\n", min(k_values), max(k_values)))
  }

  cache
}

lookup_k <- function(norm_lx_vec, n, cache) {
  cc <- cache[[as.character(n)]]
  approx(cc$nl_grid, cc$k_vals, xout = norm_lx_vec,
         rule = 2)$y  # rule=2: 범위 벗어나면 끝값 사용
}

compute_k2_ptw_fixed <- function(norm_lx_vec, nu, P, alpha,
                                  k_grid = seq(0.5, 15, by = 0.1),
                                  nl_grid_n = 30) {  # 파라미터 추가
  gamma <- 1 - alpha

  nl_grid <- seq(min(norm_lx_vec) * 0.9,
                 max(norm_lx_vec) * 1.1,
                 length.out = nl_grid_n)  # 여기 적용

  k_for_nl <- sapply(nl_grid, function(nl) {
    upper_t <- max(6 * nl, 2.0)
    
    probs <- sapply(k_grid, function(k) {
      val <- tryCatch(
        integrate(function(t) {
          ncp  <- pmin(t^2, 1e4)
          q    <- qchisq(P, df = 1, ncp = ncp)
          prob <- pchisq((nu * q) / k^2, df = nu, lower.tail = FALSE)
          prob <- ifelse(is.nan(prob) | is.na(prob), 0, prob)
          exp(-t^2 / (2 * nl^2)) * prob
        }, 0, upper_t, subdivisions = 200, rel.tol = 1e-6)$value,
        error = function(e) NA
      )
      if (is.na(val)) return(0)
      (2 / (sqrt(2 * pi) * nl)) * val
    })
    k_grid[which.min(abs(probs - gamma))]
  })

  approx(nl_grid, k_for_nl, xout = norm_lx_vec)$y
}

system.time({
  k_cache <- build_k_cache(
    n_vec   = c(50, 100, 200, 500),
    n_rep   = 100,
    content = CONTENT,
    alpha   = ALPHA
  )
  saveRDS(k_cache, "k_cache.rds")
})

# ============================================================
# 7. Method 1: Parametric TI (homo) – Guo & Young 등분산
# ============================================================
one_rep_ptw_homo <- function(model_id, n, content, alpha, k_cache) {

  dat   <- generate_data(model_id, n)
  x <- dat$x; y <- dat$y

  fit   <- fit_mean_model(x, y)
  info  <- compute_spline_params(x, y, fit$spar)  # 통합 계산

  k_vec <- lookup_k(info$norm_lx, n, k_cache)  # 캐시에서 조회

  lower <- info$f_hat - k_vec * info$sigma_hat
  upper <- info$f_hat + k_vec * info$sigma_hat

  list(
    content   = content_function(model_id, lower, upper, x),
    width     = upper - lower,
    lambda_na = rep(0L, n)
  )
}


# ============================================================
# 8. Method 2: Parametric TI (hetero) – 이분산 추정 후 역변환
# ============================================================
one_rep_ptw_hetero <- function(model_id, n, content, alpha, k_cache) {

  dat   <- generate_data(model_id, n)
  x <- dat$x; y <- dat$y

  # mean / var 추정
  fit_mean <- fit_mean_model(x, y)
  fit_var  <- fit_var_model(x, y, fit_mean)
  mu_hat   <- predict_mean(fit_mean, x)
  var_hat  <- predict_var(fit_var, x)

  # 표준화 잔차로 등분산 스플라인 재적합
  z      <- (y - mu_hat) / sqrt(pmax(var_hat, 1e-8))
  fit_z  <- smooth.spline(x, z, cv = FALSE)
  info   <- compute_spline_params(x, z, fit_z$spar)  # 통합 계산
  
  k_vec <- lookup_k(info$norm_lx, n, k_cache)  # 캐시에서 조회

  # 역변환: 원래 스케일로
  lower <- mu_hat - k_vec * info$sigma_hat * sqrt(var_hat)
  upper <- mu_hat + k_vec * info$sigma_hat * sqrt(var_hat)  

  list(
    content   = content_function(model_id, lower, upper, x),
    width     = upper - lower,
    lambda_na = rep(0L, n)
  )
}


# ============================================================
# 9. Hoeffding lambda 계산 (공통)
#    symmetric:    F_hat(k)       >= C + lambda_alpha
#    nonsymmetric: F_hat(k1, k2)  >= C + 2 * lambda_alpha
# ============================================================
hoeffding_lambda <- function(alpha, n_cal) {
  sqrt(log(2 / alpha) / (2 * n_cal))
}


# ============================================================
# 10. Method 3: HCTI (symmetric)
# ============================================================
find_lambda_sym <- function(content, alpha, z_cal) {
  # z_cal: 표준화 잔차 (calibration set)
  n_cal  <- length(z_cal)
  lam    <- hoeffding_lambda(alpha, n_cal)
  target <- min(content + lam, 1)

  z_sorted <- sort(abs(z_cal))          # |z| 기준
  idx      <- ceiling(n_cal * target)
  idx      <- max(1, min(idx, n_cal))
  z_sorted[idx]
}

one_rep_hcti_sym <- function(model_id, n, content, alpha) {

  dat <- generate_data(model_id, n)
  x <- dat$x; y <- dat$y

  # train / cal 분리
  n_train <- floor(n / 2)
  idx     <- sample.int(n)
  tr_idx  <- idx[1:n_train]
  cal_idx <- idx[(n_train + 1):n]

  # train으로 fitting
  fit_mean <- fit_mean_model(x[tr_idx], y[tr_idx])
  fit_var  <- fit_var_model(x[tr_idx], y[tr_idx], fit_mean)

  # cal에서 표준화 잔차
  mu_cal  <- predict_mean(fit_mean, x[cal_idx])
  var_cal <- predict_var(fit_var,   x[cal_idx])
  z_cal   <- (y[cal_idx] - mu_cal) / sqrt(pmax(var_cal, 1e-8))

  # Hoeffding-adjusted k
  k_hat <- find_lambda_sym(content, alpha, z_cal)

  # 전체 x에서 TI 계산
  mu_all  <- predict_mean(fit_mean, x)
  var_all <- predict_var(fit_var,   x)
  lower   <- mu_all - k_hat * sqrt(var_all)
  upper   <- mu_all + k_hat * sqrt(var_all)

  list(
    content   = content_function(model_id, lower, upper, x),
    width     = upper - lower,
    lambda_na = rep(0L, n)
  )
}


# ============================================================
# 11. Method 4: HCTI (nonsymmetric)
#
# (k1, k2) = argmin (k2 - k1) s.t.
#   F_hat(k1, k2) >= C + 2*lambda_alpha,  k1 < k2
#
# 구현: z_cal의 empirical quantile 격자 탐색
#   → width(k2-k1) 최소화하는 (q_lo, q_hi) 쌍 선택
# ============================================================
find_lambda_nonsym <- function(content, alpha, z_cal) {
  n_cal  <- length(z_cal)
  lam    <- hoeffding_lambda(alpha, n_cal)
  target <- min(content + 2 * lam, 1)

  z_sorted <- sort(z_cal)

  # 필요한 커버 개수
  need <- ceiling(n_cal * target)
  need <- max(1, min(need, n_cal))

  # window 크기 need인 슬라이딩 윈도우 → width 최소화
  best_width <- Inf
  best_k1    <- z_sorted[1]
  best_k2    <- z_sorted[n_cal]

  for (i in seq_len(n_cal - need + 1)) {
    k1 <- z_sorted[i]
    k2 <- z_sorted[i + need - 1]
    w  <- k2 - k1
    if (w < best_width) {
      best_width <- w
      best_k1    <- k1
      best_k2    <- k2
    }
  }

  c(k1 = best_k1, k2 = best_k2)
}

one_rep_hcti_nonsym <- function(model_id, n, content, alpha) {

  dat <- generate_data(model_id, n)
  x <- dat$x; y <- dat$y

  # train / cal 분리
  n_train <- floor(n / 2)
  idx     <- sample.int(n)
  tr_idx  <- idx[1:n_train]
  cal_idx <- idx[(n_train + 1):n]

  # train으로 fitting
  fit_mean <- fit_mean_model(x[tr_idx], y[tr_idx])
  fit_var  <- fit_var_model(x[tr_idx], y[tr_idx], fit_mean)

  # cal에서 표준화 잔차
  mu_cal  <- predict_mean(fit_mean, x[cal_idx])
  var_cal <- predict_var(fit_var,   x[cal_idx])
  z_cal   <- (y[cal_idx] - mu_cal) / sqrt(pmax(var_cal, 1e-8))

  # Hoeffding-adjusted (k1, k2)
  ks <- find_lambda_nonsym(content, alpha, z_cal)

  # 전체 x에서 TI 계산
  mu_all  <- predict_mean(fit_mean, x)
  var_all <- predict_var(fit_var,   x)
  lower   <- mu_all + ks["k1"] * sqrt(var_all)
  upper   <- mu_all + ks["k2"] * sqrt(var_all)

  list(
    content   = content_function(model_id, lower, upper, x),
    width     = upper - lower,
    lambda_na = rep(0L, n)
  )
}


# ============================================================
# 12. 한 setting에서 전체 M번 반복
# ============================================================
METHODS <- c("Parametric (homo)", "Parametric (hetero)",
             "HCTI (sym)", "HCTI (nonsym)")

run_one_setting <- function(model_id, n, M, content, alpha) {

  x_grid <- seq(0, 10, length.out = n)

  run_method <- function(method) {
    foreach(
      b          = seq_len(M),
      .combine   = dplyr::bind_rows,
      .packages  = c("dplyr", "gam", "MASS"),
      .export    = c(
        "generate_data", "base_mean", "content_function",
        "fit_mean_model", "fit_var_model",
        "predict_mean", "predict_var",
        "compute_spline_params",
        "lookup_k", "k_cache", 
        "hoeffding_lambda",
        "find_lambda_sym", "find_lambda_nonsym",
        "one_rep_ptw_homo", "one_rep_ptw_hetero",
        "one_rep_hcti_sym", "one_rep_hcti_nonsym",
        "model_id", "n", "content", "alpha", "x_grid"
      )
    ) %dorng% {
      tryCatch({
        r <- switch(method,
          "Parametric (homo)"   = one_rep_ptw_homo(model_id, n, content, alpha, k_cache),
          "Parametric (hetero)" = one_rep_ptw_hetero(model_id, n, content, alpha, k_cache),
          "HCTI (sym)"          = one_rep_hcti_sym(model_id, n, content, alpha),
          "HCTI (nonsym)"       = one_rep_hcti_nonsym(model_id, n, content, alpha),
          stop("Unknown method")
        )
        dplyr::tibble(
          rep       = b,
          x         = x_grid,
          content   = r$content,
          width     = r$width,
          lambda_na = r$lambda_na
        )
      }, error = function(e) {
        dplyr::tibble(
          rep       = b,
          x         = x_grid,
          content   = NA_real_,
          width     = NA_real_,
          lambda_na = 1L
        )
      })
    }
  }

  summarize_method <- function(df_long, method) {
    df_long %>%
      dplyr::group_by(x) %>%
      dplyr::summarise(
        coverage   = mean(content >= CONTENT, na.rm = TRUE),
        mean_width = mean(width, na.rm = TRUE),
        na_prop    = mean(lambda_na == 1L),
        .groups    = "drop"
      ) %>%
      dplyr::mutate(
        model  = model_id,
        n      = n,
        Method = method
      )
  }

  results <- lapply(METHODS, function(m) {
    cat(sprintf("  [Model %d, n=%d] %s...\n", model_id, n, m))
    long <- run_method(m)
    summarize_method(long, m)
  })

  dplyr::bind_rows(results)
}


# ============================================================
# 13. 전체 시뮬레이션 실행
# ============================================================
run_simulation <- function(models  = MODELS,
                           n_vec   = N_VEC,
                           M       = 500,
                           content = CONTENT,
                           alpha   = ALPHA,
                           n_cores = parallel::detectCores() - 1,
                           seed    = 2024) {

  set.seed(seed)
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  on.exit(stopCluster(cl))

  cat(sprintf("병렬 코어 수: %d\n", n_cores))
  cat(sprintf("설정: M=%d, content=%.2f, alpha=%.2f\n\n", M, content, alpha))

  all_res <- list()
  for (mod in models) {
    for (n in n_vec) {
      cat(sprintf("=== Model %d, n=%d ===\n", mod, n))
      res <- run_one_setting(mod, n, M, content, alpha)
      all_res[[length(all_res) + 1]] <- res
    }
  }

  dplyr::bind_rows(all_res)
}


# ============================================================
# 14. 시각화
# ============================================================

# Figure A: Pointwise coverage (논문 Figure 3/4 스타일)
plot_coverage <- function(sim_df, content = CONTENT,
                          model_ids = 1:6) {

  method_colors <- c(
    "Parametric (homo)"   = "#d73027",
    "Parametric (hetero)" = "#fc8d59",
    "HCTI (sym)"          = "#1a9850",
    "HCTI (nonsym)"       = "#4575b4"
  )

  df <- sim_df %>%
    filter(model %in% model_ids) %>%
    mutate(
      Method  = factor(Method, levels = names(method_colors)),
      n_label = factor(paste0("n = ", n), levels = paste0("n = ", sort(unique(n)))),
      model_label = paste0("Model ", model)
    )

  ggplot(df, aes(x = x, y = coverage, colour = Method, shape = Method)) +
    geom_point(size = 0.9, alpha = 0.8) +
    geom_hline(yintercept = content, linetype = "dashed", colour = "black") +
    facet_grid(model_label ~ n_label) +
    scale_colour_manual(values = method_colors) +
    scale_shape_manual(values = c(16, 17, 15, 18)) +
    scale_y_continuous(limits = c(0.65, 1.02),
                       breaks = seq(0.7, 1.0, 0.1)) +
    labs(x = "x", y = "Coverage Probability",
         title = sprintf("Pointwise Coverage (nominal = %.0f%%)", content * 100),
         colour = "Method", shape = "Method") +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor  = element_blank(),
          strip.background  = element_rect(fill = "white"),
          legend.position   = "bottom")
}

# Figure B: Mean width
plot_width <- function(sim_df, model_ids = 1:6) {

  method_colors <- c(
    "Parametric (homo)"   = "#d73027",
    "Parametric (hetero)" = "#fc8d59",
    "HCTI (sym)"          = "#1a9850",
    "HCTI (nonsym)"       = "#4575b4"
  )

  df <- sim_df %>%
    filter(model %in% model_ids) %>%
    mutate(
      Method      = factor(Method, levels = names(method_colors)),
      n_label     = factor(paste0("n = ", n), levels = paste0("n = ", sort(unique(n)))),
      model_label = paste0("Model ", model)
    )

  ggplot(df, aes(x = x, y = mean_width, colour = Method, linetype = Method)) +
    geom_line(linewidth = 0.8, alpha = 0.9) +
    facet_grid(model_label ~ n_label, scales = "free_y") +
    scale_colour_manual(values = method_colors) +
    labs(x = "x", y = "Mean Width",
         title = "Mean Interval Width",
         colour = "Method", linetype = "Method") +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          strip.background = element_rect(fill = "white"),
          legend.position  = "bottom")
}

# Figure C: 평균 coverage 테이블 (논문 Table 1 스타일)
summary_table <- function(sim_df, content = CONTENT) {
  sim_df %>%
    group_by(model, n, Method) %>%
    summarise(
      avg_coverage = mean(coverage, na.rm = TRUE),
      avg_width    = mean(mean_width, na.rm = TRUE),
      .groups      = "drop"
    ) %>%
    mutate(
      avg_coverage = round(avg_coverage, 3),
      avg_width    = round(avg_width, 3)
    ) %>%
    arrange(model, n, Method)
}


# ============================================================
# 15. 실행 진입점
# ============================================================
if (sys.nframe() == 0) {

  cat("=== Full Simulation (M=500) ===\n")

  # 캐시 로드 또는 생성
  if (file.exists("k_cache.rds")) {
    cat("캐시 로드 중...\n")
    k_cache <- readRDS("k_cache.rds")
  } else {
    cat("캐시 생성 중...\n")
    system.time({
      k_cache <- build_k_cache(n_vec = c(50, 100, 200, 500))
    })
    saveRDS(k_cache, "k_cache.rds")
  }

  cl <- makeCluster(39)
  registerDoParallel(cl)
  on.exit(stopCluster(cl))

  total_start <- proc.time()

  all_res <- list()
  for (mod in 1:6) {
    for (n in c(50, 100, 200, 500)) {
      setting_start <- proc.time()
      cat(sprintf("\n>>> Model %d, n=%d 시작...\n", mod, n))

      res <- run_one_setting(mod, n, M = 500,
                             content = CONTENT, alpha = ALPHA)
      all_res[[length(all_res) + 1]] <- res

      elapsed <- (proc.time() - setting_start)["elapsed"]
      cat(sprintf("    완료! %.1f초 (%.1f분)\n", elapsed, elapsed / 60))
    }
  }

  sim_res <- dplyr::bind_rows(all_res)

  total_elapsed <- (proc.time() - total_start)["elapsed"]
  cat(sprintf("\n=== 전체 완료! 총 %.1f분 ===\n", total_elapsed / 60))

  saveRDS(sim_res, "sim_results_final.rds")

  p_cov <- plot_coverage(sim_res)
  p_wid <- plot_width(sim_res)
  ggsave("fig_coverage_final.png", p_cov, width = 14, height = 16, dpi = 150)
  ggsave("fig_width_final.png",    p_wid, width = 14, height = 16, dpi = 150)

  cat("\n--- Summary Table ---\n")
  print(summary_table(sim_res))
}