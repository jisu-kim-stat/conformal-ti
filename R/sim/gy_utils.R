compute_norm <- function(vector) sqrt(sum(vector^2))

compute_probability <- function(nu, t, P, k, norm_lx_h) {
  tryCatch({
    q_val <- qchisq(P, df = 1, ncp = pmin(1e4,t^2))
    out <- numeric(length(q_val))
    for (i in seq_along(q_val)) {
      q <- q_val[i]
      if (is.nan(q) || is.na(q)) { out[i] <- 0; next }
      threshold <- (nu * q) / (k^2)
      prob <- pchisq(threshold, df = nu, lower.tail = FALSE)
      out[i] <- ifelse(is.nan(prob) || is.na(prob), 1e-6, prob)
    }
    return(out)
  }, error = function(e) rep(0, length(t)))
} #prop 3.1 의 Pr부분 계산하는 함수. t값을 받아 확률을 계산함 

integrand <- function(t, k, nu, P, norm_lx_h) {
  exp_term <- exp(-t^2 / (2 * norm_lx_h^2))
  prob_term <- compute_probability(nu, t, P, k, norm_lx_h)
  if (log(exp_term) < -20) return(1e-3) #너무 작으면 -inf이 나오므로 이부분 방지
  return(exp_term * prob_term)
}

find_k_factor <- function(nu, norm_lx_h, content, alpha) {
  c2 <- norm_lx_h^2

  # content quantile (HPD analogue)
  q1 <- qchisq(content, df = 1, ncp = c2)

  # confidence quantile
  q2 <- qchisq(alpha, df = nu)

  if (is.nan(q1) || is.nan(q2) || q2 == 0) return(NA_real_)

  sqrt((nu * q1) / q2)
}

