# R/sim/calibration_index.R

pac_calibration_index <- function(n, content, alpha,
                                  rule = c("hoeffding", "exact_binomial")) {
  rule <- match.arg(rule)

  stopifnot(
    length(n) == 1L, n >= 1L,
    length(content) == 1L, content > 0, content < 1,
    length(alpha) == 1L, alpha > 0, alpha < 1
  )

  if (rule == "hoeffding") {
    k <- ceiling(
      n * content +
        sqrt(n * log(1 / alpha) / 2)
    )
  }

  if (rule == "exact_binomial") {
    # Smallest k such that
    # P{Binomial(n, content) >= k} <= alpha.
    k <- qbinom(1 - alpha, size = n, prob = content) + 1L
  }

  # k = n + 1 means that no finite order statistic can deliver
  # the requested distribution-free PAC guarantee.
  min(as.integer(k), n + 1L)
}


pac_calibration_threshold <- function(scores, content, alpha,
                                      rule = c("hoeffding",
                                               "exact_binomial")) {
  rule <- match.arg(rule)

  scores <- sort(scores)
  n <- length(scores)

  k <- pac_calibration_index(
    n = n,
    content = content,
    alpha = alpha,
    rule = rule
  )

  threshold <- if (k <= n) scores[k] else Inf

  list(
    threshold = threshold,
    index = k,
    correction = k / n - content,
    rule = rule
  )
}