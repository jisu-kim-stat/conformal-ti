# R/02_truth_content.R
content_function <- function(model_id, lower, upper, x) {
  y0 <- 3 * cos(x) - 5 * (x / 15)^2

  if (model_id == 1) return(pnorm(upper - y0, 0, sqrt(2)) - pnorm(lower - y0, 0, sqrt(2)))
  if (model_id == 2) return(pt(upper - y0, df = 3) - pt(lower - y0, df = 3))
  if (model_id == 3) return(pnorm(upper - y0, 0, sqrt(2 * x)) - pnorm(lower - y0, 0, sqrt(2 * x)))
  if (model_id == 4) {
    sd_x <- 0.3 + 3 * exp(-(x - 5)^2 / (2 * 1.5^2))
    return(pnorm(upper - y0, 0, sd_x) - pnorm(lower - y0, 0, sd_x))
  }
  if (model_id == 5) {
    sd_x <- ifelse(x < 5, 1 + 0.5 * x, 2 - 0.3 * (x - 5))
    return(pnorm(upper - y0, 0, sd_x) - pnorm(lower - y0, 0, sd_x))
  }
  if (model_id == 6) {
    mean_x <- ifelse(x < 5, 1, -1)
    return(pnorm(upper - y0, mean_x, 2) - pnorm(lower - y0, mean_x, 2))
  }

  stop("Unknown model.")
}
