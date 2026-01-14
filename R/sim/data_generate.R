# R/sim/data_generate.R
generate_data <- function(model_id, n) {
  x <- seq(0, 10, length.out = n)
  base <- base_mean(x)

  if (model_id == 1) y <- base + rnorm(n, 0, sqrt(2))
  if (model_id == 2) y <- base + rt(n, df = 3)
  if (model_id == 3) y <- base + rnorm(n, 0, sqrt(2 * x))
  if (model_id == 4) {
    sd_x <- 0.3 + 3 * exp(-(x - 5)^2 / (2 * 1.5^2))
    y <- base + rnorm(n, 0, sd_x)
  }
  if (model_id == 5) {
    noise <- ifelse(x < 5,
                    rnorm(n, 0, 1 + 0.5 * x),
                    rnorm(n, 0, 2 - 0.3 * (x - 5)))
    y <- base + noise
  }
  if (model_id == 6) {
    noise <- ifelse(x < 5, rnorm(n, 1, 2), rnorm(n, -1, 2))
    y <- base + noise
  }

  data.frame(x = x, y = y)
}
