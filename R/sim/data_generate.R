# R/sim/data_generate.R

generate_x <- function(model_id, n, design = "uniform") {

  design <- match.arg(design, c("uniform", "normal"))

  if (model_id %in% 1:5) {
    if (design == "uniform") {
      return(runif(n, min = -2, max = 2))
    }

    if (design == "normal") {
      return(rnorm(n, mean = 0, sd = 1))
    }
  }

  if (model_id == 6) {
    d <- 20
    x <- matrix(rnorm(n * d), nrow = n, ncol = d)
    colnames(x) <- paste0("x", 1:d)
    return(x)
  }
}

# --------------------------------------------------
# generate_data () : train/calibration data for one replication
# chose random x
# --------------------------------------------------
generate_data <- function(model_id, n, design = "grid") {

  if (model_id %in% 1:5) {

    x <- generate_x(model_id, n, design = design)
    base <- base_mean(x)

    if (model_id == 1) {
      y <- base + rnorm(n, 0, 1)
    }

    if (model_id == 2) {
      y <- base + rt(n, df = 3)
    }

    if (model_id == 3) {
      y <- base + rnorm(n, 0, 1) * (1 + abs(x))
    }

    if (model_id == 4) {
      y <- base + rexp(n, rate = 1) - 1
    }

    if (model_id == 5) {
      sigma <- sqrt(0.75 + 0.5 * sin(2 * pi * x)^2)
      y <- base + rnorm(n, 0, sigma)
    }

    return(data.frame(x = x, y = y))
  }

  if (model_id == 6) {

    x <- generate_x(model_id, n, design = design)

    beta <- c(0.5, 0.4, 0.3, 0.2, rep(0, 16))
    base <- as.vector(x %*% beta)
    y <- base + rnorm(n, 0, 1)

    return(data.frame(x, y = y))
  }

  stop("Unknown model_id.")
}

# --------------------------------------------------
# generate_eval_data () : test/evaluation data for one replication
# chose x for deterministic quantile grid 
# --------------------------------------------------
generate_eval_data <- function(model_id, n, design = "uniform") {

  design <- match.arg(design, c("uniform", "normal"))

  if (model_id %in% 1:5) {

    p <- (seq_len(n) - 0.5) / n

    if (design == "uniform") {
      x <- qunif(p, min = -2, max = 2)
    }

    if (design == "normal") {
      x <- qnorm(p, mean = 0, sd = 1)
    }

    base <- base_mean(x)

    # y는 true content 계산에는 필요 없지만 형식 맞추기 위해 생성
    if (model_id == 1) {
      y <- base + rnorm(n, 0, 1)
    }

    if (model_id == 2) {
      y <- base + rt(n, df = 3)
    }

    if (model_id == 3) {
      y <- base + rnorm(n, 0, 1) * (1 + abs(x))
    }

    if (model_id == 4) {
      y <- base + rexp(n, rate = 1) - 1
    }

    if (model_id == 5) {
      sigma <- sqrt(0.75 + 0.5 * sin(2 * pi * x)^2)
      y <- base + rnorm(n, 0, sigma)
    }

    return(data.frame(x = x, y = y))
  }

  if (model_id == 6) {
    return(generate_data(model_id, n, design = design))
  }

  stop("Unknown model_id.")
}