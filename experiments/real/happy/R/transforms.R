## experiments/real/happy/R/transforms.R

tf <- function(y) log1p(y)
itf <- function(z) pmax(expm1(z), 0.0)