# ============================================================
# compute_spline_params.R
#
# smoother matrix S가 필요한 모든 값을 한 번에 계산:
#   - norm_lx : ||l_xh|| for each x (k-factor 계산용)
#   - sigma_hat: 오차 표준편차 추정값
#   - nu       : 유효 자유도
#
# n <= threshold : 단위벡터 반복 → 정확한 S
# n >  threshold : fit$lev 근사  → 빠른 계산
# ============================================================

compute_spline_params <- function(x, y, spar, threshold = 200) {
  n <- length(x)

  fit   <- smooth.spline(x, y, spar = spar)
  f_hat <- as.numeric(predict(fit, x)$y)

  if (n <= threshold) {
    # ── 정확한 방법: 단위벡터로 S 직접 구성 ──────────────────
    S <- matrix(0, n, n)
    for (j in seq_len(n)) {
      ej     <- rep(0, n); ej[j] <- 1
      S[, j] <- predict(smooth.spline(x, ej, spar = spar), x)$y
    }

    R      <- diag(n) - S
    resid  <- as.numeric(R %*% y)
    RtR    <- crossprod(R)
    tr1    <- sum(diag(RtR))
    tr2    <- sum(diag(RtR %*% RtR))

    norm_lx   <- sqrt(rowSums(S^2))      # 정확한 ||l_xh||
    sigma_hat <- sqrt(sum(resid^2) / tr1)
    nu        <- tr1^2 / tr2

  } else {
    # ── 근사 방법: fit$lev 사용 ───────────────────────────────
    # fit$lev = diag(S)
    # n 크면 S가 거의 projection → diag(S²) ≈ diag(S) = lev
    # 따라서 ||l_xh||² = (S²)_hh ≈ lev_h
    lev     <- fit$lev
    norm_lx <- sqrt(lev)                 # 근사 ||l_xh||

    # sigma_hat, nu: residual 기반으로 근사
    # tr((I-S)^T(I-S)) ≈ n - 2*tr(S) + tr(S^2)
    #                   ≈ n - 2*sum(lev) + sum(lev)   (diag(S²) ≈ lev)
    #                   = n - sum(lev)
    # tr(((I-S)^T(I-S))²) ≈ (n - sum(lev))²/n  [rough approx]
    resid     <- y - f_hat
    tr1       <- n - sum(lev)            # 근사 tr((I-S)'(I-S))
    tr2       <- sum((1 - lev)^2)        # 근사 tr(((I-S)'(I-S))²)

    sigma_hat <- sqrt(sum(resid^2) / tr1)
    nu        <- tr1^2 / tr2
  }

  list(
    f_hat     = f_hat,
    norm_lx   = norm_lx,
    sigma_hat = sigma_hat,
    nu        = nu,
    approx    = (n > threshold)
  )
}
