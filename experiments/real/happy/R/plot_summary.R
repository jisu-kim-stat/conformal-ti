suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

plot_summary <- function(
  results_csv = "experiments/real/happy/results/tables/results_happy_1d.csv",
  out_dir = "experiments/real/happy/results/figures",
  content_target = 0.90,
  conf_target = 0.95
) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  df0 <- readr::read_csv(results_csv, show_col_types = FALSE)

  # method별 전체 반복 수(NA 포함) - 실패 카운트 계산용
  n_all <- df0 %>% count(method, name = "n_all")

  # content/width가 finite인 것만 사용 (complete-case)
  df <- df0 %>%
    filter(is.finite(content), is.finite(mean_width))

  # ---- summary table (complete-case 기준)
  sumtab <- df %>%
    group_by(method) %>%
    summarise(
      n_rep = n(),
      content_mean = mean(content),
      content_sd = sd(content),
      content_median = median(content),
      prob_meet_target = mean(content >= content_target),
      width_mean = mean(mean_width),
      width_sd = sd(mean_width),
      width_median = median(mean_width),
      .groups = "drop"
    ) %>%
    left_join(n_all, by = "method") %>%
    mutate(
      n_failed = n_all - n_rep
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 4)))

  cat("\n[Summary (complete-case; NA removed)]\n")
  print(sumtab)

  # ---- Figure 1: content distribution (target=0.90)
  p1 <- ggplot(df, aes(x = method, y = content)) +
    geom_boxplot() +
    geom_hline(yintercept = content_target, linetype = "dashed", linewidth = 0.6) +
    labs(
      x = NULL,
      y = "Empirical content on test (B)",
      title = "Happy 1D: Content across repetitions",
      subtitle = paste0("Dashed line = content target ", content_target, " (NA removed)")
    ) +
    theme_bw()

  ggsave(file.path(out_dir, "happy_1d_content_boxplot.pdf"), p1, width = 7, height = 4)

  # ---- Figure 2: width distribution
  p2 <- ggplot(df, aes(x = method, y = mean_width)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(
    x = NULL,
    y = "Mean interval width on test (B) [log10 scale]",
    title = "Happy 1D: Width across repetitions (log scale; NA removed)"
  ) +
  theme_bw()

  ggsave(file.path(out_dir, "happy_1d_width_boxplot_log10.pdf"), p2, width = 7, height = 4)

  wlim <- quantile(df$mean_width, probs = 0.95, na.rm = TRUE)

  p2_zoom <- ggplot(df, aes(x = method, y = mean_width)) +
    geom_boxplot(outlier.shape = NA) +
    coord_cartesian(ylim = c(0, wlim)) +
    labs(
      x = NULL,
      y = "Mean interval width on test (B)",
      title = "Happy 1D: Width across repetitions (zoomed to 95% quantile)"
    ) +
    theme_bw()

  ggsave(file.path(out_dir, "happy_1d_width_boxplot_zoom95.pdf"), p2_zoom, width = 7, height = 4)

  # ---- Figure 3: P(content >= 0.90) with confidence target 0.95
  p3data <- sumtab %>% select(method, prob_meet_target)

  p3 <- ggplot(p3data, aes(x = method, y = prob_meet_target)) +
    geom_col() +
    geom_hline(yintercept = conf_target, linetype = "dashed", linewidth = 0.6) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      x = NULL,
      y = paste0("P(content >= ", content_target, ")"),
      title = "Happy 1D: Probability of meeting content target",
      subtitle = paste0("Dashed line = confidence target ", conf_target, " (NA removed)")
    ) +
    theme_bw()

  ggsave(file.path(out_dir, "happy_1d_prob_meet_target.pdf"), p3, width = 7, height = 4)

  # save summary
  readr::write_csv(sumtab, file.path(out_dir, "happy_1d_summary_complete_case.csv"))

  invisible(list(df = df, summary = sumtab))
}
