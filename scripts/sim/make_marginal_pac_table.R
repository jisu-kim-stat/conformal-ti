# scripts/make_marginal_pac_table.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(knitr)
  library(kableExtra)
})

# ============================================================
# File paths: 실제 경로에 맞게 수정
# ============================================================

normal_path  <- "/Users/jisukim/ti_project/results/sim/models/plots_alt_dgp/normal/summary_marginal_pac.csv"
uniform_path <- "/Users/jisukim/ti_project/results/sim/models/plots_alt_dgp/uniform/summary_marginal_pac.csv"

output_path <- "/Users/jisukim/ti_project/tables/table_marginal_pac_representative.tex"

dir.create(
  dirname(output_path),
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# Appendix table: remaining models
# ============================================================

models_to_include <- c(3,4)

output_path <- "/Users/jisukim/ti_project/tables/table_marginal_pac_representative.tex"

# ============================================================
# Read and combine
# ============================================================

normal_dat <- read_csv(normal_path, show_col_types = FALSE)
uniform_dat <- read_csv(uniform_path, show_col_types = FALSE)

dat <- bind_rows(normal_dat, uniform_dat)

# 파일이 올바르게 결합됐는지 확인
if (!all(c("normal", "uniform") %in% unique(dat$design))) {
  stop("Both normal and uniform designs must be present.")
}

# ============================================================
# Select representative models and format estimates
# ============================================================

target <- 0.95

method_order <- c(
  "SR-TI",
  "CQR-TI",
  "Parametric-TI",
  "GY-TI"
)

table_long <- dat %>%
  filter(
  design %in% c("normal", "uniform"),
  model %in% models_to_include,
  Method %in% method_order
  ) %>%
  mutate(
    design = factor(design, levels = c("normal", "uniform")),
    model = factor(model, levels = models_to_include),
    Method = factor(Method, levels = method_order),

    # Marginal PAC success
    pac_value = sprintf("%.3f", marginal_pac_success),

    # PAC target 미달인 값만 bold
    pac_value = if_else(
      marginal_pac_success < target,
      paste0("\\textbf{", pac_value, "}"),
      pac_value
    ),

    # Average width: 소수점 둘째 자리
    width_value = sprintf("%.2f", average_width_mean),

    # 최종 표기: PAC success (average width)
    estimate = paste0(
      pac_value,
      " (",
      width_value,
      ")"
    )
  ) %>%
  arrange(design, model, Method)

# ============================================================
# Wide format
# ============================================================

table_wide <- table_long %>%
  select(
    design,
    model,
    Method,
    n_cal,
    estimate
  ) %>%
  pivot_wider(
    names_from = n_cal,
    values_from = estimate,
    names_prefix = "n_"
  ) %>%
  arrange(design, model, Method) %>%
  group_by(design, model) %>%
  mutate(
    Model = if_else(
      row_number() == 1,
      paste("Model", as.character(model)),
      ""
    )
  ) %>%
  ungroup() %>%
  select(
    design,
    Model,
    Method,
    n_200,
    n_500,
    n_1000
  )

# Design 열은 pack_rows로 표시할 것이므로 제거
table_print <- table_wide %>%
  select(-design)

# Each design has 2 models x 5 methods = 10 rows.
n_normal <- sum(table_wide$design == "normal")
n_uniform <- sum(table_wide$design == "uniform")

# ============================================================
# LaTeX table
# ============================================================

latex_table <- table_print %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    escape = FALSE,
    linesep = "",
    align = c("l", "l", "c", "c", "c"),
    col.names = c(
      "Model",
      "Method",
      "$n_{\\mathrm{cal}}=200$",
      "$n_{\\mathrm{cal}}=500$",
      "$n_{\\mathrm{cal}}=1000$"
    ),
    caption = paste0(
    "Empirical marginal PAC success for homoscedastic Gaussian Model~1, ",
    "heteroscedastic heavy-tailed Model~2, and heteroscedastic \\(X\\)-dependent-skewness Model~5, ",
    "with mean interval width in parentheses, under the normal and uniform ",
    "covariate designs. Bold entries fall below the target ",
    "\\(1-\\alpha=0.95\\). Results are based on \\(M=1000\\) Monte Carlo ",
    "replications."
  ),
    label = "marginal-pac-representative"
  ) %>%
  pack_rows(
    index = c(
      "Normal covariate design" = n_normal,
      "Uniform covariate design" = n_uniform
    ),
    bold = TRUE,
    hline_before = TRUE
  ) %>%
  kable_styling(
    latex_options = "hold_position",
    position = "center",
    font_size = 9
  )

writeLines(latex_table, output_path)

cat("Saved LaTeX table to:", output_path, "\n\n")
print(table_wide)
