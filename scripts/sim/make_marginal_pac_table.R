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
  "ASR-TI",
  "CQR-TI",
  "Parametric-TI"
)

table_long <- dat %>%
  filter(
    design %in% c("normal", "uniform"),
    model %in% c(3, 4),
    Method %in% method_order
  ) %>%
  mutate(
    design = factor(
      design,
      levels = c("normal", "uniform")
    ),
    model = factor(
      model,
      levels = c(3, 4)
    ),
    Method = factor(
      Method,
      levels = method_order
    ),

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
    Model = case_when(
        row_number() == 1 & model == 3 ~ "Model 3",
        row_number() == 1 & model == 4 ~ "Model 4",
        TRUE ~ ""
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

# 각 design당 2 models × 4 methods = 8 rows
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
    "Monte Carlo estimates of marginal PAC success for the ",
    "heteroscedastic Model~3 and globally skewed Model~4. ",
    "Each entry reports the empirical marginal PAC success probability, ",
    "with average interval width in parentheses. ",
    "The upper and lower panels correspond to ",
    "$X\\sim N(0,1)$ and ",
    "$X\\sim\\operatorname{Unif}(-2,2)$, respectively. ",
    "Bold success probabilities fall below the target ",
    "$1-\\alpha=0.95$. Results are based on ",
    "$M=1000$ Monte Carlo replications."
    ),
    label = "tab:marginal-pac-representative"
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