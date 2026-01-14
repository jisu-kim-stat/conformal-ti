## experiments/real/happy/R/00_config.R

# Root assumption: working directory is ti_project/
DATA_RAW_DIR  <- "data/raw/happy"
DATA_PROC_DIR <- "data/processed/happy"

TRAIN_FILE <- file.path(DATA_RAW_DIR, "happy_A")
TEST_FILE  <- file.path(DATA_RAW_DIR, "happy_B")

HAPPY_COLS <- c(
  "id", "mag_r", "u_g", "g_r", "r_i", "i_z",
  "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5"
)

X_COL <- "mag_r"
Y_COL <- "z_spec"

SEED <- 1

TRIM_Q <- NULL


# experiment defaults
ALPHA <- 0.10
DELTA <- 0.05
N_SAMPLE <- 5000
SEEDS <- 1:50