## Simulation (Ours vs GY)

This repository also includes a simulation framework to compare:

- **Ours**: split + Hoeffding-adjusted tolerance intervals  
- **GY**: Guo–Young-style spline-based tolerance intervals (full-sample; no split)

### Setup

All simulation code is under:

- `R/sim/` : core functions (data generation, truth content, fitting, intervals, replication runner)
- `scripts/` : runnable entry scripts

Key files:
- `R/sim/data_generate.R` : `generate_data(model_id, n)` for simulation models
- `R/sim/truth_content.R` : `content_function(model_id, lower, upper, x)` for true conditional content
- `R/sim/one_replication_ours.R` : one replication for **Ours**
- `R/sim/one_replication_gy.R` : one replication for **GY**
- `R/sim/run_one_setting.R` : runs `M` replications and returns pointwise summaries for both methods
- `scripts/run_simulation_grid.R` : runs a grid over models and sample sizes and saves results

### Parameters and notation

- `content` : target content level (e.g., `0.90`)
- `mis = 1 - content` : miscoverage level
- `alpha` : confidence error level (confidence is `1 - alpha`)
- `n` : sample size per replication
- `M` : number of Monte Carlo replications per (model, n)

### How to run

From the **project root**:

```bash
Rscript scripts/run_simulation_grid.R
```

### Outputs
pointwise CSV with columns:
- `x` : evaluation grid point
- `coverage` : proportion of replicatioin with `true_content(x) >= content
- `mean_width` : mean interval with at `x`
- `na_proportion` : proportion of replications with NA/failed parameters
- `model` : model id
- `n` : sample size
- `Method` : `"Ours"` or `"GY"`

### Notes
- The evaluation grid is `x = seq(0, 10, length.out = n).
For reproducibility, seed are managed inside the replication loop.
- Results (`results/`, `*.csv`, `*.png`) are excluded.

