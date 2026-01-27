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

#### Run Simulation (CP)
From the **project root**:

```bash
Rscript scripts/run_simulation_grid.R
```

#### Make a plot
```bash
Rscript scripts/make_plot.R
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


## TSA Passenger Data Experiment
This experiment used to evaluate HCTI (Hoeffding-based Conformal Tolerance Intervals) under both time-based and random data splits.

### Dataset
- Daily U.S. air passenger throughput data (January 2019 – January 2022)

- Each observation records the number of passengers screened at TSA checkpoints

- Publicly available dataset:
https://github.com/hunj/tsa-passenger-throughput

- The series exhibits a severe structural break during the COVID-19 period

### Model
We compare the following interval construction methods:

- GY-homo / GY-hetero
Penalized smoothing spline–based intervals using classical k-factor rules
(homoskedastic vs heteroskedastic variance).

- Ours-homo / Ours-hetero
Gradient boosting models for conditional mean (and variance),
combined with Hoeffding-adjusted quantile-based intervals.

= Ours-PS-hetero
Uses the same spline base learner as GY, but applies our Hoeffding-based
interval construction rule to isolate the effect of the interval method.

### Transformation
Passenger throughput is positive and heavy-tailed.
For numerical stability, we apply the inverse hyperbolic sine (IHS) transform. 

### Running 
From the project root directory:
```bash
python3 tsa/run_tsa_3way.py
```
Key optional arguments:
```bash
python3 tsa/run_tsa_3way.py \
  --alpha 0.10 \
  --delta 0.05 \
  --seed_from 1 \
  --seed_to 50 \
  --split_mode random \
  --out_csv results_tsa_3way.csv
```

### Outputs
The script produces:
- Summary results `results_tsa_3way.csv`
- Interval-level results `results_tsa_3way_intervals.csv`

### Visualization
To visualize prediction intervals for a fixed seed:
```bash
python3 tsa/plot.py --csv results_tsa_3way_intervals.csv --seed 1
```


## References

- **Guo, Y. and Young, D. S. (2024).**  
  *Approximate tolerance intervals for nonparametric regression models.*  
  Journal of Nonparametric Statistics, **36**(1), 212–239.  
  DOI: https://doi.org/10.1080/10485252.2023.2277260
