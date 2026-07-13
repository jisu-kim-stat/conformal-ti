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

To visualize pointwise coverage for every seed : 
```bash
python3 tsa/plot_pointwise_coverage.py \
--intervals results_tsa_3way_intervals.csv \
--out plots/tsa_pointwise_coverage_5methods.png \
--target 0.90   --window 7
```
- `window` : rolling window.시각화를 위해 해당 일을 포함안 7일 단위로 local smoothing 