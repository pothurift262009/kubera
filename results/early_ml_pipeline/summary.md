# Early ML pipeline outputs (archives/early_ml_pipeline)

LightGBM and meta-model era. Published with the flags attached.

| File | What it is |
|---|---|
| `feature_importance.csv` | v1 LightGBM feature ranking (49 features) |
| `feature_importance_v2.csv` | v2 ranking after feature reduction (10 features) |
| `threshold_sweep.csv` | Precision/recall/F1 across 21 probability thresholds |
| `threshold_sweep_v2.csv` | v2 sweep — **single row, placeholder value** |
| `filtered_signals_v2.csv` | 664 filtered signals with regime context |

## What these show

**`day_progress` dominates the v1 feature importance** by a wide margin over every
price and volume feature. The model's strongest predictor was elapsed time in the
session — intraday seasonality, not tradeable edge.

**The threshold sweep is honest and unflattering.** At a 0.40 threshold: precision
0.109, recall 0.729, firing on 31.1% of all bars. That is close to the base rate;
the classifier is not separating much.

## Known problems with these files

`threshold_sweep_v2.csv` contains one row with a profit factor of exactly `1.0` —
a placeholder that was never replaced with a real computation. It should not be
read as a measurement.

Timestamps in `filtered_signals_v2.csv` are UTC (`03:45:00`, `09:05:00`) while the
rest of the pipeline assumes IST — an unlabelled 5.5-hour offset that was never
reconciled.
