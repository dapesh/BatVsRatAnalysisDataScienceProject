# Objective 2 Report: Bat vs Rat — Investigations A & B

This report revisits **Investigation A** (predator perception) and adds **Investigation B** (seasonal effects), using both provided datasets.

## Datasets
- `dataset1.csv`: Bat landing events with behavioural annotations.
- `dataset2.csv`: 30-min surveillance summaries (rat arrivals, bat landings, food availability).

## Methods & Assumptions
- Times parsed with `dayfirst=True`.
- Datasets aligned on **30min** blocks by flooring `start_time` and joining to `dataset2.time` (period start).
- `risk` and `reward` treated as **binary** (means interpreted as proportions).
- `season_label` in `dataset1` is taken as the zoologists' label without remapping; in `dataset2` season was optionally inferred from `month` for descriptive context.
- Outliers in continuous variables capped using IQR (non-destructive).

## Investigation A — Predator Perception (Classical Tests)
Figures: `figures_obj2/A1_landing_time_by_rat_presence.png`, `figures_obj2/A2_risk_by_rat_presence.png`, `figures_obj2/A3_reward_by_rat_presence.png`.

## Investigation B — Seasonal Effects (Classical Tests)
- B_bat_landing_to_food_anova: ANOVA F=16.494, p=0.0001
- B_foraging_efficiency_anova: ANOVA F=2.668, p=0.1027
Figures: `figures_obj2/B1_time_by_season_and_rat.png`, `figures_obj2/B2_activity_patterns.png`.

## Linear Regression Modelling (Focal for Assessment 3)
### Investigation A (Full Multiple Linear Regression)
- Response variable: **bat_landing_to_food**
- R² = 0.138, MAE = 6.365, RMSE = 8.207
- Top coefficients (standardised):
  - risk: 3.970
  - reward: 1.770
  - time_since_rat_left: 0.957
  - rat_arrival_number: -0.350
  - rat_minutes: -0.308
  - hours_after_sunset: 0.189
  - rat_activity_intensity: -0.030
  - food_availability: -0.000
  - rat_present_at_landing: 0.000
Figures: `figures_obj2/A_full_bat_landing_to_food_coefficients.png`, `figures_obj2/A_full_bat_landing_to_food_residuals.png`, `figures_obj2/A_full_bat_landing_to_food_qq.png`

### Investigation B (Seasonal LR: Winter vs Spring)
- Seasonal slices were empty or too small for LR.

## Limitations
- The uploaded `dataset1.csv` has all rows labelled as rat-present at landing (no control events), limiting classical comparisons.
- Seasonal split in `dataset1` may be absent or highly imbalanced; where unavailable, seasonal LR is skipped.
- Outlier capping may compress extremes; results should be interpreted with caution.
- Observational data; associations are not causal.

## Reproducibility
- python: 3.13.5
- platform: Windows-11-10.0.26100-SP0
- pandas: 2.3.2
- numpy: 2.1.3
- matplotlib: 3.10.0
- seaborn: 0.13.2
- scipy: builtins (SciPy)
- sklearn: 1.6.1
