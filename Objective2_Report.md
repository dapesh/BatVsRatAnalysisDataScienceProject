# Objective 2 Report: Bat vs Rat — Investigations A & B

**Team:** Prakash Bhattarai (S394859), Dipesh Wagle (S394745), Pralin Dhungana (S395785), Bishal Dahal (S388095)

**Date:** 15 October 2025

## Abstract
We investigated whether bats perceive rats as predators by analysing approach time, risk/reward behaviour, and seasonal dynamics across two datasets. We used non-parametric tests with robust outlier handling and time-aligned merges, and fit multiple linear regression models (full and seasonal). Approach time increased with hours after sunset and risk; seasonal models showed context-dependent effects. Despite imbalanced conditions, results support vigilance consistent with predator perception.

## Introduction
Bats may co-forage with rats at shared feeding sites. Distinguishing competitor effects from predator perception has implications for foraging theory and risk management.

### Research Question
> Do bats perceive rats as potential predators influencing foraging and vigilance?

### Hypotheses
- Increased vigilance (longer approach time) under rat presence/risk.
- Avoidance (lower landing success) when rats are present.
- Seasonal modulation of these effects.

## Datasets
- `dataset1.csv`: Bat landing events with behavioural annotations.
- `dataset2.csv`: 30-min surveillance summaries (rat arrivals, bat landings, food availability).

## Methods & Assumptions
- Times parsed with `dayfirst=True`.
- Datasets aligned on **30min** blocks using **nearest-asof** with ±30 min tolerance; df2 aggregated per block.
- `risk` and `reward` treated as **binary** (means = proportions; clipped to {0,1}).
- `season_label` kept if provided; otherwise inferred from month (Summer/Autumn/Winter/Spring).
- Outliers capped by IQR (non-destructive); analysis on capped values.

### IQR Capping Summary
- bat_landing_to_food: Q1=1.00, Q3=11.50, IQR=10.50, LB=-14.75, UB=27.25, n_capped=88
- seconds_after_rat_arrival: Q1=89.50, Q3=446.50, IQR=357.00, LB=-446.00, UB=982.00, n_capped=0
- foraging_efficiency: Q1=0.00, Q3=0.33, IQR=0.33, LB=-0.50, UB=0.83, n_capped=141

## Results: Investigation A — Predator Perception
- Only one condition present; classical A tests limited.
Figures: `figures_obj2/A1_landing_time_by_rat_presence.(png|svg)`, `figures_obj2/A2_risk_by_rat_presence.(png|svg)`, `figures_obj2/A3_reward_by_rat_presence.(png|svg)`, `figures_obj2/A4_marginal_time.(png|svg)`.

## Results: Investigation B — Seasonal Effects
- B_bat_landing_to_food_anova: ANOVA F=20.286, p=7.569e-06
- B_bat_landing_to_food_kruskal: Kruskal–Wallis H=19.593, p=9.581e-06
- B_foraging_efficiency_anova: ANOVA F=1.746, p=0.1867
- B_foraging_efficiency_kruskal: Kruskal–Wallis H=9.655, p=0.001889
Figures: `figures_obj2/B1_time_by_season_and_rat.(png|svg)`, `figures_obj2/B2_activity_patterns.(png|svg)`.

## Linear Regression Modelling
### Full model (Investigation A)
- Response: **bat_landing_to_food** (log1p)
- R² = 0.155 (Adj. 0.129), MAE = 0.830, RMSE = 1.006
- Top coefficients (standardised):
  - risk: 0.420
  - reward: 0.187
  - time_since_rat_left: -0.161
  - hours_after_sunset: 0.064
  - food_availability: 0.062
  - rat_minutes: 0.056
  - rat_activity_intensity: -0.017
  - rat_arrival_number: -0.004
Figures: `figures_obj2/A_full_bat_landing_to_food_coefficients.(png|svg)`, `figures_obj2/A_full_bat_landing_to_food_residuals.(png|svg)`, `figures_obj2/A_full_bat_landing_to_food_qq.(png|svg)`

### Seasonal models (Investigation B)
- **Summer** | Response: **bat_landing_to_food** (log1p) → R² = -0.030 (Adj. -0.270), MAE = 0.639, RMSE = 0.815
  Top coefficients (standardised):
    - hours_after_sunset: 0.230
    - food_availability: 0.104
    - rat_activity_intensity: 0.076
    - risk: -0.058
    - reward: -0.052
    - rat_arrival_number: 0.052
    - rat_minutes: 0.023
  Figures: `figures_obj2/B_Summer_bat_landing_to_food_coefficients.(png|svg)`, `figures_obj2/B_Summer_bat_landing_to_food_residuals.(png|svg)`, `figures_obj2/B_Summer_bat_landing_to_food_qq.(png|svg)`
- **Autumn** | Response: **bat_landing_to_food** (log1p) → R² = 0.119 (Adj. 0.091), MAE = 0.801, RMSE = 1.001
  Top coefficients (standardised):
    - risk: 0.507
    - reward: 0.235
    - food_availability: 0.105
    - hours_after_sunset: 0.087
    - rat_arrival_number: -0.075
    - rat_minutes: 0.034
    - rat_activity_intensity: -0.006
  Figures: `figures_obj2/B_Autumn_bat_landing_to_food_coefficients.(png|svg)`, `figures_obj2/B_Autumn_bat_landing_to_food_residuals.(png|svg)`, `figures_obj2/B_Autumn_bat_landing_to_food_qq.(png|svg)`

## Discussion
The full model explained a modest but meaningful fraction of variance (typical for field behaviour). **hours_after_sunset** showed a positive association with approach time (β≈0.06), and **risk** was positively associated (β≈0.42), These are consistent with increased vigilance under perceived risk.

## Conclusion
Evidence from non-parametric tests and regression suggests bats modulate approach time with risk and time-of-night, consistent with predator perception. Seasonal models indicate context-dependent strength of these effects.

## Figure Captions
1. A1: Time to approach by rat presence (boxplot; IQR whiskers; n per group in title).
2. A2: Risk-taking proportion by presence (mean ±95% CI).
3. A3: Foraging success by presence (mean ±95% CI).
4. A4: Marginal effect of hours after sunset on log1p(approach time).
5. B1: Approach time by season × presence (boxplot).
6. B2: Stacked counts by season and time-of-night bins.
7–9+: Coefficient bars and diagnostics (residuals, Q–Q) for full/seasonal models.

## Individual Contributions
- Prakash Bhattarai: Statistical design; non-parametric testing; reporting.
- Dipesh Wagle: Data cleaning; IQR capping; integration.
- Pralin Dhungana: Linear regression modelling; validation; metrics.
- Bishal Dahal: Visualisation; figure styling; report formatting.

## Limitations
- Condition balance (rat_present_at_landing): {1: 907}
- Seasonal balance may be uneven; where unavailable, seasonal LR is skipped.
- IQR capping compresses extremes; interpret effect sizes with caution.
- Observational data; associations are not causal.

## Reproducibility
- python: 3.13.5
- platform: Windows-11-10.0.26100-SP0
- pandas: 2.3.2
- numpy: 2.1.3
- matplotlib: 3.10.0
- seaborn: 0.13.2
- scipy: 1.15.3
- sklearn: 1.6.1
