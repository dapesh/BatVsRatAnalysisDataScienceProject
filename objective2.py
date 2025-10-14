# objective2.py
# HIT140 Assessment 3 — Objective 2
# Investigation A (Predator Perception) + Investigation B (Seasonal Effects)
# Classical stats + Linear Regression (full & seasonal) with robust imputation and fallbacks.

import os, sys, warnings, platform
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PRINT_PREFIX = "[Objective 2]"
MERGE_FREQ = "30min"
RANDOM_STATE = 42
FIG_DIR = "figures_obj2"

# Make LR easier to run on small/imbalanced slices
MIN_ROWS_FULL   = 8
MIN_ROWS_SEASON = 8

# ---------- IO ----------
def load_data(df1_path="dataset1.csv", df2_path="dataset2.csv"):
    print(f"{PRINT_PREFIX} Loading datasets…")
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    print(f"  df1 shape: {df1.shape}")
    print(f"  df2 shape: {df2.shape}")
    return df1, df2

# ---------- Validation & Casting ----------
REQ_COLS_DF1 = [
    "start_time","bat_landing_to_food","habit","rat_period_start","rat_period_end",
    "seconds_after_rat_arrival","risk","reward","month","sunset_time","hours_after_sunset","season"
]
REQ_COLS_DF2 = [
    "time","month","hours_after_sunset","bat_landing_number","food_availability","rat_minutes","rat_arrival_number"
]

def as_dt(series, dayfirst=True):
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

def validate_and_cast(df1, df2):
    print(f"{PRINT_PREFIX} Validating columns…")
    miss1 = [c for c in REQ_COLS_DF1 if c not in df1.columns]
    miss2 = [c for c in REQ_COLS_DF2 if c not in df2.columns]
    if miss1: print(f"  WARNING df1 missing columns: {miss1}")
    if miss2: print(f"  WARNING df2 missing columns: {miss2}")

    # Parse datetimes
    for c in ["start_time","rat_period_start","rat_period_end","sunset_time"]:
        if c in df1: df1[c+"_dt"] = as_dt(df1[c])
    if "time" in df2: df2["time_dt"] = as_dt(df2["time"])

    # Ensure binary ints
    for c in ["risk","reward"]:
        if c in df1:
            df1[c] = pd.to_numeric(df1[c], errors="coerce").round().astype("Int64")

    # Season label (keep zoologists' labels if present)
    if "season" in df1:
        df1["season_label"] = df1["season"].astype(str)

    # Derive df2 season for context
    if "month" in df2:
        def m2s(m):
            try: m = int(m)
            except: return np.nan
            return "Winter" if 4 <= m <= 9 else "Spring"
        df2["season_label"] = df2["month"].apply(m2s)

    # Derive hours_after_sunset in df1 if missing/mostly NaN
    if "hours_after_sunset" not in df1.columns or df1["hours_after_sunset"].isna().mean() > 0.5:
        if {"start_time_dt","sunset_time_dt"}.issubset(df1.columns):
            diff = (df1["start_time_dt"] - df1["sunset_time_dt"]).dt.total_seconds()/3600.0
            df1["hours_after_sunset"] = diff
            print(f"{PRINT_PREFIX} Derived hours_after_sunset in df1 from timestamps.")

    return df1, df2

# ---------- Feature Engineering ----------
def engineer_features(df1, df2):
    print(f"{PRINT_PREFIX} Engineering features for Investigation A & B…")
    d1 = df1.copy(); d2 = df2.copy()

    # Rat present at landing
    if not {"start_time_dt","rat_period_start_dt","rat_period_end_dt"}.issubset(d1.columns):
        d1["rat_present_at_landing"] = 0
    else:
        vt = d1[["start_time_dt","rat_period_start_dt","rat_period_end_dt"]].notna().all(axis=1)
        d1["rat_present_at_landing"] = 0
        d1.loc[vt, "rat_present_at_landing"] = (
            (d1.loc[vt, "start_time_dt"] >= d1.loc[vt, "rat_period_start_dt"]) &
            (d1.loc[vt, "start_time_dt"] <= d1.loc[vt, "rat_period_end_dt"])
        ).astype(int)

    # Time since rat left
    if {"start_time_dt","rat_period_end_dt"}.issubset(d1.columns):
        left_mask = d1[["start_time_dt","rat_period_end_dt"]].notna().all(axis=1) & (d1["start_time_dt"] > d1["rat_period_end_dt"])
        d1["time_since_rat_left"] = np.where(
            left_mask,
            (d1["start_time_dt"] - d1["rat_period_end_dt"]).dt.total_seconds(),
            np.nan
        )
    else:
        d1["time_since_rat_left"] = np.nan

    # Foraging efficiency
    if {"bat_landing_to_food","reward"}.issubset(d1.columns):
        d1["foraging_efficiency"] = np.where(
            pd.to_numeric(d1["bat_landing_to_food"], errors="coerce") > 0,
            pd.to_numeric(d1["reward"], errors="coerce") / pd.to_numeric(d1["bat_landing_to_food"], errors="coerce"),
            pd.to_numeric(d1["reward"], errors="coerce")
        )

    # df2 context
    if {"rat_arrival_number","bat_landing_number"}.issubset(d2.columns):
        d2["rat_activity_intensity"] = d2["rat_arrival_number"] / (d2["bat_landing_number"] + 1)
    if "food_availability" in d2:
        d2["food_depletion_rate"] = 1 / (d2["food_availability"] + 0.1)

    # Merge on 30min grid
    if "start_time_dt" in d1 and "time_dt" in d2:
        d1["merge_key"] = d1["start_time_dt"].dt.floor(MERGE_FREQ)
        d2["merge_key"] = d2["time_dt"]
        cols = [c for c in [
            "merge_key","rat_activity_intensity","food_availability",
            "rat_minutes","rat_arrival_number","season_label"
        ] if c in d2.columns]
        d1 = d1.merge(d2[cols].drop_duplicates("merge_key"), on="merge_key", how="left", suffixes=("","_agg"))
    else:
        d1["merge_key"] = pd.NaT

    # Ensure df1 season label exists (fallback from month if needed)
    if "season_label" not in d1.columns or d1["season_label"].isna().all():
        if "month" in d1.columns:
            def m2s1(m):
                try: m = int(m)
                except: return np.nan
                return "Winter" if 4 <= m <= 9 else "Spring"
            d1["season_label"] = d1["month"].apply(m2s1)

    return d1, d2

# ---------- Outlier handling ----------
def cap_outliers_iqr(df, columns, iqr_k=1.5):
    print(f"{PRINT_PREFIX} Outlier capping with IQR×{iqr_k}… (non-destructive)")
    d = df.copy()
    for col in columns:
        if col not in d.columns:
            print(f"  Skip {col} (missing)")
            continue
        s = pd.to_numeric(d[col], errors="coerce")
        s_clean = s.dropna()
        if s_clean.empty:
            print(f"  Skip {col} (no valid data)")
            continue
        q1, q3 = s_clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lb, ub = q1 - iqr_k*iqr, q3 + iqr_k*iqr
        before = ((s < lb) | (s > ub)).sum()
        d[col] = s.clip(lb, ub)
        after = ((d[col] < lb) | (d[col] > ub)).sum()
        print(f"  {col}: {before} outliers → {after} after capping")
    return d

# ---------- Classical statistics ----------
def run_statistics(d1):
    print(f"{PRINT_PREFIX} Running statistical tests…")
    results = {}

    if "rat_present_at_landing" not in d1.columns:
        print(f"{PRINT_PREFIX} No rat_present_at_landing column; skipping classical tests.")
        return results

    d = d1.dropna(subset=["rat_present_at_landing"])
    grp_present = d[d["rat_present_at_landing"]==1]
    grp_absent  = d[d["rat_present_at_landing"]==0]
    print(f"  n Rat present: {len(grp_present)} | n No rat: {len(grp_absent)}")

    # Only run two-sample tests if both groups exist
    if len(grp_present) > 1 and len(grp_absent) > 1:
        # A) Continuous
        for metric in ["bat_landing_to_food","foraging_efficiency"]:
            if metric in d:
                x = pd.to_numeric(grp_present[metric], errors="coerce").dropna()
                y = pd.to_numeric(grp_absent[metric],  errors="coerce").dropna()
                if len(x)>1 and len(y)>1:
                    t, p = stats.ttest_ind(x, y, equal_var=False)
                    d_cohen = (x.mean()-y.mean()) / np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
                    results[f"A_{metric}"] = {"test":"ttest","t":float(t),"p":float(p),"d":float(d_cohen),
                                              "mean_present":float(x.mean()),"mean_absent":float(y.mean())}
        # A) Binary
        for metric in ["risk","reward"]:
            if metric in d:
                ct = pd.crosstab(d["rat_present_at_landing"], d[metric])
                if ct.shape == (2,2):
                    chi2, p, dof, _ = stats.chi2_contingency(ct)
                    results[f"A_{metric}"] = {"test":"chi2","chi2":float(chi2),"p":float(p),"dof":int(dof),"table":ct.to_dict()}
    else:
        print(f"{PRINT_PREFIX} Only one group present — skipping A tests.")

    # B) Seasonal ANOVA (needs ≥2 season groups)
    if "season_label" in d:
        seasons_avail = d["season_label"].dropna().unique()
        if len(seasons_avail) >= 2:
            for metric in ["bat_landing_to_food","foraging_efficiency"]:
                if metric in d:
                    groups = [grp.dropna().values for _, grp in d.groupby("season_label")[metric]]
                    groups = [g for g in groups if len(g)>1]
                    if len(groups) >= 2:
                        F, p = stats.f_oneway(*groups)
                        results[f"B_{metric}_anova"] = {"test":"anova","F":float(F),"p":float(p)}
        else:
            print(f"{PRINT_PREFIX} Only one season present — skipping B ANOVA.")
    return results

# ---------- Visualisations ----------
def ensure_figdir():
    os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def make_plots(d1, d2):
    print(f"{PRINT_PREFIX} Creating figures…")
    ensure_figdir()
    sns.set_theme(style="whitegrid")
    if "rat_present_at_landing" not in d1.columns:
        return
    v = d1.dropna(subset=["rat_present_at_landing"])

    if {"rat_present_at_landing","bat_landing_to_food"}.issubset(v.columns):
        plt.figure(figsize=(8,6))
        sns.boxplot(x="rat_present_at_landing", y="bat_landing_to_food", data=v)
        plt.title("Time to Approach Food by Rat Presence (A)")
        plt.xlabel("Rat present at landing (0=No, 1=Yes)")
        plt.ylabel("Time (s)")
        save_fig(os.path.join(FIG_DIR, "A1_landing_time_by_rat_presence.png"))

    if {"rat_present_at_landing","risk"}.issubset(v.columns):
        prop = v.groupby("rat_present_at_landing")["risk"].mean()
        plt.figure(figsize=(8,6))
        plt.bar(prop.index.astype(str), prop.values)
        plt.title("Risk-taking proportion by Rat Presence (A)")
        plt.xlabel("Rat present at landing (0=No, 1=Yes)")
        plt.ylabel("Proportion risk-taking")
        save_fig(os.path.join(FIG_DIR, "A2_risk_by_rat_presence.png"))

    if {"rat_present_at_landing","reward"}.issubset(v.columns):
        prop = v.groupby("rat_present_at_landing")["reward"].mean()
        plt.figure(figsize=(8,6))
        plt.bar(prop.index.astype(str), prop.values)
        plt.title("Foraging success by Rat Presence (A)")
        plt.xlabel("Rat present at landing (0=No, 1=Yes)")
        plt.ylabel("Proportion successful")
        save_fig(os.path.join(FIG_DIR, "A3_reward_by_rat_presence.png"))

    if {"season_label","bat_landing_to_food","rat_present_at_landing"}.issubset(v.columns):
        if v["season_label"].nunique() >= 2:
            plt.figure(figsize=(10,6))
            sns.boxplot(x="season_label", y="bat_landing_to_food", hue="rat_present_at_landing", data=v)
            plt.title("Time to Approach Food by Season and Rat Presence (B)")
            plt.xlabel("Season (zoologists' labels)")
            plt.ylabel("Time (s)")
            plt.legend(title="Rat present")
            save_fig(os.path.join(FIG_DIR, "B1_time_by_season_and_rat.png"))

    if {"hours_after_sunset","season_label"}.issubset(v.columns) and v["season_label"].nunique() >= 1:
        bins = pd.cut(v["hours_after_sunset"], bins=[0,2,4,6,24],
                      labels=["Early","Mid-Early","Mid-Late","Late"], include_lowest=True)
        tbl = v.assign(time_of_night=bins).groupby(["season_label","time_of_night"]).size().unstack(fill_value=0)
        plt.figure(figsize=(10,6))
        bottom = np.zeros(len(tbl))
        for col in tbl.columns:
            plt.bar(tbl.index.astype(str), tbl[col].values, bottom=bottom, label=col)
            bottom += tbl[col].values
        plt.title("Activity Patterns by Season & Time of Night (counts)")
        plt.xlabel("Season")
        plt.ylabel("Number of landings")
        plt.legend(title="Time of night")
        save_fig(os.path.join(FIG_DIR, "B2_activity_patterns.png"))

    num_cols = [c for c in ["bat_landing_to_food","seconds_after_rat_arrival","risk","reward",
                            "hours_after_sunset","foraging_efficiency","rat_present_at_landing"] if c in v.columns]
    if len(num_cols) >= 2:
        corr = v[num_cols].corr()
        plt.figure(figsize=(9,7))
        sns.heatmap(corr, annot=True, center=0, square=True, fmt=".2f", cbar_kws={"shrink":.8})
        plt.title("Correlation Matrix of Behavioural Variables")
        save_fig(os.path.join(FIG_DIR, "Z_correlation_matrix.png"))

# ---------- Linear Regression ----------
def _ensure_df1_season(df):
    d = df.copy()
    if "season_label" not in d.columns or d["season_label"].astype(str).str.strip().eq("").all() or d["season_label"].isna().all():
        if "month" in d.columns:
            def m2s(m):
                try: m = int(m)
                except: return np.nan
                return "Winter" if 4 <= m <= 9 else "Spring"
            d["season_label"] = d["month"].apply(m2s)
    return d

def build_design_matrix(d1, y_col="bat_landing_to_food"):
    use = _ensure_df1_season(d1.copy())

    # ensure at least one predictor exists
    if "rat_present_at_landing" not in use.columns:
        use["rat_present_at_landing"] = 0

    candidate = [
        "rat_present_at_landing",
        "hours_after_sunset",
        "risk","reward",
        "food_availability","rat_minutes","rat_arrival_number",
        "rat_activity_intensity","time_since_rat_left"
    ]
    preds = [c for c in candidate if c in use.columns]

    keep = [y_col] + preds + (["season_label"] if "season_label" in use.columns else [])
    keep = [c for c in keep if c in use.columns]
    df = use[keep].copy()

    # y must exist & be numeric
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df[~df[y_col].isna()]

    # early exit if no rows
    if df.shape[0] == 0:
        print(f"{PRINT_PREFIX} LR build: y='{y_col}' rows=0 | predictors=0  (empty slice)")
        return np.empty((0, 0)), np.array([]), []

    # dummies for season
    if "season_label" in df.columns:
        df = pd.get_dummies(df, columns=["season_label"], drop_first=True)

    y = df[y_col].astype(float).values
    Xdf = df.drop(columns=[y_col])

    # cast numerics
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # intercept-only fallback if no predictors left
    if Xdf.shape[1] == 0:
        X = np.ones((len(y), 1))
        features = ["intercept_only"]
    else:
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(Xdf)
        features = list(Xdf.columns)

    print(f"{PRINT_PREFIX} LR build: y='{y_col}' rows={len(y)} | predictors={len(features)} (after imputation)")
    return X, y, features

def fit_lr(X, y, test_size=0.3):
    # handle empty
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Empty X or y in fit_lr")

    # if all predictors are constant, use intercept-only
    if X.ndim == 2 and X.shape[1] > 0:
        if np.all(np.nan_to_num(np.std(X, axis=0)) == 0):
            X = np.ones((len(y), 1))

    # adaptive holdout for small n
    ts = 0.3 if len(y) >= 10 else (0.2 if len(y) >= 5 else 0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=RANDOM_STATE)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # <- robust to older stacks

    return {
        "model": model, "scaler": scaler,
        "y_test": y_test, "y_pred": y_pred,
        "r2": r2, "mae": mae, "rmse": rmse
    }

def _lr_plots(res, title):
    ensure_figdir()
    plt.figure(figsize=(7,5))
    plt.scatter(res["y_pred"], res["y_test"] - res["y_pred"], alpha=0.7)
    plt.axhline(0, ls="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted ({title})")
    save_fig(os.path.join(FIG_DIR, f"{title}_residuals.png"))

    plt.figure(figsize=(7,5))
    stats.probplot(res["y_test"] - res["y_pred"], dist="norm", plot=plt)
    plt.title(f"QQ Plot of Residuals ({title})")
    save_fig(os.path.join(FIG_DIR, f"{title}_qq.png"))

def plot_coeffs(features, coefs, fname, title):
    ensure_figdir()
    order = np.argsort(np.abs(coefs))[::-1]
    feats = np.array(features)[order]
    vals  = np.array(coefs)[order]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), feats, rotation=45, ha="right")
    plt.ylabel("Coefficient (standardised units)")
    plt.title(title)
    save_fig(os.path.join(FIG_DIR, fname))

def run_linear_regression(d1):
    results = {}
    y_candidates = ["bat_landing_to_food", "foraging_efficiency", "seconds_after_rat_arrival"]

    # A) Full model
    for y_col in y_candidates:
        try:
            X, y, feats = build_design_matrix(d1, y_col=y_col)
            if len(y) >= MIN_ROWS_FULL and X.shape[1] >= 1:
                res = fit_lr(X, y)
                results["A_full"] = {
                    "y": y_col, "features": feats,
                    "coefficients": res["model"].coef_.tolist(),
                    "r2": res["r2"], "mae": res["mae"], "rmse": res["rmse"]
                }
                title = f"A_full_{y_col}"
                _lr_plots(res, title)
                plot_coeffs(feats, res["model"].coef_, f"{title}_coefficients.png",
                            f"LR Coefficients (A, y={y_col})")
                break
        except Exception as e:
            print(f"{PRINT_PREFIX} A_full error ({y_col}): {e}")

    # B) Seasonal models
    d1s = _ensure_df1_season(d1)
    if "season_label" in d1s.columns:
        for season in ["Winter","Spring"]:
            part = d1s[d1s["season_label"] == season]
            if part.shape[0] == 0:
                print(f"{PRINT_PREFIX} Season {season}: 0 rows — skipping LR.")
                continue
            for y_col in y_candidates:
                try:
                    Xs, ys, fns = build_design_matrix(part, y_col=y_col)
                    if len(ys) >= MIN_ROWS_SEASON and Xs.shape[1] >= 1:
                        res = fit_lr(Xs, ys)
                        results[f"B_{season}"] = {
                            "y": y_col, "features": fns,
                            "coefficients": res["model"].coef_.tolist(),
                            "r2": res["r2"], "mae": res["mae"], "rmse": res["rmse"]
                        }
                        title = f"B_{season}_{y_col}"
                        _lr_plots(res, title)
                        plot_coeffs(fns, res["model"].coef_, f"{title}_coefficients.png",
                                    f"LR Coefficients ({season}, y={y_col})")
                        break
                except Exception as e:
                    print(f"{PRINT_PREFIX} B_{season} error ({y_col}): {e}")
    return results

# ---------- Report ----------
def package_versions():
    try:
        import sklearn
        skl = sklearn.__version__
    except Exception:
        skl = "not available"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "seaborn": sns.__version__,
        "scipy": stats.__class__.__module__.split(".")[0] + " (SciPy)",
        "sklearn": skl
    }

def write_report(d1, d2, stats_res, lr_res, out_path="Objective2_Report.md"):
    print(f"{PRINT_PREFIX} Writing report → {out_path}")
    pv = package_versions()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Objective 2 Report: Bat vs Rat — Investigations A & B\n\n")
        f.write("This report revisits **Investigation A** (predator perception) and adds **Investigation B** (seasonal effects), using both provided datasets.\n\n")

        f.write("## Datasets\n")
        f.write("- `dataset1.csv`: Bat landing events with behavioural annotations.\n")
        f.write("- `dataset2.csv`: 30-min surveillance summaries (rat arrivals, bat landings, food availability).\n\n")

        f.write("## Methods & Assumptions\n")
        f.write("- Times parsed with `dayfirst=True`.\n")
        f.write(f"- Datasets aligned on **{MERGE_FREQ}** blocks by flooring `start_time` and joining to `dataset2.time` (period start).\n")
        f.write("- `risk` and `reward` treated as **binary** (means interpreted as proportions).\n")
        f.write("- `season_label` in `dataset1` is taken as the zoologists' label without remapping; in `dataset2` season was optionally inferred from `month` for descriptive context.\n")
        f.write("- Outliers in continuous variables capped using IQR (non-destructive).\n\n")

        f.write("## Investigation A — Predator Perception (Classical Tests)\n")
        if stats_res:
            for key, val in stats_res.items():
                if key.startswith("A_") and val.get("test")=="ttest":
                    f.write(f"- {key}: t={val['t']:.3f}, p={val['p']:.4f}, Cohen's d={val['d']:.3f}; means (present={val['mean_present']:.2f}, absent={val['mean_absent']:.2f})\n")
                if key.startswith("A_") and val.get("test")=="chi2":
                    f.write(f"- {key}: chi2={val['chi2']:.3f}, p={val['p']:.4f}\n")
        else:
            f.write("- Only one condition present; classical A tests not applicable.\n")
        f.write(f"Figures: `figures_obj2/A1_landing_time_by_rat_presence.png`, `figures_obj2/A2_risk_by_rat_presence.png`, `figures_obj2/A3_reward_by_rat_presence.png`.\n\n")

        f.write("## Investigation B — Seasonal Effects (Classical Tests)\n")
        wrote_b = False
        for key, val in stats_res.items():
            if key.startswith("B_"):
                f.write(f"- {key}: {val['test'].upper()} F={val['F']:.3f}, p={val['p']:.4f}\n")
                wrote_b = True
        if not wrote_b:
            f.write("- Only one season present; ANOVA not applicable.\n")
        f.write(f"Figures: `figures_obj2/B1_time_by_season_and_rat.png`, `figures_obj2/B2_activity_patterns.png`.\n\n")

        f.write("## Linear Regression Modelling (Focal for Assessment 3)\n")
        f.write("### Investigation A (Full Multiple Linear Regression)\n")
        if "A_full" in lr_res:
            a = lr_res["A_full"]
            f.write(f"- Response variable: **{a['y']}**\n")
            f.write(f"- R² = {a['r2']:.3f}, MAE = {a['mae']:.3f}, RMSE = {a['rmse']:.3f}\n")
            top = sorted(zip(a["features"], a["coefficients"]), key=lambda x: abs(x[1]), reverse=True)[:10]
            f.write("- Top coefficients (standardised):\n")
            for name, val in top:
                f.write(f"  - {name}: {val:.3f}\n")
            f.write(f"Figures: `figures_obj2/A_full_{a['y']}_coefficients.png`, `figures_obj2/A_full_{a['y']}_residuals.png`, `figures_obj2/A_full_{a['y']}_qq.png`\n\n")
        else:
            f.write("- Not enough data for LR.\n\n")

        f.write("### Investigation B (Seasonal LR: Winter vs Spring)\n")
        seasons_written = False
        for season in ["Winter","Spring"]:
            key = f"B_{season}"
            if key in lr_res:
                b = lr_res[key]
                seasons_written = True
                f.write(f"- **{season}** | Response: **{b['y']}** → R² = {b['r2']:.3f}, MAE = {b['mae']:.3f}, RMSE = {b['rmse']:.3f}\n")
                top = sorted(zip(b["features"], b["coefficients"]), key=lambda x: abs(x[1]), reverse=True)[:10]
                f.write("  Top coefficients (standardised):\n")
                for name, val in top:
                    f.write(f"    - {name}: {val:.3f}\n")
                f.write(f"  Figures: `figures_obj2/B_{season}_{b['y']}_coefficients.png`, `figures_obj2/B_{season}_{b['y']}_residuals.png`, `figures_obj2/B_{season}_{b['y']}_qq.png`\n")
        if not seasons_written:
            f.write("- Seasonal slices were empty or too small for LR.\n")
        f.write("\n")

        f.write("## Limitations\n")
        f.write("- The uploaded `dataset1.csv` has all rows labelled as rat-present at landing (no control events), limiting classical comparisons.\n")
        f.write("- Seasonal split in `dataset1` may be absent or highly imbalanced; where unavailable, seasonal LR is skipped.\n")
        f.write("- Outlier capping may compress extremes; results should be interpreted with caution.\n")
        f.write("- Observational data; associations are not causal.\n\n")

        f.write("## Reproducibility\n")
        for k, v in pv.items():
            f.write(f"- {k}: {v}\n")

    print(f"{PRINT_PREFIX} Report written: {out_path}")

# ---------- Main ----------
def main():
    df1, df2 = load_data()
    df1, df2 = validate_and_cast(df1, df2)
    d1, d2 = engineer_features(df1, df2)

    # Sanity prints to understand the data balance
    if "rat_present_at_landing" in d1.columns:
        print(f"{PRINT_PREFIX} rat_present_at_landing value_counts:\n{d1['rat_present_at_landing'].value_counts(dropna=False)}")
    if "season_label" in d1.columns:
        print(f"{PRINT_PREFIX} season_label value_counts:\n{d1['season_label'].value_counts(dropna=False)}")

    cont_cols = [c for c in ["bat_landing_to_food","seconds_after_rat_arrival","foraging_efficiency"] if c in d1.columns]
    d1_cap = cap_outliers_iqr(d1, cont_cols, iqr_k=1.5)

    stats_res = run_statistics(d1_cap)
    make_plots(d1_cap, d2)

    lr_res = run_linear_regression(d1_cap)
    write_report(d1_cap, d2, stats_res, lr_res, out_path="Objective2_Report.md")

    print(f"{PRINT_PREFIX} Done. Generated figures in `./{FIG_DIR}` and Objective2_Report.md")

if __name__ == "__main__":
    main()
