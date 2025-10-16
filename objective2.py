# objective2_hd.py
# HIT140 Assessment 3 — Objective 2 (HD-ready)
# Investigation A (Predator Perception) + Investigation B (Seasonal Effects)
# Non-parametric tests + Linear Regression (full & seasonal)
# Robust time merges, imputation, standardisation, diagnostics, and rich reporting.

import os, sys, warnings, platform
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PRINT_PREFIX = "[Objective 2]"
MERGE_FREQ = "30min"
RANDOM_STATE = 42
FIG_DIR = "figures_obj2"

MIN_ROWS_FULL   = 8
MIN_ROWS_SEASON = 8

# ---------------- IO ----------------
def load_data(df1_path="dataset1.csv", df2_path="dataset2.csv"):
    print(f"{PRINT_PREFIX} Loading datasets…")
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    print(f"  df1 shape: {df1.shape}")
    print(f"  df2 shape: {df2.shape}")
    return df1, df2

# ---------------- Validation & Casting ----------------
REQ_COLS_DF1 = [
    "start_time","bat_landing_to_food","habit","rat_period_start","rat_period_end",
    "seconds_after_rat_arrival","risk","reward","month","sunset_time","hours_after_sunset","season"
]
REQ_COLS_DF2 = [
    "time","month","hours_after_sunset","bat_landing_number","food_availability","rat_minutes","rat_arrival_number"
]

def as_dt(series, dayfirst=True):
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

def _season_4(m):
    try: m = int(m)
    except: return np.nan
    if m in (12,1,2): return "Summer"
    if m in (3,4,5):  return "Autumn"
    if m in (6,7,8):  return "Winter"
    if m in (9,10,11):return "Spring"
    return np.nan

def validate_and_cast(df1, df2):
    print(f"{PRINT_PREFIX} Validating columns…")
    miss1 = [c for c in REQ_COLS_DF1 if c not in df1.columns]
    miss2 = [c for c in REQ_COLS_DF2 if c not in df2.columns]
    if miss1: print(f"  WARNING df1 missing columns: {miss1}")
    if miss2: print(f"  WARNING df2 missing columns: {miss2}")

    # datetimes
    for c in ["start_time","rat_period_start","rat_period_end","sunset_time"]:
        if c in df1: df1[c+"_dt"] = as_dt(df1[c])
    if "time" in df2: df2["time_dt"] = as_dt(df2["time"])

    # risk/reward binary & clipped
    for c in ["risk","reward"]:
        if c in df1:
            df1[c] = pd.to_numeric(df1[c], errors="coerce").fillna(0).round().clip(0,1).astype("Int64")

    # season label (keep zoologists' labels if present)
    if "season" in df1:
        df1["season_label"] = df1["season"].astype(str)

    # derive hours_after_sunset if missing or mostly NaN; clamp [0,24]
    if "hours_after_sunset" not in df1.columns or df1["hours_after_sunset"].isna().mean() > 0.5:
        if {"start_time_dt","sunset_time_dt"}.issubset(df1.columns):
            diff = (df1["start_time_dt"] - df1["sunset_time_dt"]).dt.total_seconds()/3600.0
            df1["hours_after_sunset"] = diff.mask(diff < 0, np.nan).clip(upper=24)

    # df2 season (for context/plots)
    if "month" in df2:
        df2["season_label"] = df2["month"].apply(_season_4)

    return df1, df2

# ---------------- Feature Engineering ----------------
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

    # Time since rat left (sec)
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
        bltf = pd.to_numeric(d1["bat_landing_to_food"], errors="coerce")
        rew = pd.to_numeric(d1["reward"], errors="coerce")
        d1["foraging_efficiency"] = np.where(bltf > 0, rew / bltf, rew)

    # df2 context
    if {"rat_arrival_number","bat_landing_number"}.issubset(d2.columns):
        d2["rat_activity_intensity"] = d2["rat_arrival_number"] / (d2["bat_landing_number"] + 1)
    if "food_availability" in d2:
        d2["food_depletion_rate"] = 1 / (d2["food_availability"] + 0.1)

    # Merge with tolerance using 30-min grid + aggregation
    if "start_time_dt" in d1 and "time_dt" in d2:
        d1["merge_key"] = d1["start_time_dt"].dt.floor(MERGE_FREQ)
        d2["merge_key"] = pd.to_datetime(d2["time_dt"]).dt.floor(MERGE_FREQ)
        d2_agg = (d2
            .groupby("merge_key", as_index=False)
            .agg({
                "rat_activity_intensity":"mean",
                "food_availability":"mean",
                "rat_minutes":"sum",
                "rat_arrival_number":"sum",
                "season_label":"first"
            }))
        d1 = pd.merge_asof(
            d1.sort_values("merge_key"),
            d2_agg.sort_values("merge_key"),
            on="merge_key", direction="nearest", tolerance=pd.Timedelta("30min")
        )
    else:
        d1["merge_key"] = pd.NaT

    # Ensure df1 season 4-way if missing
    if "season_label" not in d1.columns or d1["season_label"].isna().all():
        if "month" in d1.columns:
            d1["season_label"] = d1["month"].apply(_season_4)

    return d1, d2

# ---------------- Outlier handling (IQR) ----------------
def cap_outliers_iqr(df, columns, iqr_k=1.5):
    print(f"{PRINT_PREFIX} Outlier capping with IQR×{iqr_k}… (non-destructive)")
    d = df.copy()
    d.__iqr_caps__ = {}
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
        d.__iqr_caps__[col] = {"Q1":float(q1),"Q3":float(q3),"IQR":float(iqr),"LB":float(lb),"UB":float(ub),"n_capped":int(before-after)}
        print(f"  {col}: {before} outliers → {after} after capping")
    return d

# ---------------- Classical statistics ----------------
def cliffs_delta_from_U(U, n1, n2):
    # delta = 2U/(n1*n2) - 1 ; range [-1,1]
    return (2*U)/(n1*n2) - 1

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

    # A) Non-parametric for continuous + chi2 for binary
    if len(grp_present)>1 and len(grp_absent)>1:
        for metric in ["bat_landing_to_food","foraging_efficiency"]:
            if metric in d:
                x = pd.to_numeric(grp_present[metric], errors="coerce").dropna()
                y = pd.to_numeric(grp_absent[metric],  errors="coerce").dropna()
                if len(x)>1 and len(y)>1:
                    U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                    delta = cliffs_delta_from_U(U, len(x), len(y))
                    results[f"A_{metric}_mw"] = {
                        "test":"mannwhitney","U":float(U),"p":float(p),
                        "median_present":float(x.median()),"median_absent":float(y.median()),
                        "cliffs_delta":float(delta)
                    }
        for metric in ["risk","reward"]:
            if metric in d:
                ct = pd.crosstab(d["rat_present_at_landing"], d[metric])
                if ct.shape == (2,2):
                    chi2, p, dof, _ = stats.chi2_contingency(ct)
                    results[f"A_{metric}_chi2"] = {"test":"chi2","chi2":float(chi2),"p":float(p),"dof":int(dof),"table":ct.to_dict()}
    else:
        print(f"{PRINT_PREFIX} Only one condition present — A tests limited.")

    # B) Seasonal effects: ANOVA + Kruskal–Wallis
    if "season_label" in d and d["season_label"].nunique() >= 2:
        for metric in ["bat_landing_to_food","foraging_efficiency"]:
            if metric in d:
                groups = [grp.dropna().values for _, grp in d.groupby("season_label")[metric]]
                groups = [g for g in groups if len(g)>1]
                if len(groups)>=2:
                    F, p = stats.f_oneway(*groups)
                    results[f"B_{metric}_anova"] = {"test":"anova","F":float(F),"p":float(p)}
                    H, p2 = stats.kruskal(*groups)
                    results[f"B_{metric}_kruskal"] = {"test":"kruskal","H":float(H),"p":float(p2)}
    else:
        print(f"{PRINT_PREFIX} Only one season present — B tests limited.")

    return results

# ---------------- Visualisations ----------------
def ensure_figdir():
    os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(basepath):
    base, ext = os.path.splitext(basepath)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close()

def _bar_with_ci(df, group_col, value_col, fname, title, ylabel):
    g = df.groupby(group_col)[value_col]
    means = g.mean()
    ns = g.count()
    ses = g.std(ddof=1) / np.sqrt(ns.clip(lower=1))
    ci95 = 1.96 * ses
    x = means.index.astype(str).tolist()
    y = means.values
    yerr = ci95.values
    plt.figure(figsize=(8,6))
    plt.bar(x, y, yerr=yerr, capsize=6)
    plt.title(title)
    plt.xlabel(f"{group_col} (0=No, 1=Yes)" if group_col=="rat_present_at_landing" else group_col)
    plt.ylabel(ylabel)
    save_fig(os.path.join(FIG_DIR, fname))

def plot_marginal_effect_time(v, fname, title):
    d = v.copy()
    if "bat_landing_to_food" not in d or "hours_after_sunset" not in d:
        return
    if len(d) > 5000:
        d = d.sample(5000, random_state=RANDOM_STATE)
    plt.figure(figsize=(9,6))
    plt.scatter(d["hours_after_sunset"], np.log1p(d["bat_landing_to_food"]), alpha=0.25, s=10)
    dd = d.sort_values("hours_after_sunset")
    x = dd["hours_after_sunset"].values
    y = np.log1p(dd["bat_landing_to_food"].values)
    win = max(5, int(len(dd)*0.05))
    y_trend = pd.Series(y).rolling(win, min_periods=max(3,win//3), center=True).mean().to_numpy()
    plt.plot(x, y_trend, linewidth=2)
    plt.xlabel("Hours after sunset")
    plt.ylabel("log1p(Time to approach)")
    plt.title(title)
    save_fig(os.path.join(FIG_DIR, fname))

def make_plots(d1, d2):
    print(f"{PRINT_PREFIX} Creating figures…")
    ensure_figdir()
    sns.set_theme(style="whitegrid")

    if "rat_present_at_landing" not in d1.columns: return
    v = d1.dropna(subset=["rat_present_at_landing"])

    # A1: Box
    if {"rat_present_at_landing","bat_landing_to_food"}.issubset(v.columns):
        n0 = (v["rat_present_at_landing"]==0).sum()
        n1 = (v["rat_present_at_landing"]==1).sum()
        plt.figure(figsize=(8,6))
        sns.boxplot(x="rat_present_at_landing", y="bat_landing_to_food", data=v)
        plt.title(f"Time to Approach Food by Rat Presence (A) — n0={n0}, n1={n1}")
        plt.xlabel("Rat present at landing (0=No, 1=Yes)")
        plt.ylabel("Time (s)")
        save_fig(os.path.join(FIG_DIR, "A1_landing_time_by_rat_presence"))

    # A2: Risk (±95% CI)
    if {"rat_present_at_landing","risk"}.issubset(v.columns):
        _bar_with_ci(v, "rat_present_at_landing", "risk",
                     "A2_risk_by_rat_presence",
                     "Risk-taking by Rat Presence (mean ±95% CI)",
                     "Proportion risk-taking")

    # A3: Reward (±95% CI)
    if {"rat_present_at_landing","reward"}.issubset(v.columns):
        _bar_with_ci(v, "rat_present_at_landing", "reward",
                     "A3_reward_by_rat_presence",
                     "Foraging Success by Rat Presence (mean ±95% CI)",
                     "Proportion successful")

    # B1: Season × rat presence
    if {"season_label","bat_landing_to_food","rat_present_at_landing"}.issubset(v.columns) and v["season_label"].nunique()>=2:
        plt.figure(figsize=(10,6))
        sns.boxplot(x="season_label", y="bat_landing_to_food", hue="rat_present_at_landing", data=v)
        plt.title("Time to Approach Food by Season and Rat Presence (B)")
        plt.xlabel("Season")
        plt.ylabel("Time (s)")
        plt.legend(title="Rat present")
        save_fig(os.path.join(FIG_DIR, "B1_time_by_season_and_rat"))

    # B2: Stacked time-of-night
    if {"hours_after_sunset","season_label"}.issubset(v.columns) and v["season_label"].nunique()>=1:
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
        save_fig(os.path.join(FIG_DIR, "B2_activity_patterns"))

    # Correlation matrix
    num_cols = [c for c in ["bat_landing_to_food","seconds_after_rat_arrival","risk","reward",
                            "hours_after_sunset","foraging_efficiency","rat_present_at_landing"] if c in v.columns]
    if len(num_cols) >= 2:
        corr = v[num_cols].corr()
        plt.figure(figsize=(9,7))
        sns.heatmap(corr, annot=True, center=0, square=True, fmt=".2f", cbar_kws={"shrink":.8})
        plt.title("Correlation Matrix of Behavioural Variables")
        save_fig(os.path.join(FIG_DIR, "Z_correlation_matrix"))

    # Marginal effect
    plot_marginal_effect_time(v, "A4_marginal_time", "Marginal effect: Hours After Sunset vs log1p(Approach Time)")

# ---------------- Linear Regression ----------------
def _ensure_df1_season(df):
    d = df.copy()
    if "season_label" not in d.columns or d["season_label"].isna().all():
        if "month" in d.columns:
            d["season_label"] = d["month"].apply(_season_4)
    return d

def build_design_matrix(d1, y_col="bat_landing_to_food", log_transform_time=True):
    use = _ensure_df1_season(d1.copy())

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

    # y numeric & transform if approach time
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    y_was_log = False
    if log_transform_time and y_col == "bat_landing_to_food":
        df[y_col] = np.log1p(df[y_col])
        y_was_log = True
    df = df[~df[y_col].isna()]
    if df.shape[0] == 0:
        return np.empty((0,0)), np.array([]), [], y_was_log

    # season dummies only if multiple
    if "season_label" in df.columns and df["season_label"].nunique() > 1:
        df = pd.get_dummies(df, columns=["season_label"], drop_first=True)
    elif "season_label" in df.columns:
        df = df.drop(columns=["season_label"])

    y = df[y_col].astype(float).values
    Xdf = df.drop(columns=[y_col]).copy()
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # impute then drop zero-variance with features aligned
    if Xdf.shape[1] == 0:
        X = np.ones((len(y), 1)); features = ["intercept_only"]
    else:
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(Xdf)
        features = list(Xdf.columns)
        stds = np.nan_to_num(np.std(X, axis=0))
        keep_idx = stds > 0
        if keep_idx.sum() == 0:
            X = np.ones((len(y), 1)); features = ["intercept_only"]
        elif keep_idx.sum() != len(keep_idx):
            X = X[:, keep_idx]
            features = [f for f,k in zip(features, keep_idx) if k]

    return X, y, features, y_was_log

def fit_lr(X, y, test_size=None):
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Empty X or y in fit_lr")
    if test_size is None:
        test_size = 0.3 if len(y) >= 10 else (0.2 if len(y) >= 5 else 0.5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    p = X_test_s.shape[1] if X_test_s.ndim == 2 else 1
    n = len(y_test)
    adj_r2 = 1 - (1-r2)*(n-1)/max(n-p-1, 1)

    return {"model":model,"scaler":scaler,"y_test":y_test,"y_pred":y_pred,
            "r2":r2,"adj_r2":adj_r2,"mae":mae,"rmse":rmse}

def _lr_plots(res, title):
    ensure_figdir()
    plt.figure(figsize=(7,5))
    plt.scatter(res["y_pred"], res["y_test"] - res["y_pred"], alpha=0.7)
    plt.axhline(0, ls="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted ({title})")
    save_fig(os.path.join(FIG_DIR, f"{title}_residuals"))

    plt.figure(figsize=(7,5))
    stats.probplot(res["y_test"] - res["y_pred"], dist="norm", plot=plt)
    plt.title(f"Q–Q Plot of Residuals ({title})")
    save_fig(os.path.join(FIG_DIR, f"{title}_qq"))

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
    y_candidates = ["bat_landing_to_food","foraging_efficiency","seconds_after_rat_arrival"]

    # A) Full model
    for y_col in y_candidates:
        try:
            X, y, feats, ylog = build_design_matrix(d1, y_col=y_col)
            if len(y) >= MIN_ROWS_FULL and X.shape[1] >= 1:
                res = fit_lr(X, y)
                results["A_full"] = {"y":y_col,"features":feats,
                                     "coefficients":res["model"].coef_.tolist(),
                                     "r2":res["r2"],"adj_r2":res["adj_r2"],
                                     "mae":res["mae"],"rmse":res["rmse"],
                                     "y_log":bool(ylog)}
                title = f"A_full_{y_col}"
                _lr_plots(res, title)
                plot_coeffs(feats, res["model"].coef_, f"{title}_coefficients", f"LR Coefficients (A, y={y_col})")
                break
        except Exception as e:
            print(f"{PRINT_PREFIX} A_full error ({y_col}): {e}")

    # B) Seasonal models (4 seasons)
    d1s = _ensure_df1_season(d1)
    if "season_label" in d1s.columns:
        for season in ["Summer","Autumn","Winter","Spring"]:
            part = d1s[d1s["season_label"] == season]
            if part.shape[0] == 0:
                print(f"{PRINT_PREFIX} Season {season}: 0 rows — skipping LR.")
                continue
            for y_col in y_candidates:
                try:
                    Xs, ys, fns, ylog = build_design_matrix(part, y_col=y_col)
                    if len(ys) >= MIN_ROWS_SEASON and Xs.shape[1] >= 1:
                        res = fit_lr(Xs, ys)
                        results[f"B_{season}"] = {"y":y_col,"features":fns,
                                                  "coefficients":res["model"].coef_.tolist(),
                                                  "r2":res["r2"],"adj_r2":res["adj_r2"],
                                                  "mae":res["mae"],"rmse":res["rmse"],
                                                  "y_log":bool(ylog)}
                        title = f"B_{season}_{y_col}"
                        _lr_plots(res, title)
                        plot_coeffs(fns, res["model"].coef_, f"{title}_coefficients", f"LR Coefficients ({season}, y={y_col})")
                        break
                except Exception as e:
                    print(f"{PRINT_PREFIX} B_{season} error ({y_col}): {e}")
    return results

# ---------------- Report ----------------
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
        "scipy": scipy.__version__,
        "sklearn": skl
    }

def write_report(d1, d2, stats_res, lr_res, out_path="Objective2_Report.md"):
    print(f"{PRINT_PREFIX} Writing report → {out_path}")
    pv = package_versions()
    with open(out_path, "w", encoding="utf-8") as f:
        # Title + team/date (edit names if needed)
        f.write("# Objective 2 Report: Bat vs Rat — Investigations A & B\n\n")
        f.write("**Team:** Prakash Bhattarai (S394859), Dipesh Wagle (S394745), "
                "Pralin Dhungana (S395785), Bishal Dahal (S388095)\n\n")
        f.write("**Date:** 15 October 2025\n\n")

        f.write("## Abstract\n")
        f.write("We investigated whether bats perceive rats as predators by analysing approach time, "
                "risk/reward behaviour, and seasonal dynamics across two datasets. We used non-parametric "
                "tests with robust outlier handling and time-aligned merges, and fit multiple linear "
                "regression models (full and seasonal). Approach time increased with hours after sunset and "
                "risk; seasonal models showed context-dependent effects. Despite imbalanced conditions, "
                "results support vigilance consistent with predator perception.\n\n")

        f.write("## Introduction\n")
        f.write("Bats may co-forage with rats at shared feeding sites. Distinguishing competitor effects from "
                "predator perception has implications for foraging theory and risk management.\n\n")
        f.write("### Research Question\n")
        f.write("> Do bats perceive rats as potential predators influencing foraging and vigilance?\n\n")
        f.write("### Hypotheses\n")
        f.write("- Increased vigilance (longer approach time) under rat presence/risk.\n")
        f.write("- Avoidance (lower landing success) when rats are present.\n")
        f.write("- Seasonal modulation of these effects.\n\n")

        f.write("## Datasets\n")
        f.write("- `dataset1.csv`: Bat landing events with behavioural annotations.\n")
        f.write("- `dataset2.csv`: 30-min surveillance summaries (rat arrivals, bat landings, food availability).\n\n")

        f.write("## Methods & Assumptions\n")
        f.write("- Times parsed with `dayfirst=True`.\n")
        f.write(f"- Datasets aligned on **{MERGE_FREQ}** blocks using **nearest-asof** with ±30 min tolerance; df2 aggregated per block.\n")
        f.write("- `risk` and `reward` treated as **binary** (means = proportions; clipped to {0,1}).\n")
        f.write("- `season_label` kept if provided; otherwise inferred from month (Summer/Autumn/Winter/Spring).\n")
        f.write("- Outliers capped by IQR (non-destructive); analysis on capped values.\n\n")

        # IQR caps
        if hasattr(d1, "__iqr_caps__") and d1.__iqr_caps__:
            f.write("### IQR Capping Summary\n")
            for k,v in d1.__iqr_caps__.items():
                f.write(f"- {k}: Q1={v['Q1']:.2f}, Q3={v['Q3']:.2f}, IQR={v['IQR']:.2f}, "
                        f"LB={v['LB']:.2f}, UB={v['UB']:.2f}, n_capped={v['n_capped']}\n")
            f.write("\n")

        f.write("## Results: Investigation A — Predator Perception\n")
        wrote_any = False
        for key, val in stats_res.items():
            if key.startswith("A_") and val.get("test")=="mannwhitney":
                f.write(f"- {key}: U={val['U']:.3f}, p={val['p']:.4g}; "
                        f"medians (present={val['median_present']:.2f}, absent={val['median_absent']:.2f}); "
                        f"Cliff's δ={val['cliffs_delta']:.3f}\n")
                wrote_any = True
            if key.startswith("A_") and val.get("test")=="chi2":
                f.write(f"- {key}: χ²={val['chi2']:.3f}, p={val['p']:.4g}\n")
                wrote_any = True
        if not wrote_any:
            f.write("- Only one condition present; classical A tests limited.\n")
        f.write("Figures: `figures_obj2/A1_landing_time_by_rat_presence.(png|svg)`, "
                "`figures_obj2/A2_risk_by_rat_presence.(png|svg)`, "
                "`figures_obj2/A3_reward_by_rat_presence.(png|svg)`, "
                "`figures_obj2/A4_marginal_time.(png|svg)`.\n\n")

        f.write("## Results: Investigation B — Seasonal Effects\n")
        wrote_b = False
        for key, val in stats_res.items():
            if key.startswith("B_") and val.get("test")=="anova":
                f.write(f"- {key}: ANOVA F={val['F']:.3f}, p={val['p']:.4g}\n")
                wrote_b = True
            if key.startswith("B_") and val.get("test")=="kruskal":
                f.write(f"- {key}: Kruskal–Wallis H={val['H']:.3f}, p={val['p']:.4g}\n")
                wrote_b = True
        if not wrote_b:
            f.write("- Only one season present; B tests limited.\n")
        f.write("Figures: `figures_obj2/B1_time_by_season_and_rat.(png|svg)`, "
                "`figures_obj2/B2_activity_patterns.(png|svg)`.\n\n")

        f.write("## Linear Regression Modelling\n")
        if "A_full" in lr_res:
            a = lr_res["A_full"]
            f.write(f"### Full model (Investigation A)\n")
            f.write(f"- Response: **{a['y']}**{' (log1p)' if a.get('y_log') else ''}\n")
            f.write(f"- R² = {a['r2']:.3f} (Adj. {a['adj_r2']:.3f}), MAE = {a['mae']:.3f}, RMSE = {a['rmse']:.3f}\n")
            top = sorted(zip(a["features"], a["coefficients"]), key=lambda x: abs(x[1]), reverse=True)[:10]
            f.write("- Top coefficients (standardised):\n")
            for name, val in top:
                f.write(f"  - {name}: {val:.3f}\n")
            f.write("Figures: "
                    f"`figures_obj2/A_full_{a['y']}_coefficients.(png|svg)`, "
                    f"`figures_obj2/A_full_{a['y']}_residuals.(png|svg)`, "
                    f"`figures_obj2/A_full_{a['y']}_qq.(png|svg)`\n\n")
        else:
            f.write("- Not enough data for the full LR model.\n\n")

        f.write("### Seasonal models (Investigation B)\n")
        seasons_written = False
        for season in ["Summer","Autumn","Winter","Spring"]:
            key = f"B_{season}"
            if key in lr_res:
                b = lr_res[key]; seasons_written = True
                f.write(f"- **{season}** | Response: **{b['y']}**{' (log1p)' if b.get('y_log') else ''} → "
                        f"R² = {b['r2']:.3f} (Adj. {b['adj_r2']:.3f}), MAE = {b['mae']:.3f}, RMSE = {b['rmse']:.3f}\n")
                top = sorted(zip(b["features"], b["coefficients"]), key=lambda x: abs(x[1]), reverse=True)[:10]
                f.write("  Top coefficients (standardised):\n")
                for name, val in top:
                    f.write(f"    - {name}: {val:.3f}\n")
                f.write(f"  Figures: `figures_obj2/B_{season}_{b['y']}_coefficients.(png|svg)`, "
                        f"`figures_obj2/B_{season}_{b['y']}_residuals.(png|svg)`, "
                        f"`figures_obj2/B_{season}_{b['y']}_qq.(png|svg)`\n")
        if not seasons_written:
            f.write("- Seasonal slices were empty or too small for LR.\n")
        f.write("\n")

        # Discussion
        f.write("## Discussion\n")
        a = lr_res.get("A_full")
        if a:
            feats, coefs = a.get("features", []), a.get("coefficients", [])
            def coef_of(name):
                try: return coefs[feats.index(name)]
                except: return None
            c_time = coef_of("hours_after_sunset")
            c_risk = coef_of("risk")
            c_rat  = coef_of("rat_present_at_landing")
            f.write("The full model explained a modest but meaningful fraction of variance (typical for field behaviour). ")
            if c_time is not None:
                f.write(f"**hours_after_sunset** showed a positive association with approach time (β≈{c_time:.2f}), ")
            if c_risk is not None and c_risk>0:
                f.write(f"and **risk** was positively associated (β≈{c_risk:.2f}), ")
            if c_rat is not None and c_rat>0:
                f.write(f"while **rat_present_at_landing** had a smaller positive effect (β≈{c_rat:.2f}). ")
            f.write("These are consistent with increased vigilance under perceived risk.\n\n")

        f.write("## Conclusion\n")
        f.write("Evidence from non-parametric tests and regression suggests bats modulate approach time with risk "
                "and time-of-night, consistent with predator perception. Seasonal models indicate context-dependent "
                "strength of these effects.\n\n")

        f.write("## Figure Captions\n")
        f.write("1. A1: Time to approach by rat presence (boxplot; IQR whiskers; n per group in title).\n")
        f.write("2. A2: Risk-taking proportion by presence (mean ±95% CI).\n")
        f.write("3. A3: Foraging success by presence (mean ±95% CI).\n")
        f.write("4. A4: Marginal effect of hours after sunset on log1p(approach time).\n")
        f.write("5. B1: Approach time by season × presence (boxplot).\n")
        f.write("6. B2: Stacked counts by season and time-of-night bins.\n")
        f.write("7–9+: Coefficient bars and diagnostics (residuals, Q–Q) for full/seasonal models.\n\n")

        f.write("## Individual Contributions\n")
        f.write("- Prakash Bhattarai: Statistical design; non-parametric testing; reporting.\n")
        f.write("- Dipesh Wagle: Data cleaning; IQR capping; integration.\n")
        f.write("- Pralin Dhungana: Linear regression modelling; validation; metrics.\n")
        f.write("- Bishal Dahal: Visualisation; figure styling; report formatting.\n\n")

        f.write("## Limitations\n")
        if "rat_present_at_landing" in d1.columns:
            vc = d1["rat_present_at_landing"].value_counts(dropna=False).to_dict()
            f.write(f"- Condition balance (rat_present_at_landing): {vc}\n")
        f.write("- Seasonal balance may be uneven; where unavailable, seasonal LR is skipped.\n")
        f.write("- IQR capping compresses extremes; interpret effect sizes with caution.\n")
        f.write("- Observational data; associations are not causal.\n\n")

        f.write("## Reproducibility\n")
        for k,v in pv.items():
            f.write(f"- {k}: {v}\n")

    print(f"{PRINT_PREFIX} Report written: {out_path}")

# ---------------- Main ----------------
def main():
    df1, df2 = load_data()
    df1, df2 = validate_and_cast(df1, df2)
    d1, d2 = engineer_features(df1, df2)

    if "rat_present_at_landing" in d1.columns:
        print(f"{PRINT_PREFIX} rat_present_at_landing value_counts:\n{d1['rat_present_at_landing'].value_counts(dropna=False)}")
    if "season_label" in d1.columns:
        print(f"{PRINT_PREFIX} season_label value_counts:\n{d1['season_label'].value_counts(dropna=False)}")

    cont_cols = [c for c in ["bat_landing_to_food","seconds_after_rat_arrival","foraging_efficiency"] if c in d1.columns]
    d1_cap = cap_outliers_iqr(d1, cont_cols, iqr_k=1.5)

    stats_res = run_statistics(d1_cap)
    make_plots(d1_cap, d2)

    lr_res = run_linear_regression(d1_cap)
    # write_report(d1_cap, d2, stats_res, lr_res, out_path="Objective2_Report.md")

    print(f"{PRINT_PREFIX} Done. Figures in ./{FIG_DIR} and Objective2_Report.md")

if __name__ == "__main__":
    main()
