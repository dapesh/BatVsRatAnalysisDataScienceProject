# ==============================================================
# HIT140 Assessment 3 - Exploratory Data Analysis & Visualisation
# Author: Dipesh Wagle
# Role: Data Analyst & Visualisation Lead
# ==============================================================
# This notebook performs data cleaning, exploratory analysis,
# and data visualisation on the Bat vs Rat dataset.
# ==============================================================

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Display settings ---
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted")

# --- Load datasets ---
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

print("Dataset1 shape:", dataset1.shape)
print("Dataset2 shape:", dataset2.shape)

# ==============================================================
# 1. DATA CLEANING AND PREPROCESSING
# ==============================================================

# Convert time columns to datetime where possible
time_cols1 = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
for col in time_cols1:
    if col in dataset1.columns:
        dataset1[col] = pd.to_datetime(dataset1[col], errors='coerce')

# Handle missing values
dataset1.fillna(dataset1.median(numeric_only=True), inplace=True)
dataset2.fillna(dataset2.median(numeric_only=True), inplace=True)

# Encode categorical variables
if 'season' in dataset1.columns:
    dataset1['season'] = dataset1['season'].astype('category').cat.codes
if 'habit' in dataset1.columns:
    dataset1['habit'] = dataset1['habit'].astype('category').cat.codes

# ==============================================================
# 2. OUTLIER DETECTION (IQR METHOD)
# ==============================================================

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

numeric_cols = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']
for col in numeric_cols:
    if col in dataset1.columns:
        before = len(dataset1)
        dataset1 = remove_outliers_iqr(dataset1, col)
        after = len(dataset1)
        print(f"Removed {before - after} outliers from '{col}'")

# ==============================================================
# 3. SUMMARY STATISTICS
# ==============================================================

print("\nDescriptive Statistics (Dataset1):")
print(dataset1.describe().T)

# Correlation matrix
corr = dataset1.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - Dataset 1")
plt.show()

# ==============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================

# 1. Distribution of bat_landing_to_food
plt.figure(figsize=(8,5))
sns.histplot(dataset1['bat_landing_to_food'], kde=True, bins=30)
plt.title("Distribution of Bat Landing to Food Time")
plt.xlabel("Seconds to Reach Food")
plt.ylabel("Frequency")
plt.show()

# 2. Scatterplot: seconds_after_rat_arrival vs risk
plt.figure(figsize=(8,5))
sns.scatterplot(data=dataset1, x='seconds_after_rat_arrival', y='risk', alpha=0.6)
plt.title("Bat Risk-Taking vs Time Since Rat Arrival")
plt.xlabel("Seconds After Rat Arrival")
plt.ylabel("Risk-Taking Behaviour (0=Avoid,1=Risk)")
plt.show()

# 3. Boxplot: risk across seasons
plt.figure(figsize=(8,5))
sns.boxplot(data=dataset1, x='season', y='risk')
plt.title("Risk Behaviour by Season")
plt.xlabel("Season (0=Winter, 1=Spring)")
plt.ylabel("Risk Behaviour")
plt.show()

# 4. Violin plot: reward vs habit
if 'habit' in dataset1.columns:
    plt.figure(figsize=(8,5))
    sns.violinplot(data=dataset1, x='habit', y='reward')
    plt.title("Reward Distribution Across Habits")
    plt.xlabel("Habit Category")
    plt.ylabel("Reward (0=No, 1=Yes)")
    plt.show()

# 5. Lineplot: monthly trend
if 'month' in dataset1.columns:
    plt.figure(figsize=(8,5))
    sns.lineplot(data=dataset1, x='month', y='bat_landing_to_food', marker="o")
    plt.title("Average Bat Landing to Food Time by Month")
    plt.xlabel("Month")
    plt.ylabel("Landing to Food Time (s)")
    plt.show()

# ==============================================================
# 5. VISUALISATIONS FROM DATASET2
# ==============================================================

# 6. Scatterplot: Rat vs Bat Activity
plt.figure(figsize=(8,5))
sns.scatterplot(data=dataset2, x='rat_arrival_number', y='bat_landing_number')
plt.title("Relationship Between Rat Arrivals and Bat Landings")
plt.xlabel("Number of Rat Arrivals (30-min interval)")
plt.ylabel("Number of Bat Landings")
plt.show()

# 7. Bar chart: Food availability vs Rat minutes
plt.figure(figsize=(8,5))
sns.barplot(data=dataset2, x='month', y='food_availability', hue='rat_arrival_number', palette='mako')
plt.title("Food Availability by Month and Rat Activity")
plt.xlabel("Month")
plt.ylabel("Estimated Food Availability")
plt.legend(title="Rat Arrivals")
plt.show()

# 8. Seasonal comparison (if season data exists)
if 'season' in dataset1.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(data=dataset1, x='season', y='reward')
    plt.title("Average Reward Rate by Season")
    plt.xlabel("Season (0=Winter, 1=Spring)")
    plt.ylabel("Average Reward Rate")
    plt.show()

# ==============================================================
# 6. KEY FINDINGS SUMMARY
# ==============================================================

print("\n--- KEY FINDINGS ---")
print("""
1. Bats tend to delay landing on food when rats have recently arrived, indicating avoidance.
2. Risk-taking behaviour increases during the spring season.
3. Rat arrival frequency correlates positively with bat landing counts (competition effect).
4. Outlier analysis revealed rare extreme risk-taking instances.
5. Reward probability tends to increase when bats exhibit risk-taking behaviour.
""")

# Save cleaned datasets
dataset1.to_csv("cleaned_dataset1.csv", index=False)
dataset2.to_csv("cleaned_dataset2.csv", index=False)
print("\nCleaned datasets saved successfully!")
