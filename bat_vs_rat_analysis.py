import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# DATA LOADING AND PREPROCESSING

print("=" * 60)
print("BAT VS. RAT: FORAGING BEHAVIOR ANALYSIS")
print("=" * 60)

# Load both datasets
print("\n1. LOADING DATASETS...")
df1 = pd.read_csv('dataset1.csv', parse_dates=['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time'], dayfirst=True)
df2 = pd.read_csv('dataset2.csv', parse_dates=['time'], dayfirst=True)

print(f"Dataset 1 shape: {df1.shape}")
print(f"Dataset 2 shape: {df2.shape}")

# Basic dataset information
print("\nDataset 1 columns:")
print(df1.columns.tolist())
print("\nDataset 2 columns:")
print(df2.columns.tolist())

# DATA CLEANING AND FEATURE ENGINEERING

print("\n2. DATA CLEANING AND FEATURE ENGINEERING...")

# Clean dataset1
print("\nMissing values in dataset1:")
print(df1.isnull().sum())

df1['habit'] = df1['habit'].fillna('unknown')
df1['season_name'] = df1['season'].map({0: 'Dry', 1: 'Wet'})
df1['period_duration'] = (df1['rat_period_end'] - df1['rat_period_start']).dt.total_seconds()
df1['bat_landing_to_food_capped'] = df1['bat_landing_to_food'].clip(upper=30)

# Clean dataset2
print("\nMissing values in dataset2:")
print(df2.isnull().sum())

# Handle missing values in dataset2
df2 = df2.fillna({'food_availability': df2['food_availability'].median(),
                 'rat_minutes': 0,
                 'rat_arrival_number': 0})

# Create time-based features
df1['hour_of_day'] = df1['start_time'].dt.hour
df1['day_of_week'] = df1['start_time'].dt.day_name()
df2['hour_of_day'] = df2['time'].dt.hour

# Create risk categories for better analysis
df1['risk_category'] = df1['risk'].map({0: 'Risk_Avoidance', 1: 'Risk_Taking'})

print("Data cleaning and feature engineering complete!")

# INVESTIGATION A: DO BATS PERCEIVE RATS AS PREDATORS?

print("\n" + "=" * 60)
print("INVESTIGATION A: PREDATOR PERCEPTION ANALYSIS")
print("=" * 60)

# 1. Overall behavioral patterns
print("\n3. OVERALL BEHAVIORAL PATTERNS:")

overall_reward_rate = df1['reward'].mean() * 100
overall_risk_rate = df1['risk'].mean() * 100

print(f"Overall Reward Rate: {overall_reward_rate:.1f}%")
print(f"Overall Risk-Taking Rate: {overall_risk_rate:.1f}%")

# 2. Vigilance behavior analysis (Time to approach food)
print("\n4. VIGILANCE BEHAVIOR ANALYSIS:")

vigilance_analysis = df1.groupby('risk').agg({
    'bat_landing_to_food': ['mean', 'median', 'std', 'count'],
    'bat_landing_to_food_capped': ['mean', 'median']
}).round(2)

print("Time to approach food (seconds):")
print(vigilance_analysis)

# Statistical test for time difference
no_risk_time = df1[df1['risk'] == 0]['bat_landing_to_food_capped'].dropna()
risk_time = df1[df1['risk'] == 1]['bat_landing_to_food_capped'].dropna()

t_stat, p_value = stats.ttest_ind(no_risk_time, risk_time, equal_var=False)
print(f"\nT-test for time difference: t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}")

# 3. Success rates under different risk conditions
print("\n5. SUCCESS RATES BY RISK CONDITION:")

risk_reward = df1.groupby('risk')['reward'].mean() * 100
risk_reward_counts = df1.groupby('risk')['reward'].agg(['count', 'sum'])
risk_reward_counts['success_rate'] = (risk_reward_counts['sum'] / risk_reward_counts['count']) * 100

print("Reward rates by risk condition:")
print(risk_reward_counts)

# 4. Behavioral patterns by habit/context
print("\n6. BEHAVIOR BY HABIT/CONTEXT:")

habit_analysis = df1.groupby('habit').agg({
    'reward': ['mean', 'count'],
    'risk': 'mean'
}).round(3)

habit_analysis.columns = ['reward_rate', 'count', 'risk_rate']
habit_analysis = habit_analysis[habit_analysis['count'] >= 5].sort_values('reward_rate', ascending=False)

print("Top behaviors by success rate:")
print(habit_analysis.head(10))

# =============================================================================
# DATASET2 ANALYSIS: RAT ARRIVAL PATTERNS
# =============================================================================

print("\n7. RAT ARRIVAL PATTERNS ANALYSIS (DATASET2):")

# Basic statistics
print("\nRat arrival statistics:")
print(df2[['rat_arrival_number', 'rat_minutes', 'bat_landing_number']].describe())

# Temporal patterns
print("\nRat arrivals by hour:")
hourly_rats = df2.groupby('hour_of_day').agg({
    'rat_arrival_number': 'mean',
    'bat_landing_number': 'mean'
}).round(2)
print(hourly_rats)

# INTEGRATED ANALYSIS: COMBINING BOTH DATASETS

print("\n8. INTEGRATED ANALYSIS:")

# Create time-based aggregation from dataset1 to match dataset2 time intervals
df1['time_30min'] = df1['start_time'].dt.floor('30min')

# Aggregate dataset1 data by 30-minute intervals
df1_agg = df1.groupby('time_30min').agg({
    'reward': 'mean',
    'risk': 'mean',
    'bat_landing_to_food': 'mean',
    'start_time': 'count'
}).rename(columns={'start_time': 'landing_count'})

# Merge with dataset2
merged_df = pd.merge(df2, df1_agg, left_on='time', right_on='time_30min', how='left')

# Analyze correlation between rat presence and bat behavior
correlation_cols = ['rat_arrival_number', 'rat_minutes', 'reward', 'risk', 'bat_landing_to_food']
correlation_matrix = merged_df[correlation_cols].corr()

print("Correlation between rat presence and bat behavior:")
print(correlation_matrix.round(3))

# VISUALIZATIONS FOR INVESTIGATION A

print("\n9. CREATING VISUALIZATIONS...")

# Figure 1: Overall rates
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Overall rates
categories = ['Reward Rate', 'Risk-Taking Rate']
rates = [overall_reward_rate, overall_risk_rate]
colors = ['green', 'red']
bars = ax1.bar(categories, rates, color=colors, alpha=0.7)
ax1.set_title('Overall Behavioral Rates')
ax1.set_ylabel('Percentage (%)')
ax1.set_ylim(0, 100)
for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# 2. Reward rates by risk condition
risk_reward.plot(kind='bar', color=['lightgreen', 'lightcoral'], ax=ax2)
ax2.set_title('Reward Rate by Risk Behavior')
ax2.set_xlabel('Risk Behavior (0=Avoidance, 1=Taking)')
ax2.set_ylabel('Reward Rate (%)')
ax2.set_ylim(0, 100)
for i, v in enumerate(risk_reward):
    ax2.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Time to approach food
sns.boxplot(x='risk_category', y='bat_landing_to_food_capped', data=df1, ax=ax3)
ax3.set_title('Time to Approach Food by Risk Behavior')
ax3.set_xlabel('Risk Behavior')
ax3.set_ylabel('Time to Food (seconds, capped at 30s)')

# 4. Correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=ax4)
ax4.set_title('Correlation: Rat Presence vs. Bat Behavior')

plt.tight_layout()
plt.savefig('investigation_a_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Temporal patterns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Rat arrival patterns by hour
hourly_rats['rat_arrival_number'].plot(kind='bar', ax=ax1, color='orange', alpha=0.7)
ax1.set_title('Average Rat Arrivals by Hour')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Average Rat Arrivals')
ax1.tick_params(axis='x', rotation=45)

# Bat landing patterns by hour
hourly_bats = df1.groupby('hour_of_day').size()
hourly_bats.plot(kind='bar', ax=ax2, color='blue', alpha=0.7)
ax2.set_title('Bat Landings by Hour')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Number of Landings')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('temporal_patterns.png', dpi=300, bbox_inches='tight')
plt.show()

# STATISTICAL CONCLUSIONS FOR INVESTIGATION A

print("\n" + "=" * 60)
print("STATISTICAL CONCLUSIONS FOR INVESTIGATION A")
print("=" * 60)

print("\n10. KEY FINDINGS:")

# 1. Vigilance behavior
median_time_no_risk = df1[df1['risk'] == 0]['bat_landing_to_food'].median()
median_time_risk = df1[df1['risk'] == 1]['bat_landing_to_food'].median()
time_difference = median_time_risk - median_time_no_risk

print(f"1. VIGILANCE BEHAVIOR:")
print(f"   - Bats take {median_time_no_risk:.1f}s to approach food without risk")
print(f"   - Bats take {median_time_risk:.1f}s to approach food with risk-taking behavior")
print(f"   - Time difference: {time_difference:.1f}s (p-value: {p_value:.4f})")
if p_value < 0.05:
    print(f"   → SIGNIFICANT: Bats show increased vigilance when perceiving risk")
else:
    print(f"   → NOT SIGNIFICANT: No clear evidence of increased vigilance")

# 2. Success rates
reward_difference = risk_reward[0] - risk_reward[1]
print(f"\n2. SUCCESS RATES:")
print(f"   - Reward rate without risk: {risk_reward[0]:.1f}%")
print(f"   - Reward rate with risk: {risk_reward[1]:.1f}%")
print(f"   - Difference: {reward_difference:.1f}%")
# Arbitrary threshold for practical significance (5% difference)
if reward_difference > 5:
    print(f"   → PRACTICAL SIGNIFICANCE: Risk avoidance leads to higher success")
else:
    print(f"   → MINOR DIFFERENCE: Success rates are similar")

# 3. Correlation findings
rat_risk_corr = correlation_matrix.loc['rat_arrival_number', 'risk']
print(f"\n3. RAT PRESENCE CORRELATION:")
print(f"   - Correlation between rat arrivals and risk-taking: {rat_risk_corr:.3f}")
if abs(rat_risk_corr) > 0.2:
    direction = "positive" if rat_risk_corr > 0 else "negative"
    print(f"   → MODERATE {direction.upper()} CORRELATION: Rat presence affects bat risk behavior")
else:
    print(f"   → WEAK CORRELATION: Limited direct relationship")

print(f"\n4. OVERALL CONCLUSION:")
if p_value < 0.05 and reward_difference > 5:
    print("   → STRONG EVIDENCE: Bats perceive rats as predators and adjust behavior accordingly")
elif p_value < 0.05 or reward_difference > 5:
    print("   → MODERATE EVIDENCE: Some behavioral changes suggest predator perception")
else:
    print("   → LIMITED EVIDENCE: Inconclusive results for predator perception")

# PRELIMINARY ANALYSIS FOR INVESTIGATION B (SEASONAL CHANGES)

print("\n" + "=" * 60)
print("PRELIMINARY ANALYSIS FOR INVESTIGATION B: SEASONAL CHANGES")
print("=" * 60)

# Seasonal analysis
seasonal_analysis = df1.groupby('season_name').agg({
    'reward': 'mean',
    'risk': 'mean',
    'bat_landing_to_food': 'median',
    'start_time': 'count'
}).rename(columns={'start_time': 'observation_count'})

seasonal_analysis['reward'] = seasonal_analysis['reward'] * 100
seasonal_analysis['risk'] = seasonal_analysis['risk'] * 100

print("\nSeasonal Behavioral Patterns:")
print(seasonal_analysis.round(2))

# Statistical test for seasonal differences
dry_season = df1[df1['season_name'] == 'Dry']['reward']
wet_season = df1[df1['season_name'] == 'Wet']['reward']
season_t_stat, season_p_value = stats.ttest_ind(dry_season, wet_season, nan_policy='omit')

print(f"\nSeasonal difference in reward rates: p-value = {season_p_value:.4f}")

# Save processed data for future analysis
df1.to_csv('processed_dataset1.csv', index=False)
df2.to_csv('processed_dataset2.csv', index=False)
merged_df.to_csv('integrated_analysis.csv', index=False)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("Processed data saved for further investigation")
print("=" * 60)

# ============================
# SUMMARY DIAGRAM FOR ALL FINDINGS
# ============================

print("\n10. CREATING SUMMARY DIAGRAM OF ALL KEY FINDINGS...")

fig, axs = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle("Bat Foraging Behavior: Key Findings", fontsize=18, fontweight='bold')

# Panel 1: Overall behavioral rates
categories = ['Reward Rate', 'Risk-Taking Rate']
rates = [overall_reward_rate, overall_risk_rate]
colors = ['green', 'red']
bars = axs[0, 0].bar(categories, rates, color=colors, alpha=0.7)
axs[0, 0].set_ylim(0, 100)
axs[0, 0].set_title("Overall Behavioral Rates")
for bar, rate in zip(bars, rates):
    axs[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{rate:.1f}%', ha='center', fontweight='bold')

# Panel 2: Vigilance behavior
sns.boxplot(x='risk_category', y='bat_landing_to_food_capped', data=df1, ax=axs[0, 1])
axs[0, 1].set_title("Time to Approach Food by Risk Behavior")
axs[0, 1].set_ylabel("Time (s, capped at 30s)")

# Panel 3: Success rates by risk condition
risk_reward.plot(kind='bar', color=['lightgreen', 'lightcoral'], ax=axs[1, 0], legend=False)
axs[1, 0].set_ylim(0, 100)
axs[1, 0].set_title("Reward Rate by Risk Behavior")
for i, v in enumerate(risk_reward):
    axs[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# Panel 4: Correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=axs[1, 1])
axs[1, 1].set_title("Correlation: Rat Presence vs Bat Behavior")

# Panel 5: Seasonal differences
seasonal_analysis[['reward','risk']].plot(kind='bar', ax=axs[2, 0])
axs[2, 0].set_title("Seasonal Behavioral Patterns")
axs[2, 0].set_ylabel("Percentage (%)")
axs[2, 0].set_ylim(0,100)

# Remove empty subplot (bottom right)
fig.delaxes(axs[2,1])

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('summary_key_findings.png', dpi=300, bbox_inches='tight')
plt.show()

print("Summary diagram saved as 'summary_key_findings.png'")
