import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv('dataset1.csv', parse_dates=['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time'], dayfirst=True)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

print("Missing Values in Each Column:")
print(df.isnull().sum())

df['habit'] = df['habit'].fillna('unknown')
print(f"\nHabit categories after filling NA: {df['habit'].unique()}")

df['season_name'] = df['season'].map({0: 'Dry', 1: 'Wet'})

df['period_duration'] = (df['rat_period_end'] - df['rat_period_start']).dt.total_seconds()

print(f"\nKey Numerical Columns Description:")
print(df[['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']].describe())

df['bat_landing_to_food_capped'] = df['bat_landing_to_food'].clip(upper=30)

print("\nData cleaning and preprocessing complete.")

# 3.1 Calculate overall rates (Overall Success vs. Risk Rates)
overall_reward_rate = df['reward'].mean() * 100
overall_risk_rate = df['risk'].mean() * 100

# Create a bar plot
fig, ax = plt.subplots()
categories = ['Reward Rate', 'Risk Rate']
rates = [overall_reward_rate, overall_risk_rate]
bars = ax.bar(categories, rates, color=['green', 'red'])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.title('Overall Reward and Risk Rates')
plt.ylabel('Percentage of Events (%)')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

print(f"Overall Reward Rate: {overall_reward_rate:.1f}%")
print(f"Overall Risk Rate: {overall_risk_rate:.1f}%")



# Group by risk presence and calculate reward rate
risk_reward = df.groupby('risk')['reward'].mean() * 100
print(risk_reward)
plt.figure(figsize=(8, 6))
ax = risk_reward.plot(kind='bar', color=['lightgreen', 'lightcoral'])
plt.title('Reward Rate With vs. Without Risk (Rat Presence)')
plt.xlabel('Risk Present (0 = No, 1 = Yes)')
plt.ylabel('Reward Rate (%)')
plt.xticks(rotation=0)
plt.ylim(0, 100)

# Add value labels on top of bars
for i, v in enumerate(risk_reward):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Calculate the difference
reward_difference = risk_reward[0] - risk_reward[1]
print(f"Reward rate is {reward_difference:.1f}% higher when no risk is present.")

# Analyze the most common habits and their success rates
habit_analysis = df.groupby('habit').agg(
    count=('habit', 'size'),
    reward_rate=('reward', 'mean'),
    risk_rate=('risk', 'mean')
).sort_values('count', ascending=False)

# Filter for habits with a minimum number of occurrences for significance
significant_habits = habit_analysis[habit_analysis['count'] >= 10]

print("Habit Analysis (Significant Samples):")
print(significant_habits.head(10))

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=significant_habits.index[:8], y=significant_habits['reward_rate'][:8] * 100)
plt.title('Reward Rate by Habit/Behavior (Top 8)')
plt.xlabel('Habit/Behavior Type')
plt.ylabel('Reward Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)

for i, v in enumerate(significant_habits['reward_rate'][:8] * 100):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


# Compare the time to get food with and without risk
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['risk'], y=df['bat_landing_to_food_capped'])
plt.title('Time to Get Food (Capped at 30s) With vs. Without Risk')
plt.xlabel('Risk Present (0 = No, 1 = Yes)')
plt.ylabel('Time from Landing to Food (seconds)')
plt.show()

# Calculate median times
median_times = df.groupby('risk')['bat_landing_to_food'].median()
print(f"Median time to get food without risk: {median_times[0]:.2f}s")
print(f"Median time to get food with risk: {median_times[1]:.2f}s")



# Analyze reward and risk rates by season
season_analysis = df.groupby('season_name').agg(
    reward_rate=('reward', 'mean'),
    risk_rate=('risk', 'mean')
) * 100

print("Seasonal Analysis:")
print(season_analysis)

# Create a grouped bar plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(season_analysis.index))
width = 0.35

bars1 = ax.bar(x - width/2, season_analysis['reward_rate'], width, label='Reward Rate', color='green')
bars2 = ax.bar(x + width/2, season_analysis['risk_rate'], width, label='Risk Rate', color='red')

ax.set_xlabel('Season')
ax.set_ylabel('Rate (%)')
ax.set_title('Reward and Risk Rates by Season')
ax.set_xticks(x)
ax.set_xticklabels(season_analysis.index)
ax.legend()
ax.set_ylim(0, 100)

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.show()