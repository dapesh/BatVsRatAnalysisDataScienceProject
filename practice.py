import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# STEP_1 Load and inspect data

df1 = pd.read_csv('dataset1.csv', parse_dates=['start_time', 'rat_period_start','rat_period_end','sunset_time'], dayfirst=True)
df2 = pd.read_csv('dataset2.csv',parse_dates=['time'],dayfirst=True)



# STEP_2: Clean and Prepare Data
#Calculate Outliers
Q1 = df1['bat_landing_to_food'].quantile(0.25)  # 25th percentile
Q3 = df1['bat_landing_to_food'].quantile(0.75)  # 75th percentile

Q1 = df1['bat_landing_to_food'].quantile(0.25)  # 25th percentile
Q3 = df1['bat_landing_to_food'].quantile(0.75)  # 75th percentile

print("Q1 (25th percentile):", Q1)
print("Q3 (75th percentile):", Q3)

IQR = Q3 - Q1  # Interquartile Range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
outliers = df1[(df1['bat_landing_to_food'] < lower_bound) | 
               (df1['bat_landing_to_food'] > upper_bound)]
print(outliers)
print("Number of outliers:", outliers.shape[0])
df1['bat_landing_to_food_capped'] = df1['bat_landing_to_food'].clip(lower=lower_bound, upper=upper_bound)


df1['season_name'] = df1['season'].map({0: 'Dry', 1: 'Wet'})
df1['risk_category'] = df1['risk'].map({0: 'Risk Avoidance', 1: 'Risk Taking'})

#STEP_3: Calculate Metrics

overall_reward_rate = df1['reward'].mean() * 100
print("Overall Reward Rate:", overall_reward_rate, "%")

overall_risk_rate = df1['risk'].mean() * 100
print("Overall Risk Rate:", overall_risk_rate, "%")

avg_time_to_food = df1['bat_landing_to_food_capped'].mean()
print(f"Average Time-to-Food: {avg_time_to_food:.2f} seconds")

#When bats take risk vs. when they avoid risk
risk_reward = df1.groupby('risk')['reward'].mean() * 100

#STEP_4: Statistical Tests

# Does the presence of rat affects bat behaviour
no_risk_time = df1[df1['risk'] == 0]["bat_landing_to_food_capped"]
risk_time = df1[df1['risk'] == 1]["bat_landing_to_food_capped"]

t_stat, p_value = stats.ttest_ind(no_risk_time,risk_time, equal_var= False)
print (p_value)


# Does the season change that behaviour (It's about comparing behaviour of bats with season)

dry_season_times = df1[df1['season_name'] == 'Dry']['bat_landing_to_food_capped']
wet_season_times = df1[df1['season_name'] == 'Wet']['bat_landing_to_food_capped']
t_stat, p_value = stats.ttest_ind(dry_season_times, wet_season_times, equal_var=False)
print(p_value)

# Risk Taking with Seasons
risk_by_season = df1.groupby('season_name')['risk'].mean() * 100
print(risk_by_season)

# Reward Taking with Seasons
reward_by_season = df1.groupby('season_name')['reward'].mean() * 100
print(reward_by_season)

#STEP_5 Visualize

#Visualization for if presence of rat affects bat behaviour
plt.figure(figsize=(8,6))
sns.boxplot(x='risk_category', y='bat_landing_to_food_capped', data=df1)
plt.title('Time to Approach Food by Risk Behavior')
plt.ylabel('Bat Time from landing to food (seconds)')  # <- updated label
plt.xlabel('Risk Behavior')
plt.show()



#Visualization for comparing behaviour of bat on the basis of season
reward_by_season.plot(kind='bar', color=['green','purple'])
plt.ylabel("Reward Rate (%)")
plt.title("Success / Reward Rate by Season")
plt.show()

risk_by_season.plot(kind='bar', color=['blue','orange'])
plt.ylabel('Risk-Taking Rate (%)')
plt.title('Risk-Taking Behavior by Season')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='season_name', y='bat_landing_to_food_capped', data=df1)
plt.title('Bat Vigilance by Season')
plt.ylabel('Time from landing to food (seconds)')
plt.xlabel('Season')
plt.show()


plt.figure(figsize=(8,6))
sns.boxplot(x='risk', y='reward', data=df1)
plt.title('Bat Vigilance by Season')
plt.ylabel('Reward')
plt.xlabel('Risk')
plt.show()


# Data for bar charts
labels = ['Reward Rate', 'Risk Rate', 'Avg Time-to-Food (s)']
values = [overall_reward_rate, overall_risk_rate, avg_time_to_food]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Bar chart for reward rate
axes[0].bar(labels[0], values[0], color='green')
axes[0].set_ylabel('Proportion')
axes[0].set_title('Overall Reward Rate')

# Bar chart for risk rate
axes[1].bar(labels[1], values[1], color='red')
axes[1].set_ylabel('Proportion')
axes[1].set_title('Overall Risk Rate')

# Bar chart for time-to-food
axes[2].bar(labels[2], values[2], color='blue')
axes[2].set_ylabel('Seconds')
axes[2].set_title('Average Time-to-Food')

plt.tight_layout()
plt.show()






