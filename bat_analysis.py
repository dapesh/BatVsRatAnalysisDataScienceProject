import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

# Set visualization style
plt.style.use('default')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)

# Change to the correct directory
os.chdir(r"C:\Users\shish\HIT140-Group-Project\prakash_data_prep")

print("Loading and preparing both datasets...")

# Load datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')

print(f"Dataset 1 shape: {df1.shape}")
print(f"Dataset 2 shape: {df2.shape}")

# =============================================================================
# DATA CLEANING AND PREPARATION
# =============================================================================

print("\n=== DATA CLEANING AND PREPARATION ===")

# Clean Dataset 1
df1_clean = df1.dropna(how='all').dropna(axis=1, how='all')
# Handle missing values in 'habit' column
df1_clean['habit'] = df1_clean['habit'].fillna('unknown')

# Clean Dataset 2
df2_clean = df2.dropna(how='all').dropna(axis=1, how='all')

print(f"After cleaning - Dataset 1: {df1_clean.shape}, Dataset 2: {df2_clean.shape}")

# Create condition variables
df1_clean['condition'] = df1_clean['risk'].apply(lambda x: 'rat_present' if x == 1 else 'control')
df2_clean['condition'] = df2_clean['rat_minutes'].apply(lambda x: 'rat_present' if x > 0 else 'control')

# Add dataset source identifier
df1_clean['dataset_source'] = 'individual_events'
df2_clean['dataset_source'] = 'time_series'

# Convert time columns to datetime
def convert_time(time_str):
    try:
        return pd.to_datetime(time_str, dayfirst=True)
    except:
        return pd.NaT

df1_clean['start_time_dt'] = df1_clean['start_time'].apply(convert_time)
df1_clean['sunset_time_dt'] = df1_clean['sunset_time'].apply(convert_time)
df2_clean['time_dt'] = df2_clean['time'].apply(convert_time)

# =============================================================================
# COMBINE DATASETS FOR ANALYSIS
# =============================================================================

print("\n=== COMBINING DATASETS ===")

# Select common columns for combined analysis
common_columns = ['condition', 'dataset_source', 'hours_after_sunset']

# Create a combined dataframe with key metrics
combined_data = pd.DataFrame()

# Add metrics from Dataset 1
df1_metrics = df1_clean[['condition', 'dataset_source', 'hours_after_sunset', 
                         'bat_landing_to_food', 'seconds_after_rat_arrival']].copy()
df1_metrics['metric_type'] = 'individual_behavior'
combined_data = pd.concat([combined_data, df1_metrics], ignore_index=True)

# Add metrics from Dataset 2
df2_metrics = df2_clean[['condition', 'dataset_source', 'hours_after_sunset',
                         'bat_landing_number', 'food_availability']].copy()
df2_metrics['metric_type'] = 'aggregate_behavior'
combined_data = pd.concat([combined_data, df2_metrics], ignore_index=True)

print(f"Combined dataset shape: {combined_data.shape}")

# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n=== EXPLORATORY DATA ANALYSIS ===")

# 1. Condition distribution
print("\n1. Condition Distribution:")
condition_counts = combined_data['condition'].value_counts()
print(condition_counts)

# 2. Dataset source distribution
print("\n2. Dataset Source Distribution:")
source_counts = combined_data['dataset_source'].value_counts()
print(source_counts)

# 3. Basic statistics by condition and dataset
print("\n3. Summary Statistics by Condition and Dataset:")

# For individual behavior metrics (from Dataset 1)
indiv_stats = df1_clean.groupby('condition')[['bat_landing_to_food', 'seconds_after_rat_arrival']].agg(['mean', 'std', 'count'])
print("\nIndividual Behavior Metrics:")
print(indiv_stats)

# For aggregate behavior metrics (from Dataset 2)
agg_stats = df2_clean.groupby('condition')[['bat_landing_number', 'food_availability']].agg(['mean', 'std', 'count'])
print("\nAggregate Behavior Metrics:")
print(agg_stats)

# =============================================================================
# DATA VISUALIZATION
# =============================================================================

print("\n=== DATA VISUALIZATION ===")

# Create a comprehensive visualization dashboard
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('Comprehensive Analysis of Bat Behavior with Rats Present', fontsize=16, fontweight='bold')

# 1. Bat landing time by condition (Dataset 1)
ax1 = axes[0, 0]
sns.boxplot(x='condition', y='bat_landing_to_food', data=df1_clean, ax=ax1)
ax1.set_title('Time to Land on Food (Individual Events)')
ax1.set_ylabel('Time (seconds)')
ax1.set_xlabel('Condition')

# Add statistical annotation
control_mean = df1_clean[df1_clean['condition'] == 'control']['bat_landing_to_food'].mean()
rat_mean = df1_clean[df1_clean['condition'] == 'rat_present']['bat_landing_to_food'].mean()
ax1.text(0.5, 0.9, f'Control: {control_mean:.2f}s\nRat: {rat_mean:.2f}s', 
         transform=ax1.transAxes, ha='center', va='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# 2. Seconds after rat arrival (Dataset 1)
ax2 = axes[0, 1]
sns.boxplot(x='condition', y='seconds_after_rat_arrival', data=df1_clean, ax=ax2)
ax2.set_title('Time After Rat Arrival (Individual Events)')
ax2.set_ylabel('Time (seconds)')
ax2.set_xlabel('Condition')

# 3. Bat landing number by condition (Dataset 2)
ax3 = axes[1, 0]
sns.boxplot(x='condition', y='bat_landing_number', data=df2_clean, ax=ax3)
ax3.set_title('Number of Bat Landings (Time Series)')
ax3.set_ylabel('Count')
ax3.set_xlabel('Condition')

# Add statistical annotation
control_mean = df2_clean[df2_clean['condition'] == 'control']['bat_landing_number'].mean()
rat_mean = df2_clean[df2_clean['condition'] == 'rat_present']['bat_landing_number'].mean()
ax3.text(0.5, 0.9, f'Control: {control_mean:.2f}\nRat: {rat_mean:.2f}', 
         transform=ax3.transAxes, ha='center', va='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# 4. Food availability by condition (Dataset 2)
ax4 = axes[1, 1]
sns.boxplot(x='condition', y='food_availability', data=df2_clean, ax=ax4)
ax4.set_title('Food Availability (Time Series)')
ax4.set_ylabel('Availability Score')
ax4.set_xlabel('Condition')

# 5. Activity by hours after sunset (combined)
ax5 = axes[2, 0]
for condition in ['control', 'rat_present']:
    subset = df2_clean[df2_clean['condition'] == condition]
    sns.lineplot(x='hours_after_sunset', y='bat_landing_number', data=subset, 
                 label=condition, ax=ax5, ci='sd')
ax5.set_title('Bat Activity Patterns by Time After Sunset')
ax5.set_ylabel('Number of Bat Landings')
ax5.set_xlabel('Hours After Sunset')
ax5.legend()

# 6. Condition distribution
ax6 = axes[2, 1]
condition_counts.plot(kind='bar', color=['skyblue', 'lightcoral'], ax=ax6)
ax6.set_title('Distribution of Observations by Condition')
ax6.set_ylabel('Count')
ax6.set_xlabel('Condition')
for i, v in enumerate(condition_counts):
    ax6.text(i, v + 10, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

print("\n=== STATISTICAL ANALYSIS ===")

# Function to perform statistical tests
def perform_statistical_test(control_data, treatment_data, metric_name):
    print(f"\n--- Statistical Analysis for {metric_name} ---")
    
    # Check if we have enough data
    if len(control_data) < 2 or len(treatment_data) < 2:
        print("Not enough data for analysis")
        return None
    
    # Check normality
    _, p_control = stats.shapiro(control_data)
    _, p_treatment = stats.shapiro(treatment_data)
    
    print(f"Normality test p-values: control={p_control:.3f}, treatment={p_treatment:.3f}")
    
    # Use appropriate test
    if p_control > 0.05 and p_treatment > 0.05:
        # Parametric test (t-test)
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
        test_type = "Independent t-test"
        print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
    else:
        # Non-parametric test (Mann-Whitney U)
        u_stat, p_value = stats.mannwhitneyu(control_data, treatment_data)
        test_type = "Mann-Whitney U test"
        print(f"Mann-Whitney U test: U={u_stat}, p={p_value:.3f}")
    
    # Calculate effect size
    def cohens_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    d = cohens_d(treatment_data, control_data)
    print(f"Effect size (Cohen's d): {d:.3f}")
    
    # Interpret results
    if p_value < 0.05:
        print(f"Significant difference in {metric_name} between conditions (p < 0.05)")
        if np.mean(treatment_data) > np.mean(control_data):
            print(f"Higher {metric_name} when rats were present")
        else:
            print(f"Lower {metric_name} when rats were present")
    else:
        print(f"No significant difference in {metric_name} between conditions (p = {p_value:.3f})")
    
    return {
        'test_type': test_type,
        'statistic': t_stat if test_type == "Independent t-test" else u_stat,
        'p_value': p_value,
        'effect_size': d,
        'control_mean': np.mean(control_data),
        'treatment_mean': np.mean(treatment_data),
        'control_count': len(control_data),
        'treatment_count': len(treatment_data)
    }

# Perform statistical tests
results = {}

# Dataset 1 metrics
results['bat_landing_time'] = perform_statistical_test(
    df1_clean[df1_clean['condition'] == 'control']['bat_landing_to_food'],
    df1_clean[df1_clean['condition'] == 'rat_present']['bat_landing_to_food'],
    "Bat Landing Time"
)

results['time_after_rat'] = perform_statistical_test(
    df1_clean[df1_clean['condition'] == 'control']['seconds_after_rat_arrival'],
    df1_clean[df1_clean['condition'] == 'rat_present']['seconds_after_rat_arrival'],
    "Time After Rat Arrival"
)

# Dataset 2 metrics
results['bat_landing_count'] = perform_statistical_test(
    df2_clean[df2_clean['condition'] == 'control']['bat_landing_number'],
    df2_clean[df2_clean['condition'] == 'rat_present']['bat_landing_number'],
    "Bat Landing Count"
)

results['food_availability'] = perform_statistical_test(
    df2_clean[df2_clean['condition'] == 'control']['food_availability'],
    df2_clean[df2_clean['condition'] == 'rat_present']['food_availability'],
    "Food Availability"
)

# =============================================================================
# SAVE RESULTS AND CONCLUSION
# =============================================================================

print("\n=== SAVING RESULTS ===")

# Save cleaned datasets
df1_clean.to_csv('cleaned_dataset1.csv', index=False)
df2_clean.to_csv('cleaned_dataset2.csv', index=False)
combined_data.to_csv('combined_analysis_data.csv', index=False)

# Create a comprehensive report
with open('comprehensive_analysis_report.txt', 'w') as f:
    f.write("COMPREHENSIVE BAT BEHAVIOR ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write(f"Dataset 1 (Individual Events): {df1.shape} -> {df1_clean.shape} after cleaning\n")
    f.write(f"Dataset 2 (Time Series): {df2.shape} -> {df2_clean.shape} after cleaning\n")
    f.write(f"Combined Dataset: {combined_data.shape}\n\n")
    
    f.write("CONDITION DISTRIBUTION:\n")
    f.write(f"Control observations: {condition_counts.get('control', 0)}\n")
    f.write(f"Rat present observations: {condition_counts.get('rat_present', 0)}\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-" * 40 + "\n")
    
    for metric, result in results.items():
        if result:
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            f.write(f"  Control mean: {result['control_mean']:.2f} (n={result['control_count']})\n")
            f.write(f"  Rat present mean: {result['treatment_mean']:.2f} (n={result['treatment_count']})\n")
            f.write(f"  Test: {result['test_type']}\n")
            f.write(f"  p-value: {result['p_value']:.4f}\n")
            f.write(f"  Effect size: {result['effect_size']:.3f}\n")
            
            if result['p_value'] < 0.05:
                if result['treatment_mean'] > result['control_mean']:
                    f.write("  CONCLUSION: Significant increase when rats present\n")
                else:
                    f.write("  CONCLUSION: Significant decrease when rats present\n")
            else:
                f.write("  CONCLUSION: No significant difference\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("OVERALL INTERPRETATION:\n")
    f.write("-" * 40 + "\n")
    
    # Provide overall interpretation
    significant_results = [k for k, v in results.items() if v and v['p_value'] < 0.05]
    
    if significant_results:
        f.write("The analysis provides evidence that bats change their behavior when rats are present.\n")
        f.write(f"Significant differences were found in: {', '.join(significant_results)}\n")
        
        if 'bat_landing_time' in significant_results:
            if results['bat_landing_time']['treatment_mean'] > results['bat_landing_time']['control_mean']:
                f.write("Bats take longer to approach food when rats are present, suggesting increased caution.\n")
            else:
                f.write("Bats approach food more quickly when rats are present.\n")
                
        if 'bat_landing_count' in significant_results:
            if results['bat_landing_count']['treatment_mean'] < results['bat_landing_count']['control_mean']:
                f.write("Fewer bats land when rats are present, suggesting avoidance behavior.\n")
    
    else:
        f.write("The analysis did not find statistically significant evidence that bats change their behavior when rats are present.\n")

print("Analysis complete!")
print("Generated files:")
print("- comprehensive_analysis.png (visualizations)")
print("- comprehensive_analysis_report.txt (detailed results)")
print("- cleaned_dataset1.csv, cleaned_dataset2.csv, combined_analysis_data.csv (cleaned data)")