import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv("dataset1.csv")

# Convert relevant columns to datetime
df['start_time'] = pd.to_datetime(df['start_time'], format='%d/%m/%Y %H:%M')
df['rat_period_start'] = pd.to_datetime(df['rat_period_start'], format='%d/%m/%Y %H:%M')
df['rat_period_end'] = pd.to_datetime(df['rat_period_end'], format='%d/%m/%Y %H:%M')
df['sunset_time'] = pd.to_datetime(df['sunset_time'], format='%d/%m/%Y %H:%M')

# Handle missing values in 'habit' by filling with 'unknown'
df['habit'].fillna('unknown', inplace=True)

# Correlation analysis: calculate Pearson correlation between key variables
correlation_matrix = df[['bat_landing_to_food', 'seconds_after_rat_arrival', 'risk', 'reward']].corr()

# T-test: Compare 'bat_landing_to_food' for risk-taking (risk=1) vs risk-avoidance (risk=0)
risk_avoidance_group = df[df['risk'] == 0]['bat_landing_to_food']
risk_taking_group = df[df['risk'] == 1]['bat_landing_to_food']
t_stat, p_val = stats.ttest_ind(risk_avoidance_group, risk_taking_group)

# Chi-square test: Relationship between 'risk' and 'reward'
contingency_table = pd.crosstab(df['risk'], df['reward'])
chi2_stat, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)

# Print correlation results, T-test, and Chi-square results
print("Correlation Matrix:")
print(correlation_matrix)
print("\nT-test result for foraging time between risk-taking and risk-avoidance bats:")
print(f"T-statistic: {t_stat}, P-value: {p_val}")
print("\nChi-square result for relationship between risk and reward:")
print(f"Chi2-statistic: {chi2_stat}, P-value: {p_chi2}")

# Visualizations
# 1. Foraging Time vs Predation Risk
plt.figure(figsize=(10, 6))
sns.boxplot(x='risk', y='bat_landing_to_food', data=df, palette="Set2")
plt.title('Foraging Time vs Predation Risk (Risk 0 vs Risk 1)')
plt.xlabel('Predation Risk (0 = Avoidance, 1 = Risk-taking)')
plt.ylabel('Time to Food (seconds)')
plt.show()

# 2. Foraging Success vs Predation Risk
plt.figure(figsize=(10, 6))
sns.countplot(x='risk', hue='reward', data=df, palette="Set1")
plt.title('Foraging Success vs Predation Risk (Risk 0 vs Risk 1)')
plt.xlabel('Predation Risk (0 = Avoidance, 1 = Risk-taking)')
plt.ylabel('Count of Observations')
plt.legend(title='Foraging Success (Reward)', loc='upper right')
plt.show()

# 3. Foraging Time vs Habit (Behavioral Context)
plt.figure(figsize=(12, 6))
sns.boxplot(x='habit', y='bat_landing_to_food', data=df, palette="Set1")
plt.title('Foraging Time vs Habit (Behavioral Context)')
plt.xlabel('Behavioral Context (Habit)')
plt.ylabel('Time to Food (seconds)')
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap (visualising the matrix)
print("\n[My Contribution] Correlation heatmap (visualising the correlation matrix).")
plt.figure(figsize=(6,5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Correlation Heatmap (key variables)')
plt.tight_layout()
plt.show()

# Regression plot: rat arrival delay vs time to food
print("\n[My Contribution] Relationship between rat arrival delay and time to food (with linear fit).")
mask = df[['seconds_after_rat_arrival','bat_landing_to_food']].dropna()
if not mask.empty:
    ax = sns.regplot(
        x='seconds_after_rat_arrival',
        y='bat_landing_to_food',
        data=mask,
        scatter_kws={'alpha':0.6},
        line_kws={'linewidth':2}
    )
    plt.title('Rat arrival delay vs Time to food')
    plt.xlabel('Seconds after rat arrival')
    plt.ylabel('Time to food (seconds)')
    plt.tight_layout()
    plt.show()

    # Print simple linear regression stats
    slope, intercept, r, p, se = stats.linregress(
        mask['seconds_after_rat_arrival'],
        mask['bat_landing_to_food']
    )
    print(f"Linear fit: slope = {slope:.3f}, r = {r:.2f}, p = {p:.4f}")
else:
    print("Not enough non-missing data to draw the regression plot.")
