import os
import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns


def annotate_boxplot(ax, series):
    stats = series.describe()
    q1 = stats['25%']
    q2 = stats['50%']
    q3 = stats['75%']
    min_val = stats['min']
    max_val = stats['max']

    data_range = max_val - min_val
    y_offset = max(data_range * 0.05, 0.2)

    # Adjust y-limits to give space for labels
    ax.set_ylim(min_val - y_offset * 2, max_val + y_offset * 2)

    # Label positions
    
    ax.text(0, q1, f"Q1: {q1:.2f}", ha='center', va='bottom', fontsize=9)

    # Determine if median overlaps with Q1 or Q3
    small_gap = y_offset * 1.5  # define what counts as 'too close'
    if abs(q2 - q1) < small_gap and abs(q3 - q2) < small_gap:
        # If too close on both sides, move median slightly above Q3
        ax.text(0, q3 + y_offset * 0.6, f"Med: {q2:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')
    elif abs(q2 - q1) < abs(q3 - q2):
        # Closer to Q1 — shift upward
        ax.text(0, q2 + y_offset * 0.4, f"Med: {q2:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')
    else:
        # Closer to Q3 — shift downward
        ax.text(0, q2 - y_offset * 0.4, f"Med: {q2:.2f}", ha='center', va='top', fontsize=9, weight='bold')

    ax.text(0, q3, f"Q3: {q3:.2f}", ha='center', va='bottom', fontsize=9)

metric_groups = {
    "Heart Rate Metrics": ['min_heart_rate', 'max_heart_rate', 'resting_heart_rate', 'sleep_resting_heart_rate'],
    "SPO2 Metrics": ['avg_spo2', 'min_spo2', 'avg_sleep_spo2'],
    "Respiration Metrics": ['max_respiration', 'min_respiration', 'avg_waking_respiration', 'sleep_avg_respiration'],
    "Body Composition": ['bmi', 'weight_kg'],
    "Fitness & Performance": ['vo2_max_precise', 'fitness_age'],
    "Sleep Metrics": ['sleep_time_sec'],
    "Movement & Hydration": ['steps', 'hydration_ml']
}

# # --- Risk label function (your original one) ---
# def compute_cvd_risk_label(row, age):
#     weights = {
#         'vo2max_bmi': 0.13,
#         'vo2max_sleep_time': 0.09,
#         'bmi_resting_hr': 0.12,
#         'sleep_resting_hr': 0.08,
#         'sleep_avg_respiration': 0.07,
#         'avg_waking_respiration_resting_hr': 0.10,
#         'fitness_age_resting_hr': 0.11,
#         'sleep_avg_spo2_sleep_time': 0.08,
#         'steps': 0.06,
#         'hydration_ml': 0.04,
#         'avg_spo2': 0.07,
#         'avg_waking_respiration': 0.05
#     }

#     score = 0
#     if row['vo2_max_precise'] < 41.7 and (row['bmi'] < 18.5 or row['bmi'] > 24.9):
#         score += weights['vo2max_bmi']
#     if row['vo2_max_precise'] < 41.7 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
#         score += weights['vo2max_sleep_time']
#     if (row['bmi'] < 18.5 or row['bmi'] > 24.9) and row['resting_heart_rate'] > 60:
#         score += weights['bmi_resting_hr']
#     if row['sleep_resting_heart_rate'] < 40 or row['sleep_resting_heart_rate'] > 60:
#         score += weights['sleep_resting_hr']
#     if row['sleep_avg_respiration'] < 14 or row['sleep_avg_respiration'] > 16:
#         score += weights['sleep_avg_respiration']
#     if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20) and row['resting_heart_rate'] > 60:
#         score += weights['avg_waking_respiration_resting_hr']
#     if row['fitness_age'] >= age and row['resting_heart_rate'] > 60:
#         score += weights['fitness_age_resting_hr']
#     if row['avg_sleep_spo2'] < 95 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
#         score += weights['sleep_avg_spo2_sleep_time']
#     if row['steps'] < 4500:
#         score += weights['steps']
#     if row['hydration_ml'] < 1500:
#         score += weights['hydration_ml']
#     if row['avg_spo2'] < 95:
#         score += weights['avg_spo2']
#     if row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20:
#         score += weights['avg_waking_respiration']

#     if score <= 0.3:
#         return 'Low'
#     elif score < 0.5:
#         return 'Medium'
#     else:
#         return 'High'

# # --- Simulation Parameters ---
# np.random.seed(42)
# random.seed(42)
# start_date = datetime.date(2024, 4, 1)
# records = 2000
# age = 26
# height_cm = 175

# # --- Profile Type Distribution (adjust as desired) ---
# profile_types = ['healthy', 'sedentary', 'cvd']
# # profile_probs = [0.4, 0.35, 0.25]  # simulates real-world population (testing)
# profile_probs = [0.2, 0.4, 0.4] # simulates balanced risk scnearios for training (not working properly)
# # --- Init Lists ---
# rows = []

# # --- Main Loop ---
# for i in range(records):
#     date = start_date + datetime.timedelta(days=i)
#     profile = np.random.choice(profile_types, p=profile_probs)

#     # --- Base parameters for each profile ---
#     if profile == 'healthy':
#         vo2 = np.random.normal(47, 4)
#         bmi = np.random.normal(19, 1.5)
#         resting_hr = np.random.normal(55, 4)
#         spo2 = np.random.normal(96, 0.5)
#         sleep_spo2 = np.random.normal(96, 0.6)
#         steps = int(np.random.normal(7000, 2000))
#         hydration = int(np.random.normal(2200, 300))
#         sleep_sec = int(np.random.normal(7 * 3600, 1800))
#         sleep_rr = int(np.random.normal(14, 0.5))
#         sleep_hr = int(np.random.normal(47, 5))
#         avg_resp = round(random.randint(13, 17))
#         max_hr = int(np.random.normal(168, 10))
#     elif profile == 'sedentary':
#         vo2 = np.random.normal(33, 4)
#         bmi = np.random.normal(25, 2)
#         resting_hr = np.random.normal(63, 6)
#         spo2 = np.random.normal(94, 1)
#         sleep_spo2 = np.random.normal(93, 1)
#         steps = int(np.random.normal(2500, 1000))
#         hydration = int(np.random.normal(1600, 250))
#         sleep_sec = int(np.random.normal(6 * 3600, 2400))
#         sleep_rr = int(np.random.normal(16, 1))
#         sleep_hr = int(np.random.normal(64, 10))
#         avg_resp = random.randint(14, 17)
#         max_hr = int(np.random.normal(150, 10))
#     else:  # cvd
#         vo2 = np.random.normal(24, 5)
#         bmi = np.random.normal(28, 4)
#         resting_hr = np.random.normal(75, 8)
#         spo2 = np.random.normal(91, 2)
#         sleep_spo2 = np.random.normal(90, 1.5)
#         steps = int(np.random.normal(1500, 800))
#         hydration = int(np.random.normal(1500, 250))
#         sleep_sec = int(np.random.normal(9 * 3600, 3600))
#         sleep_rr = int(np.random.normal(17, 1.5))
#         sleep_hr = int(np.random.normal(78, 12))
#         avg_resp = int(np.random.normal(17, 1))
#         max_hr = int(np.random.normal(135, 12))

#     # --- Add noise & crossover symptoms ---
#     bmi = max(15, bmi + np.random.normal(0, 1.5))
#     vo2 = np.clip(vo2 + np.random.normal(0, 2), 15, 60)
#     #resting_hr = max(35, resting_hr + np.random.normal(0, 3))
#     resting_hr = max(35, (72 - vo2 * 0.6) + np.random.normal(0, 3))
#     spo2 = np.clip(spo2 + np.random.normal(0, 0.5), 85, 100)
#     steps = max(1000, int(steps + np.random.normal(0, 300)))
#     hydration = max(800, int(hydration + np.random.normal(0, 150)))
#     sleep_sec = max(3 * 3600, sleep_sec + int(np.random.normal(0, 600)))
#     sleep_rr = int(sleep_rr + np.random.normal(0, 1))
#     sleep_hr = round(sleep_hr + np.random.normal(0, 3), 1)
#     avg_resp = round(avg_resp + np.random.normal(0, 1), 1)
#     max_hr = max(max_hr, int(resting_hr + 20))

#     # --- Derived features ---
#     weight = round(bmi * ((height_cm / 100) ** 2), 1)
#     fitness_age = int(60 - (vo2 * 0.9) + np.random.normal(0, 1.5)) #added 0.9 intead of 0.7 coef to test
#     fitness_age = np.clip(fitness_age, 18, 75)

#     row = {
#         'calendar_date': date.strftime('%Y-%m-%d'),
#         'vo2_max_precise': round(vo2, 1),
#         'steps': steps,
#         'min_heart_rate': int(resting_hr - np.random.normal(10, 3)),
#         'max_heart_rate': max_hr, #int(resting_hr + np.random.normal(70, 8)),
#         'resting_heart_rate': int(resting_hr),
#         'max_respiration': int(avg_resp + np.random.normal(3, 0.5)), #int(np.random.normal(20, 2)),
#         'min_respiration': int(avg_resp - np.random.normal(3, 0.5)), #int(np.random.normal(11, 1)),
#         'avg_waking_respiration': avg_resp,
#         'weight_kg': weight,
#         'fitness_age': fitness_age,
#         'bmi': round(bmi, 1),
#         'hydration_ml': hydration,
#         'avg_spo2': round(spo2, 1),
#         'min_spo2': round(sleep_spo2 - np.random.uniform(1.0, 3.5), 1),
#         'avg_sleep_spo2': round(sleep_spo2, 1),
#         'sleep_time_sec': sleep_sec / 3600,
#         'sleep_avg_respiration': sleep_rr,
#         'sleep_resting_heart_rate': sleep_hr
#     }

#     # --- Label it using the logic-based function ---
#     row['cvd_risk'] = compute_cvd_risk_label(row, age)
#     rows.append(row)

# # --- Create final dataset ---
# df = pd.DataFrame(rows)

# # Optional: numeric label
# df['cvd_risk_numeric'] = df['cvd_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})

# print(df['cvd_risk'].value_counts())

# # Save
# df.to_csv("realistic_cvd_dataset2.csv", index=False)

# print(f"Duplicated values: {df.duplicated().sum()}")

output_folder = "plots_perfil2_pontoA"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv("realistic_cvd_dataset1.csv")

# ✅ Filter only low-risk individuals
df = df[df['cvd_risk'] == 'High']

# Drop non-numeric columns for correlation and boxplot input
df_numeric = df.drop(columns=["calendar_date", "cvd_risk", "cvd_risk_numeric"])

# Set seaborn theme
sns.set_theme(style="whitegrid", palette="muted")

# --- Boxplots ---
for group_name, metrics in metric_groups.items():
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df_numeric, y=metric, ax=ax, color='lightblue')
        ax.set_ylabel("")
        ax.set_xticks([0])
        ax.set_xticklabels([metric], ha='center')
        ax.grid(True)

        annotate_boxplot(ax, df_numeric[metric])

    fig.suptitle(f"{group_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = os.path.join(output_folder, f"{group_name.replace(' ', '_').lower()}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# --- Correlation Heatmap ---
correlation_matrix = df_numeric.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

plt.figure(figsize=(16, 12))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75}
)
plt.title("Heatmap - Correlation between Variables", fontsize=20)
plt.tight_layout()

heatmap_file = os.path.join(output_folder, "correlation_heatmap.png")
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
plt.show()
plt.close()