def compute_cvd_risk_label(row, age):
    """
    Computes a cardiovascular disease (CVD) risk label based on multiple physiological conditions.
    Returns 'Low', 'Medium', or 'High' risk.
    """
    weights = {
        'vo2max_bmi': 0.13,
        'vo2max_sleep_time': 0.09,
        'bmi_resting_hr': 0.12,
        'sleep_resting_hr': 0.08,
        'sleep_avg_respiration': 0.07,
        'avg_waking_respiration_resting_hr': 0.10,
        'fitness_age_resting_hr': 0.11,
        'sleep_avg_spo2_sleep_time': 0.08,
        'steps': 0.06,
        'hydration_ml': 0.04,
        'avg_spo2': 0.07,
        'avg_waking_respiration': 0.05
    }

    score = 0

    if row['vo2_max_precise'] < 41.7 and (row['bmi'] < 18.5 or row['bmi'] > 24.9):
        score += weights['vo2max_bmi']

    if row['vo2_max_precise'] < 41.7 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
        score += weights['vo2max_sleep_time']

    if (row['bmi'] < 18.5 or row['bmi'] > 24.9) and row['resting_heart_rate'] > 60:
        score += weights['bmi_resting_hr']

    if row['sleep_resting_heart_rate'] < 40 or row['sleep_resting_heart_rate'] > 60:
        score += weights['sleep_resting_hr']

    if row['sleep_avg_respiration'] < 14 or row['sleep_avg_respiration'] > 16:
        score += weights['sleep_avg_respiration']

    if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20) and row['resting_heart_rate'] > 60:
        score += weights['avg_waking_respiration_resting_hr']

    if row['fitness_age'] >= age and row['resting_heart_rate'] > 60:
        score += weights['fitness_age_resting_hr']

    if row['avg_sleep_spo2'] < 95 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
        score += weights['sleep_avg_spo2_sleep_time']

    if row['steps'] < 4500:
        score += weights['steps']

    if row['hydration_ml'] < 1500:
        score += weights['hydration_ml']

    if row['avg_spo2'] < 95:
        score += weights['avg_spo2']

    if row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20:
        score += weights['avg_waking_respiration']

    # Return label based on total score
    if score <= 0.3:
        return 'Low'
    elif score <= 0.6:
        return 'Medium'
    else:
        return 'High'