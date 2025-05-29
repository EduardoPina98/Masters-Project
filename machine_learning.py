import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def fetch_records_for_ML():
    try:
        # Get connection parameters from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT')

        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")
        
        # SQLAlchemy engine connection for easier data handling with Pandas
        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(connection_string)
        
        # Query to select data from the table 'person_record_ml' which is processed in the database
        # The data imported in the database is filtered and selected into a new table.
        # This makes the processing faster and prevents sending unnecssary data to the pre-processing stage area

        query = "SELECT * FROM person_record_ml;"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logging.warning("No data returned from the query.")
            return df
        else:
            logging.info(f"Loaded {len(df)} rows from person_record_ml.")
        
        return df

    except Exception as e:
        logging.error(f"Error fetching data with Pandas: {e}")
        return None

def preprocess_data(df):

    #Naming conventions
    df.rename(columns={
        'vo2maxprecisevalue': 'vo2_max_precise',
        'calendardate': 'calendar_date',
        'minheartrate': 'min_heart_rate',
        'maxheartrate': 'max_heart_rate',
        'restingheartrate': 'resting_heart_rate',
        'highestrespirationvalue': 'max_respiration',
        'lowestrespirationvalue': 'min_respiration',
        'avgwakingrespirationvalue': 'avg_waking_respiration',
        'weight': 'weight_kg',
        'fitnessage': 'fitness_age',
        'hydrationvalueinml': 'hydration_ml',
        'averagespo2': 'avg_spo2',
        'lowestspo2': 'min_spo2',
        'avgsleepspo2': 'avg_sleep_spo2',
        'sleeptimeseconds': 'sleep_time_sec',
        'sleepaveragerespirationvalue': 'sleep_avg_respiration',
        'sleeprestingheartrate': 'sleep_resting_heart_rate'
    }, inplace=True)

    # Convert weight from grams to kg and fitness age with 3 decimal places to 1 decimal place
    df["weight_kg"] = (df["weight_kg"] / 1000).round(1)
    df["fitness_age"] = df["fitness_age"].round(0).astype(int)
    df["max_respiration"] = df["max_respiration"].round(0).astype(int)
    df["min_respiration"] = df["min_respiration"].round(0).astype(int)
    df["sleep_avg_respiration"] = df["sleep_avg_respiration"].round(0).astype(int)
    df["avg_spo2"] = df["avg_spo2"].round(0).astype(int)
    df["min_spo2"] = df["min_spo2"].round(0).astype(int)
    df["avg_sleep_spo2"] = df["avg_sleep_spo2"].round(0).astype(int)
    df["avg_waking_respiration"] = df["avg_waking_respiration"].round(0).astype(int)
    df["hydration_ml"] = df["hydration_ml"].round(0).astype(int)

    # remove id column
    df = df.drop(columns=['id'])

    #remove null or NaN values from any row
    df.dropna(how='any', axis=1, inplace=True)

    print(f"\nTotal number of rows in dataset after cleaning {len(df)}")

    #Check missing dates
    start_date = df['calendar_date'].min()
    end_date = df['calendar_date'].max()
    expected_dates = pd.date_range(start=start_date, end=end_date)

    actual_dates = len(df['calendar_date'])
    missing_count = len(expected_dates) - actual_dates
    total_expected = len(expected_dates)
    missing_percetage = (missing_count / total_expected) * 100

    print(f"Total expected dates: {total_expected}")
    print(f"Missing dates/days: {missing_count}")
    print(f"Missing percentage: {int(missing_percetage)}%\n")

    #Create Target Column
    #2 options_ use weights for each feature giving weight to dependent features or add weights to combined feature calculations
    # disregarding dependant features
    age = 26

    # Example normalized weights (sum = 1)
    weights = {
        'vo2max_bmi': 0.13,
        'vo2max_sleep_time': 0.09,
        'bmi_resting_hr': 0.12,
        'sleep_resting_hr': 0.08,
        'sleep_avg_respiration': 0.07,
        'avg_waking_respiration_resting_hr': 0.1,
        'fitness_age_resting_hr': 0.11,
        'sleep_avg_spo2_sleep_time': 0.08,
        'steps': 0.06,
        'hydration_ml': 0.04,
        'avg_spo2': 0.07,
        'avg_waking_respiration': 0.05
    }

    df['cvd_risk_score'] = 0.0
    df['cvd_risk'] = 'NaN'

    for index, row in df.iterrows():
        score = 0.0

        # Combination 1: vo2max + bmi (validated) BMI: Association between obesity categories with cardiovascular disease and its related risk factors in the MASHAD cohort study population, Journal of Clinical Laboratory Analysis, 2020
        # Articles: Cardiorespiratory fitness, body mass index and mortality: a systematic review and meta-analysis
        # Justification: High fitness reduces CVD risk significantly even with high BMI.
        # Weight article and Justification: Same article. Vo2max and BMI together strongly predict cardiovascular fitness and obesity-related cardiovascular risk.
        if row['vo2_max_precise'] < 41.7 and (row['bmi'] < 18.5 or row['bmi'] > 24.9):
            score += weights['vo2max_bmi']

        # Combination 2: vo2max + sleep time (validated) sleep validate: Association of sleep duration and cardiovascular events, European Heart Journal, 2020
        # Article: Joint association of physical activity and sleep duration with risk of all-cause and cause-specific mortality: a population-based cohort study using accelerometry
        # Justification: High fitness mitigates CVD risk associated with unhealthy sleep of short or long durations.
        # Weight article and Justification: Same article. Both VO₂max (fitness) and sleep duration independently affect cardiovascular recovery; combined effect enhances prediction.

        if row['vo2_max_precise'] < 41.7 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
            score += weights['vo2max_sleep_time']
        
        ## Effect of resting heart rate on the risk of metabolic syndrome in adults: a dose–response meta-analysis (T2D) article
        # Combination 3: bmi + resting heart rate (Validated Value Article) RHR: Association of resting heart rate and physical activity with cardiovascular mortality: A population-based cohort study of Korean adults Journal of sports sciences, 2024
        # Article: Exercise capacity and body mass index - important predictors of change in resting heart rate, BMC Cardiovascular Disorders 2019: High Resting Heart Rate and High BMI Predicted Severe Coronary Atherosclerosis Burden in Patients With Stable Angina Pectoris by SYNTAX Score, Angiology 2018
        # Weight article and Justification: Same first article This study found that increases in BMI were correlated with increases in RHR, even among individuals with normal BMI ranges. The findings suggest that BMI and RHR are interrelated and that their combined elevation may contribute to increased cardiovascular risk.
        if (row['bmi'] < 18.5 or row['bmi'] > 24.9) and row['resting_heart_rate'] > 60:
            score += weights['bmi_resting_hr']

        # Combination 4: avg_waking_respiration + resting_heart_rate (Validated Value Article): Heart/breathing rate ratio (HBR) as a predictor of mortality in critically ill patients, 2024 Heliyon
        # Article: ##############################################
        # Weight article and Justification: Both breathing and heart rate are important, but the relationship between the two are not well studied. It is very uncommon for a
        #significant disturbance of a single physiological parameter to occur in isolation. Thus, NEWS Development and Implementation Group
        #believed multiple physiological parameters is a more robust measure of acute-illness severity than single-parameter scoring systems
        if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20) and row['resting_heart_rate'] > 60:
            score += weights['avg_waking_respiration_resting_hr']
        
        # Combination 5: Fitness Age + Resting Heart Rate (Validated)
        # Article: #########################################################
        # Justification: #########################################################
        # Weight article and Justification: RESTING HEART RATE AND MORTALITY RISK IN HYPERTENSIVE PATIENTS WITH NO ATRIAL FIBRILLATION, Journal of Hypertension 2024: Collectively, these peer-reviewed studies demonstrate that combining fitness level (or “fitness age” derived from VO₂max) and resting heart rate yields a stronger predictive insight into CVD risk than either alone; Protective Role of Resting Heart Rate on All-Cause and Cardiovascular Disease Mortality, Mayo Clinic Proceedings 2013: In
        #practical terms, assessing both a person’s cardiorespiratory fitness (VO₂max/fitness age) and resting
        #heart rate can improve CVD risk stratification. High-risk individuals may be more accurately identified
        #when these two features are evaluated together, as demonstrated by hazard ratios and effect sizes in
        #the studies above.
        if row['fitness_age'] >= age and row['resting_heart_rate'] > 60:
            score += weights['fitness_age_resting_hr']

        #  Respiratory effort during sleep and the rate of prevalent type 2 diabetes in obstructive sleep apnoea (T2D) article
        # Combination 6: sleep_avg_spo2 + sleep_time (validated value article): How Are Sleep Characteristics Related to Cardiovascular Health? Results From the Population‐Based HypnoLaus study, 2019 Journal of the American Heart Association: Higher mean sleep SpO₂
        #strongly associated with better cardiovascular health status, whereas short (<6 h) or long (>8 h)
        #sleep were more frequent in those with poor CV health  
        # Article: #################################################
        # Justification:
        # Weight article and Justification:
        if row['avg_sleep_spo2'] < 95 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
            score += weights['sleep_avg_spo2_sleep_time']

        #individual variables
        # steps (Validated value Article): Do the associations of daily steps with mortality and incident cardiovascular disease differ by sedentary time levels? A device-based cohort study British Journal of Sports Medicine, 2024; 
        # Aricle: Prospective Association of Daily Steps With Cardiovascular Disease: A Harmonized Meta-Analysis
        # Justification: Increased daily steps inversely correlate with CVD risk.
        # Weight article and Justification: Same article. Moderate-to-high evidence supporting physical activity as a protective cardiovascular factor, thus moderately weighted.
        if row['steps'] < 4500:
            score += weights['steps']

        # hydration_ml (Validated Value Article): Fluid and Water Balance: A Scoping Review for the Nordic Nutrition Recommendations, Foods;recommend an adequate intake (AI) of 2.0 liters/day for women and 2.5 liters/day for men, including water from all sources. Given that about 20–30% of water intake comes from food, this translates to a fluid intake of approximately 1.5 to 2.0 liters/day.
        # Article: Middle age serum sodium levels in the upper part of normal range and risk of heart failure, Middle-age high normal serum sodium as a risk factor for accelerated biological aging, chronic diseases, and premature mortality
        # Justification: The article underscores the importance of maintaining good hydration throughout life to potentially slow down the decline in cardiac function and decrease the prevalence of heart failure. While the study does not establish a direct causal relationship, it suggests that poor hydration may increase long-term CVD and heart failure risk.
        # Weight article and Justification: Water intake from foods and beverages and risk of mortality from CVD: the Japan Collaborative Cohort (JACC) Study, 2018:Participants in the highest quintile of total water intake had lower adjusted risk of CVD death
        # Total and drinking water intake and risk of all-cause and cardiovascular mortality: A systematic review and dose-response meta-analysis of prospective cohort studies, 2021: High consumption of total water is associated with a lower risk of CVD mortality.
        if row['hydration_ml'] < 1500:
            score += weights['hydration_ml']
        
        # avg_spo2 (Validated Value Article): Effect of a lower target oxygen saturation range on the risk of hypoxaemia and elevated NEWS2 scores at a university hospital: a retrospective study BMJ Open Respiratory ResearchOpen Access; Non-contact oxygen saturation monitoring for wound healing process using dual-wavelength simultaneous acquisition imaging system Biomedical Engineering Letters (Springer, 2023)
        # Article: ######################################
        # Justification: ######################################
        # Weight article and Justification: Association of hypoxic burden metrics with cardiovascular outcomes in heart failure and sleep-disordered breathing, 2023: were strong independent predictors of the composite outcome of cardiovascular death or HF hospitalization
        if row['avg_spo2'] < 95:
            score += weights['avg_spo2']

        # avg_waking_respiration (Validated Value Article): Respiratory rate and its associations with disease and lifestyle factors in the general population – results from the KORA-FF4 study PloS one, 2025;
        # Article: Respiratory rate predicts outcome after acute myocardial infarction: a prospective cohort study
        # Justification: Elevated respiration rate predicts higher cardiovascular risk.
        # Weight article and Justification: Mean nocturnal respiratory rate predicts cardiovascular and all-cause mortality in community-dwelling older men and women, European Respiratory Journal, 2019:  In adults at rest, any RR between 12 and 20 breathes per minute (bpm) is considered normal; tachypnea is defined as RR greater than 20 bpm
        #more: Abnormal Awake Respiratory Patterns Are Common in Chronic Heart Failure and May Prevent Evaluation of Autonomic Tone by Measures of Heart Rate Variability, 1997: In conclusion, mounting evidence indicates that a persistently high respiratory
        #rate is an independent red flag for coronary heart disease, heart failure, stroke, and CV mortality risk –
        #underlining the adage that “the breath is the indicator of life” in cardiovascular health assessment.

        if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20):
            score += weights['avg_waking_respiration']

        #Sleep Resting Heart Rate (Validated Value Article) Measure by measure: Resting heart rate across the 24-hour cycle (no scopus), Resting heart rate is a population-level biomarker of cardiorespiratory fitness: The Fenland Study
        # Article: Nighttime heart rate and cardiovascular mortality in ICD patients
        # Justification: Higher nighttime heart rate increases CVD mortality risk.
        # Weight article and Justification: Resting Heart Rate as a Cardiovascular Risk Factor in Hypertensive Patients: An Update. Indicates cardiovascular recovery; predictive but weaker in isolation compared to broader combinations.
        if row['sleep_resting_heart_rate'] < 40 or row['sleep_resting_heart_rate'] > 60:
            score += weights['sleep_resting_hr']

        #Sleep Average Respiration (Validated value article): Mean nocturnal respiratory rate predicts cardiovascular and all-cause mortality in community-dwelling older men and women European Respiratory Journal, 2019;
        # Article: Association between nighttime heart rate and cardiovascular mortality in patients with implantable cardioverter-defibrillator: A cohort study
        # Justification: Elevated nocturnal respiration rate is associated with higher CVD risk.
        # Weight article and Justification: Same article. Respiratory effort during sleep and prevalent hypertension in obstructive sleep apnoea. Moderate standalone predictor of cardiovascular and autonomic health, especially related to nocturnal cardiovascular stress.
        if (row['sleep_avg_respiration'] < 14 or row['sleep_avg_respiration'] > 16):
            score += weights['sleep_avg_respiration']

        # Assign the risk score to the 'cvd_risk_score' column for the current row
        df.at[index, 'cvd_risk_score'] = score

        if score <= 0.3:
            df.at[index, 'cvd_risk'] = 'Low'
        elif score >= 0.3 and score <= 0.6:
            df.at[index, 'cvd_risk'] = 'Medium'
        else:
            df.at[index, 'cvd_risk'] = 'High'

    #Instead of using Label Encoder which assigns values on alphabetic order, i create a mapping dictionary with the order i want
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        
    df['cvd_risk'] = df['cvd_risk'].map(risk_mapping)
    df['cvd_risk_score'] = df['cvd_risk_score'].round(2)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df.to_csv('real_data.csv', index=False)
    
    # Dependent features 
    #max_respiration
    #min_respiration
    #max_heart_rate
    #min_heart_rate
    #weight
    #min_spo2

    #other combinations but not related to CVD
    # Min Heart Rate + Min Respiration: Classifying sleep-wake stages through recurrent neural networks using pulse oximetry signals
    # Fitness Age + VO₂ Max: VO₂ max is a direct measure of aerobic capacity and is often used to estimate fitness age.
    # Fitness Age + BMI + resting heart rate: Independent determinants of VO2max

    #Check data distribution
    #df_numeric_cols = df.iloc[:, df.columns != 'calendar_date']
    
    # for column in df_numeric_cols:
    #     plt.figure(figsize=(12, 4))
    #     sns.boxplot(x=df[column])
    #     plt.title(f"Boxplot for {column}")
    #     plt.show()

    # Drop 'calendar_date' to get only numeric columns
    #numeric_cols = df.columns[df.columns != 'calendar_date']

    # Compute correlation matrix
    #corr = df[numeric_cols].corr()

    # Plot the fature correlation heatmap
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.title("Feature Correlation Heatmap")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    # Given the boxplot here is what i found:

    
    #TODO When choosing a multivariate outlier detection method, i need to have in mind 3 steps: data destribution, sample size and number of dimensions
    #The data is not normally distributed so i need to identify which scaling method should i use, the sample is small for the moment and small dimension
    # Compare features with outliers with other variables to verify relanshion ships between the outliers and other variables

    #Feature scaling (normalization) robustScaling because of outliers
    # X = features, y = target (encoded as 0,1,2)
    
    # x_features = df.drop(columns=['cvd_risk_score', 'calendar_date'])

    # #Integrate sintetic data 

    # df_high_risk = pd.read_csv('high_risk_generated_data.csv')

    # df_combined = pd.concat([df_high_risk, x_features], ignore_index=True)

    # y_target = df_combined['cvd_risk']

    # x_features = df_combined.drop(columns=['cvd_risk'])    

    # scaler = RobustScaler()
    # x_scaled = pd.DataFrame(scaler.fit_transform(x_features), columns=x_features.columns)
    
    # Melt the dataframe to long format for seaborn
    # df_melted = x_scaled.melt(var_name='Feature', value_name='Scaled Value')

    # # Create the boxplot
    # plt.figure(figsize=(10, len(x_features) * 0.5))
    # ax = sns.boxplot(y='Feature', x='Scaled Value', data=df_melted)

    # # Annotate outliers
    # for feature in x_features:
    #     values = x_scaled[feature]
    #     q1 = values.quantile(0.25)
    #     q3 = values.quantile(0.75)
    #     iqr = q3 - q1
    #     lower_bound = q1 - 1.5 * iqr
    #     upper_bound = q3 + 1.5 * iqr

    #     outliers = values[(values < lower_bound) | (values > upper_bound)]
    #     for idx in outliers.index:
    #         ax.annotate(f"{values[idx]:.2f}",
    #                     xy=(values[idx], feature),
    #                     xytext=(5, 0),
    #                     textcoords='offset points',
    #                     fontsize=7,
    #                     color='red')

    # plt.title('Boxplot with Outlier Annotations (Robust Scaled Features)')
    # plt.tight_layout()
    # plt.show()

    # # Feature Selection (Will use RFE with randomforest since using svm will not verify for correlated features. Cant also use chi'square since one of the settings to use chi2 is to not have non negative values (RobustScaler) and only works on categorical values and not on continuous values)
    # Define the base classifier
    # rf = RandomForestClassifier(random_state=42)

    # # Define RFE wrapper (no need to fix n_features_to_select here)
    # rfe = RFE(estimator=rf)

    # # Create the pipeline
    # pipeline = Pipeline([
    #     ('feature_selection', rfe),
    #     ('classification', rf)
    # ])

    # # Define the grid of parameters to search
    # param_grid = {
    #     'feature_selection__n_features_to_select': [5, 6, 7, 8, 9],
    #     'classification__n_estimators': [100, 200],
    #     'classification__max_depth': [None, 10, 20],
    #     'classification__min_samples_split': [2, 4]
    # }

    # # Set up GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=pipeline,
    #     param_grid=param_grid,
    #     cv=5,
    #     scoring='accuracy',
    #     n_jobs=-1
    # )

    # # Fit the model
    # grid_search.fit(x_scaled, y_target)

    # # Extract results
    # best_rfe = grid_search.best_estimator_['feature_selection']
    # selected_features = x_scaled.columns[best_rfe.support_]

    # print("Best parameters found:", grid_search.best_params_)
    # print("Selected features:", selected_features.tolist())

    return df


if __name__ == "__main__":

    df = fetch_records_for_ML()
    
    if df is not None:
        print("Data fetched successfully. Proceeding with preprocessing...")

        # Preprocessing the data
        df_processed = preprocess_data(df)
    else:
        print("No data fetched.")
