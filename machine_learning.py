import os
import pandas as pd
from sqlalchemy import create_engine
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns

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
    df["fitness_age"] = df["fitness_age"].round(1)

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

    df['cvd_risk_score'] = 0
    df['cvd_risk'] = 'NaN'

    for index, row in df.iterrows():
        score = 0.0

        # Combination 1: vo2max + bmi (validated)
        # Articles: Cardiorespiratory fitness, body mass index and mortality: a systematic review and meta-analysis
        if row['vo2_max_precise'] < 38.0 and (row['bmi'] < 18.5 or row['bmi'] > 24.9):
            score += weights['vo2max_bmi']

        # Combination 2: vo2max + sleep time (validated)
        # Article: The importance of cardiorespiratory fitness and sleep duration in early CVD prevention: BMI, resting heart rate and questions about sleep patterns are suggested in risk assessment of young adults, 18–25 years
        if row['vo2_max_precise'] < 38.0 and (row['sleep_time_sec'] < 21600  or row['sleep_time_sec'] > 28800):
            score += weights['vo2max_sleep_time']

        # Combination 3: bmi + resting heart rate (validated)
        # Article: Effect of resting heart rate on the risk of metabolic syndrome in adults: a dose–response meta-analysis
        if (row['bmi'] < 18.5 or row['bmi'] > 24.9) and row['resting_heart_rate'] > 70:
            score += weights['bmi_resting_hr']

        # Combination 4: avg_waking_respiration + resting_heart_rate (validated Awake)
        # Article: Predictive Modeling of Heart Rate from Respiratory Signals at Rest in Young Healthy Humans
        if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20) and row['resting_heart_rate'] > 70:
            score += weights['avg_waking_respiration_resting_hr']
        
        # Combination 5: Fitness Age + Resting Heart Rate (validated)
        # Article: Resting heart rate is a population-level biomarker of cardiorespiratory fitness: The Fenland Study
        if row['fitness_age'] > age and row['resting_heart_rate'] > 70:
            score += weights['fitness_age_resting_hr']

        # Combination 6: sleep_avg_spo2 + sleep_time (validated)
        # Article: Respiratory effort during sleep and the rate of prevalent type 2 diabetes in obstructive sleep apnoea
        if row['avg_sleep_spo2'] < 95.0 and (row['sleep_time_sec'] / 3600 < 6 or row['sleep_time_sec'] / 3600 > 9):
            score += weights['sleep_avg_spo2_sleep_time']

        #individual variables
        #steps
        if row['steps'] < 5000:
            score += weights['steps']

        #hydration_ml 
        if row['hydration_ml'] < 1250:
            score += weights['hydration_ml']
        #avg_spo2
        if row['avg_spo2'] < 95.0:
            score += weights['avg_spo2']

        #avg_waking_respiration 
        if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20):
            score += weights['avg_waking_respiration']

        #Sleep Resting Heart Rate
        if row['sleep_resting_heart_rate'] < 30 or row['sleep_resting_heart_rate'] > 60:
            score += weights['sleep_resting_hr']

        #Sleep Average Respiration
        if (row['sleep_avg_respiration'] < 11 or row['sleep_avg_respiration'] > 18):
            score += weights['sleep_avg_respiration']

        # Assign the risk score to the 'cvd_risk_score' column for the current row
        df.at[index, 'cvd_risk_score'] = score


        if score <= 0.3:
            df.at[index, 'cvd_risk'] = 'Low'
        elif score <= 0.6:
            df.at[index, 'cvd_risk'] = 'Medium'
        else:
            df.at[index, 'cvd_risk'] = 'High'

    print(df)
    
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

    
    # When choosing a multivariate outlier detection method, i need to have in mind 3 steps: data destribution, sample size and number of dimensions
    #The data is not normally distributed so i need to identify which scaling method should i use, the sample is small for the moment and small dimension
    #Feature scaling (normalization) robustScaling because of outliers

    numeric_cols = df.select_dtypes(include='number').columns
    df_numeric = df[numeric_cols]

    # Apply RobustScaler
    #TODO Depois de normalizar os dados, criar box plot que inclua todas as features.
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_cols)
    
    # Melt the dataframe to long format for seaborn
    df_melted = df_scaled.melt(var_name='Feature', value_name='Scaled Value')

    # Create the boxplot
    plt.figure(figsize=(10, len(numeric_cols) * 0.5))
    ax = sns.boxplot(y='Feature', x='Scaled Value', data=df_melted)

    # Annotate outliers
    for feature in numeric_cols:
        values = df_scaled[feature]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = values[(values < lower_bound) | (values > upper_bound)]
        for idx in outliers.index:
            ax.annotate(f"{values[idx]:.2f}",
                        xy=(values[idx], feature),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=7,
                        color='red')

    plt.title('Boxplot with Outlier Annotations (Robust Scaled Features)')
    plt.tight_layout()
    plt.show()

    # Apply LOF unsupervised
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1, novelty=False)
    outlier_flags = lof.fit_predict(df_scaled)
    
    # Add the results to DataFrame (outlier = -1)
    df['outlier'] = outlier_flags

    # Plot and color-code by outlier status
    # sns.scatterplot(data=df, x='steps', y='vo2_max_precise', hue='outlier', palette={1: "blue", -1: "red"})
    # plt.title("Steps vs VO2 Max with Outliers Highlighted")
    # plt.xlabel("Steps")
    # plt.ylabel("VO2 Max")
    # plt.grid(True)
    # plt.show()


    # Compare features with outliers with other variables to verify relanshion ships between the outliers and other variables






    #Feature selection Technique
 

    #TODO: Check other pre-processing techniques
    """
    Encoding categorical labels for target
    """

    return df


if __name__ == "__main__":

    df = fetch_records_for_ML()
    
    if df is not None:
        print("Data fetched successfully. Proceeding with preprocessing...")

        # Preprocessing the data
        df_processed = preprocess_data(df)
    else:
        print("No data fetched.")
