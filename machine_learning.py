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

    print(f"\nTotal number of rows in dataset {len(df)}")
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

    # remove id column
    df = df.drop(columns=['id'])

    #remove null or NaN values from any row
    df.dropna(how='any', axis=0, inplace=True)

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
    df['cvd_risk'] = 'Nan'

    for index, row in df.iterrows():
        score = 0.0
        
        if row['vo2_max_precise'] < 41.7 and (row['bmi'] < 18.5 or row['bmi'] > 24.9):
            score += weights['vo2max_bmi']

        if row['vo2_max_precise'] < 41.7 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
            score += weights['vo2max_sleep_time']
        
        if (row['bmi'] < 18.5 or row['bmi'] > 24.9) and row['resting_heart_rate'] > 60:
            score += weights['bmi_resting_hr']

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

        if row['sleep_resting_heart_rate'] < 40 or row['sleep_resting_heart_rate'] > 60:
            score += weights['sleep_resting_hr']

        if (row['sleep_avg_respiration'] < 14 or row['sleep_avg_respiration'] > 16):
            score += weights['sleep_avg_respiration']

        # Assign the risk score to the 'cvd_risk_score' column for the current row
        df.at[index, 'cvd_risk_score'] = score

        if score <= 0.3:
            df.at[index, 'cvd_risk'] = 'Low'
        elif score >= 0.3 and score < 0.5:
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


    return df


if __name__ == "__main__":

    df = fetch_records_for_ML()
    
    if df is not None:
        print("Data fetched successfully. Proceeding with preprocessing...")

        # Preprocessing the data
        df_processed = preprocess_data(df)
    else:
        print("No data fetched.")
