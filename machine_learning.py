import os
import pandas as pd
from sqlalchemy import create_engine
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

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
    df.dropna(how='any', axis=0, inplace=True)

    print(f"\nTotal number of rows in dataset after cleaning {len(df)}")

    # Add target column



    #Check missing dates
    start_date = df['calendar_date'].min()
    end_date = df['calendar_date'].max()
    expected_dates = pd.date_range(start=start_date, end=end_date)

    actual_dates = len(df['calendar_date'])
    missing_count = len(expected_dates) - actual_dates
    total_expected = len(expected_dates)
    missing_percetage = (missing_count / total_expected) * 100

    print(f"Total expected dates {total_expected}")
    print(f"Missing dates {missing_count}")
    print(f"Missing percentage {int(missing_percetage)}%\n")

    #Check if data in each column is normalized

    numeric_cols = [col for col in df.columns if col != 'calendar_date' and pd.api.types.is_numeric_dtype(df[col])]

    # for column in numeric_cols:
    #     plt.figure(figsize=(12, 4))

    #     plt.subplot(1,2,1)
    #     df[column].hist(bins=30)
    #     plt.title(f"Outliers {column}")

    #     plt.show()

    #Check if the data is tilted to one side with skewness
    skewness = df[numeric_cols].skew()
    print(skewness)

    #The data is not normally distributed so i need to identify which scaling method should i use
    #Feature scaling (normalization) robustScaling

    features_scaled = df.drop(columns=['calendar_date'])

    scaler = RobustScaler()

    scaled_data = scaler.fit_transform(features_scaled)

    df_scaled = pd.DataFrame(scaled_data, columns=features_scaled.columns)


    # Multivariate method to find outliers

    # Remove/Outliers
 

    #TODO: Check other pre-processing techniques
    """
    Feature scaling (normalization)
    Encoding categorical variables (no need since i only have numeric values and not categorical values)
    Dealing with outliers or any other domain-specific transformations
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
