import os
import pandas as pd
from sqlalchemy import create_engine
import logging
from dotenv import load_dotenv

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
        
        # Create SQLAlchemy engine for easier data handling with Pandas
        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(connection_string)
        
        # Query to select data from the table
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
    'hydrationvalueinmL': 'hydration_ml',
    'averagespo2': 'avg_spo2',
    'lowestspo2': 'min_spo2',
    'avgsleepspo2': 'avg_sleep_spo2',
    'sleeptimeseconds': 'sleep_time_sec',
    'sleepaveragerespirationvalue': 'sleep_avg_respiration',
    'sleeprestingheartrate': 'sleep_resting_heart_rate'
    }, inplace=True)

    # Convert weight from grams to kg and fitness age with 3 decimal places to 1 decimal
    df["weight_kg"] = (df["weight_kg"] / 1000).round(1)
    df["fitness_age"] = df["fitness_age"].round(1)

    #TODO: Check other pre-processing techniques
    """
    Handling missing values
    Feature scaling (e.g., standardization or normalization)
    Encoding categorical variables
    Feature extraction (e.g., from time-based data)
    Dropping unneeded columns
    Dealing with outliers or any other domain-specific transformations
    """

    return df


if __name__ == "__main__":

    df = fetch_records_for_ML()
    
    if df is not None:
        print("Data fetched successfully. Proceeding with preprocessing...")

        # Preprocess the data here
        df_processed = preprocess_data(df)
    else:
        print("No data fetched.")
