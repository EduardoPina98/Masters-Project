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
    df.dropna(how='any', axis=0, inplace=True)

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


    #2 options_ use weights for each feature giving weight to dependent features or add weights to combined feature calculations
    # disregarding dependant features

    score = 0.0

    # Combination 1: vo2max + bmi + sleep time(validated)
    # Articles: Is BMI Associated with Cardiorespiratory Fitness? A Cross-Sectional Analysis Among 8470 Apparently Healthy Subjects Aged 18–94 Years from the Low-Lands Fitness Registry
    # Article: The importance of cardiorespiratory fitness and sleep duration in early CVD prevention: BMI, resting heart rate and questions about sleep patterns are suggested in risk assessment of young adults, 18–25 years

    if 18.5 > df['bmi'] < 24.9 & df['vo2maxprecisevalue'] < 3 & df['sleeptimeseconds'] / 3600 < 6:
        score += 0.25

    # Combination 2: sleep_avg_spo2 + sleep_avg_respiration
    if df['avgsleepspo2'] < 94 & df['sleepaveragerespirationvalue'] > 18:
        score += 0.15

    # Combination 3: steps + sleep time
    if df['steps'] < 5000 & df['sleeptimeseconds'] / 3600 < 6:
        score += 0.10

    # Combination 4: average_spo2 + avg_waking_respiration
    if df['averagespo2'] < 94 & df['avgwakingrespirationvalue'] > 20:
        score += 0.10

    # Combination 5: hydration + max respiration
    if df['hydrationvalueinml'] < 1500 & df['highestrespirationvalue'] > 22:
        score += 0.10

    # Combination 6: fitness age + min heart rate
    if df['fitnessage'] > 55 & df['minheartrate'] < 50:
        score += 0.15

    # Combination 7: bmi + resting heart rate
    if df['bmi'] >= 28 & df['restingheartrate'] > 80:
        score += 0.15
    
    # combination 8: bmi and spo2 ()
    
    if score <= 0.3:
        return 'low'
    elif score <= 0.6:
        return 'medium'
    else:
        return 'high'
    
    df['risk_score'] = df.apply(calculate_weighted_risk_score, axis=1)
    df['risk_level'] = df['risk_score'].apply(weighted_risk_label)

    #Check data distribution
    df_numeric_cols = df.iloc[:, df.columns != 'calendar_date']
    
    # for column in df_numeric_cols:
    #     plt.figure(figsize=(12, 4))
    #     sns.boxplot(x=df[column])
    #     plt.title(f"Boxplot for {column}")
    #     plt.show()

    # Drop 'calendar_date' to get only numeric columns
    numeric_cols = df.columns[df.columns != 'calendar_date']

    # Compute correlation matrix
    corr = df[numeric_cols].corr()

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

    # Add target column
    #




    #Feature selection Technique
 

    #TODO: Check other pre-processing techniques
    """
    Encoding categorical variables (no need since i only have numeric values and not categorical values)
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
