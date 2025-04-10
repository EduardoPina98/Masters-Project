import json
import psycopg2
import logging
import os
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

# Configure logging for error tracking
logging.basicConfig(filename='db_insert_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def insert_user_metrics_data(file_path):
    try:

        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure the environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")
        
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Check if the data is a list of lists and insert each record
        for inner_list in data:
            for record in inner_list:  # Each record is a dictionary
                try:
                    user_id = record['userId']  # Access 'userId' from the current record (which is a dictionary)
                    if not isinstance(user_id, int):
                        raise ValueError(f"Invalid userId {user_id}. It should be an integer.")

                    # Check if the calendarDate exists and is in the correct format
                    calendar_date = record.get('generic', {}).get('calendarDate', None)
                    if not calendar_date:
                        raise ValueError(f"Missing or invalid calendarDate for userId {user_id}.")

                    details = json.dumps(record)  # Convert the entire record into a JSON string

                    # Use INSERT with a check for unique calendarDate
                    cursor.execute("""
                        INSERT INTO user_metrics (userId, details)
                        VALUES (%s, %s)
                        ON CONFLICT ((details->'generic'->>'calendarDate')) DO NOTHING;
                    """, (user_id, details))

                    # Commit after every successful insert
                    conn.commit()

                    if cursor.rowcount > 0:  # Only log if the record was inserted
                        conn.commit()
                        logging.info(f"Inserted record for calendarDate {calendar_date}")
                    else:
                        logging.warning(f"Skipped duplicate record for calendarDate {calendar_date}")

                except (ValueError, KeyError) as e:
                    # Log validation errors
                    logging.error(f"Error in record {record}: {e}")
                    continue
                except Exception as e:
                    # Log unexpected errors
                    logging.error(f"Unexpected error inserting record {record}: {e}")
                    conn.rollback()

        logging.info("Finished processing JSON file.")

    except (psycopg2.DatabaseError, Exception) as e:
        logging.error(f"Database error: {e}")
    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()

def insert_steps_data(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure all required variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")

        # Connect to PostgreSQL
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Check if data is a list of lists and process each record
        for inner_list in data:
            for record in inner_list:  # Each record is a dictionary
                try:
                    # Extract and validate calendarDate
                    calendar_date = record.get("calendarDate")
                    if not calendar_date:
                        raise ValueError(f"Missing or invalid calendarDate in record: {record}")

                    details = json.dumps(record)  # Convert dictionary to JSON string

                    # Use INSERT with ON CONFLICT to avoid duplicate calendarDate entries
                    cursor.execute("""
                        INSERT INTO steps_data (details)
                        VALUES (%s)
                        ON CONFLICT ((details->>'calendarDate')) DO NOTHING;
                    """, (details,))

                    # Commit after every successful insert
                    conn.commit()

                    if cursor.rowcount > 0:  # Only log if the record was inserted
                        conn.commit()
                        logging.info(f"Inserted record for calendarDate {calendar_date}")
                    else:
                        logging.warning(f"Skipped duplicate record for calendarDate {calendar_date}")

                except (ValueError, KeyError) as e:
                    # Log validation errors
                    logging.error(f"Error in record {record}: {e}")
                    continue
                except Exception as e:
                    # Log unexpected errors
                    logging.error(f"Unexpected error inserting record {record}: {e}")
                    conn.rollback()

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Critical error: {e}")

    finally:
        # Close the database connection
        if 'conn' in locals() and conn is not None:
            conn.close()
            logging.info("Database connection closed.")

def insert_hydration_data(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")

        # Connect to PostgreSQL
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Process each record in the JSON array
        for record in data:
            try:
                user_id = record.get("userId")
                calendar_date = record.get("calendarDate")

                if not user_id or not calendar_date:
                    raise ValueError(f"Missing userId or calendarDate in record: {record}")

                details = json.dumps(record)  # Convert dictionary to JSON string

                # Use INSERT with ON CONFLICT to prevent duplicate calendarDate entries
                cursor.execute("""
                    INSERT INTO hydration_data (userId, details)
                    VALUES (%s, %s)
                    ON CONFLICT ((details->>'calendarDate')) DO NOTHING;
                """, (user_id, details))

                if cursor.rowcount > 0:
                    conn.commit()
                    logging.info(f"Inserted record for userId {user_id}, calendarDate {calendar_date}")
                else:
                    logging.warning(f"Skipped duplicate record for calendarDate {calendar_date}")

            except (ValueError, KeyError) as e:
                logging.error(f"Error in record {record}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error inserting record {record}: {e}")
                conn.rollback()

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Critical error: {e}")

    finally:
        # Close the database connection
        if 'conn' in locals() and conn is not None:
            conn.close()
            logging.info("Database connection closed.")

def insert_respiration_data(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")

        # Connect to PostgreSQL
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Process each record in the JSON array
        for record in data:
            try:
                user_profile_pk = record.get("userProfilePK")
                calendar_date = record.get("calendarDate")

                if not user_profile_pk or not calendar_date:
                    raise ValueError(f"Missing userProfilePK or calendarDate in record: {record}")

                details = json.dumps(record)  # Convert dictionary to JSON string

                # Use INSERT with ON CONFLICT to prevent duplicate calendarDate entries
                cursor.execute("""
                    INSERT INTO respiration_data (userProfilePK, details)
                    VALUES (%s, %s)
                    ON CONFLICT ((details->>'calendarDate')) DO NOTHING;
                """, (user_profile_pk, details))

                if cursor.rowcount > 0:
                    conn.commit()
                    logging.info(f"Inserted record for userProfilePK {user_profile_pk}, calendarDate {calendar_date}")
                else:
                    logging.warning(f"Skipped duplicate record for calendarDate {calendar_date}")

            except (ValueError, KeyError) as e:
                logging.error(f"Error in record {record}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error inserting record {record}: {e}")
                conn.rollback()

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Critical error: {e}")

    finally:
        # Close the database connection
        if 'conn' in locals() and conn is not None:
            conn.close()
            logging.info("Database connection closed.")

def insert_pulse_ox_data(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure the environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")

        # Connect to PostgreSQL
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Insert each record
        for record in data:
            try:
                user_profile_pk = record.get("userProfilePK")
                calendar_date = record.get("calendarDate")

                if not user_profile_pk or not calendar_date:
                    raise ValueError(f"Missing userProfilePK or calendarDate in record: {record}")

                # Convert the entire record into JSON format
                details = json.dumps(record)

                # Insert into table with ON CONFLICT for calendarDate
                cursor.execute("""
                    INSERT INTO pulse_ox_data (userProfilePK, details)
                    VALUES (%s, %s)
                    ON CONFLICT ((details->>'calendarDate')) DO NOTHING;
                """, (user_profile_pk, details))

                # Commit each insert
                if cursor.rowcount > 0:
                    conn.commit()
                    logging.info(f"Inserted record for userProfilePK {user_profile_pk}, calendarDate {calendar_date}")
                else:
                    logging.warning(f"Skipped duplicate record for calendarDate {calendar_date}")

            except (ValueError, KeyError) as e:
                logging.error(f"Error in record {record}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error inserting record {record}: {e}")
                conn.rollback()

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Database connection error: {e}")

    finally:
        # Close the database connection
        if 'conn' in locals() and conn is not None:
            conn.close()
            logging.info("Database connection closed.")

def insert_fitness_data(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure the environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")
        
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Insert data into the fitness_data table
        for record in data:
            try:
                details = json.dumps(record)  # Entire record as JSON

                # Try inserting the data into the database
                cursor.execute("""
                    INSERT INTO fitness_data (details)
                    VALUES (%s)
                    ON CONFLICT ((details->>'lastUpdated'))
                    DO NOTHING
                """, (details,))

                # Verify if any rows were affected (i.e., inserted or updated)
                if cursor.rowcount > 0:
                    # Commit after successful insert or update
                    conn.commit()
                    logging.info(f"Inserted record with lastUpdated {record['lastUpdated']}")
                else:
                    # Log warning if no row was inserted/updated (duplicate entry)
                    logging.warning(f"Skipped duplicate record for lastUpdated {record['lastUpdated']}")

            except Exception as e:
                # Log unexpected errors
                logging.error(f"Unexpected error inserting record {record}: {e}")
                conn.rollback()  # Rollback transaction for this insert

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def insert_daily_weight(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure the environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")
        
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Insert data into the daily_weight table
        for record in data:
            try:
                # Extract the endDate from the JSON (ensure it's in the proper format)
                end_date = record.get('endDate')  # Assuming endDate exists in the JSON data
                details = json.dumps(record)  # Entire record as JSON

                # Try inserting the data into the database
                cursor.execute("""
                    INSERT INTO daily_weight (details, end_date)
                    VALUES (%s, %s)
                    ON CONFLICT (end_date)  -- Conflict based on the `end_date` column
                    DO NOTHING;
                """, (details, end_date))

                # Commit after every successful insert
                if cursor.rowcount > 0:
                    conn.commit()
                    logging.info(f"Inserted record with endDate {end_date}")
                else:
                    logging.warning(f"Skipped duplicate record for endDate {end_date}")

            except Exception as e:
                # Log unexpected errors
                logging.error(f"Unexpected error inserting record {record}: {e}")
                conn.rollback()  # Rollback transaction for this insert

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def insert_daily_activity_data(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure the environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")
        
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Insert data into the daily_activity_data table
        for record in data:
            try:
                details = json.dumps(record)  # Entire record as JSON

                # Try inserting the data into the database
                cursor.execute("""
                    INSERT INTO daily_activity_data (details)
                    VALUES (%s)
                    ON CONFLICT (calendar_date) 
                    DO NOTHING
                """, (details,))

                # Commit the transaction if one or more rows are inserted
                if cursor.rowcount > 0:
                    conn.commit()
                    logging.info(f"Inserted record for calendarDate {record['calendarDate']}")
                else:
                    logging.warning(f"Skipped duplicate record for calendarDate {record['calendarDate']}")

            except Exception as e:
                # Log unexpected errors
                logging.error(f"Unexpected error inserting record {record}: {e}")
                conn.rollback()  # Rollback transaction for this insert

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def insert_daily_sleep_data(file_path):
    try:
        # Fetch database connection details from environment variables
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')

        # Ensure the environment variables are set
        if not all([dbname, user, password, host]):
            raise ValueError("Database connection attributes are missing from environment variables.")
        
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Insert data into the sleep_data table
        for record in data:
            try:
                details = json.dumps(record)  # Entire record as JSON

                # Extract calendarDate from the record
                calendar_date = record['dailySleepDTO']['calendarDate']

                # Try inserting the data into the database
                cursor.execute("""
                    INSERT INTO sleep_data (details)
                    VALUES (%s)
                    ON CONFLICT (calendar_date) 
                    DO NOTHING
                """, (details,))

                # Commit the transaction if one or more rows are inserted
                if cursor.rowcount > 0:
                    conn.commit()
                    logging.info(f"Inserted record for calendarDate {calendar_date}")
                else:
                    logging.warning(f"Skipped duplicate record for calendarDate {calendar_date}")

            except Exception as e:
                # Log unexpected errors
                logging.error(f"Unexpected error inserting record {record}: {e}")
                conn.rollback()  # Rollback transaction for this insert

        logging.info("Finished processing JSON file.")

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

#TODO Create new connection to database and execute select bulk of the table for ML
#TODO Create new table and select parameters 


# Call the function to insert data
#insert_user_metrics_data('userMetrics.json')
#insert_hydration_data('hydration.json')
#insert_pulse_ox_data('pulseOx.json')
#insert_fitness_data('fitness_age.json')
#insert_daily_activity_data('stats_and_body_composition.json')
insert_daily_sleep_data('sleeps.json')

#USE UNLESS ACTIVITY_DATA STOPS READING WEIGHT, STEPS, RESPIRATION
#insert_daily_weight('daily_weight.json')
#insert_steps_data('steps_daily.json')
#insert_respiration_data('respiration.json')
