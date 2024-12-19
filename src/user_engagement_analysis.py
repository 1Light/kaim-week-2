import pandas as pd
import os

def user_engagement_analysis(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Check the first few rows to understand the structure of the data
    # print(df.head())

    # Aggregate metrics per customer ID (MSISDN)
    # 1. Sessions Frequency (counting the number of sessions per customer)
    sessions_frequency = df.groupby('MSISDN/Number').size().reset_index(name='sessions_frequency')

    # 2. Duration of the session (sum of 'Dur. (ms)' per customer)
    session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='total_session_duration')

    # 3. Total Traffic (download + upload bytes per customer)
    total_traffic = df.groupby('MSISDN/Number').agg({
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()

    # Add the total traffic column
    total_traffic['total_traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']

    # Merge all metrics on 'MSISDN/Number'
    engagement_data = pd.merge(sessions_frequency, session_duration, on='MSISDN/Number')
    engagement_data = pd.merge(engagement_data, total_traffic[['MSISDN/Number', 'total_traffic']], on='MSISDN/Number')

    # Report top 10 customers for each engagement metric
    top_10_sessions = engagement_data.nlargest(10, 'sessions_frequency')
    top_10_duration = engagement_data.nlargest(10, 'total_session_duration')
    top_10_traffic = engagement_data.nlargest(10, 'total_traffic')

    # Print the top 10 customers for each metric
    print("Top 10 Customers by Sessions Frequency:")
    print(top_10_sessions[['MSISDN/Number', 'sessions_frequency']])

    print("\nTop 10 Customers by Total Session Duration:")
    print(top_10_duration[['MSISDN/Number', 'total_session_duration']])

    print("\nTop 10 Customers by Total Traffic:")
    print(top_10_traffic[['MSISDN/Number', 'total_traffic']])

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to perform user engagement analysis
user_engagement_analysis(file_path)
