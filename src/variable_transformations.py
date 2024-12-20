import pandas as pd
import os
import numpy as np

def variable_transformations(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Ensure the 'Dur. (ms)', 'Total UL (Bytes)', and 'Total DL (Bytes)' columns exist
    required_columns = ['Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: '{col}' column is missing in the dataset.")
            return

    # Compute the total duration per user (IMSI) and the total data volume
    df['total_data_volume'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    user_aggregates = df.groupby('IMSI').agg(
        total_duration=('Dur. (ms)', 'sum'),
        total_data_volume=('total_data_volume', 'sum')
    ).reset_index()

    # Segment users into deciles based on total duration
    user_aggregates['duration_decile'] = pd.qcut(user_aggregates['total_duration'], 10, labels=False)

    # Compute total data volume per decile class
    decile_data = user_aggregates.groupby('duration_decile').agg(
        total_data_volume=('total_data_volume', 'sum')
    ).reset_index()

    print("\nTotal Data Volume per Decile Class:")
    print(decile_data)

    # Save the decile data to a CSV file
    os.makedirs('results', exist_ok=True)
    decile_data.to_csv('results/decile_data.csv', index=False)
    print("Decile data saved to 'results/decile_data.csv'.")

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function for variable transformations
variable_transformations(file_path)
