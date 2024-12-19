import pandas as pd
import os

def aggregate_traffic_per_application(file_path):
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Select columns related to different applications
    app_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 
        'Other DL (Bytes)', 'Total DL (Bytes)', 
        'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)', 
        'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 
        'Other UL (Bytes)', 'Total UL (Bytes)'
    ]
    
    # Aggregate traffic per user (MSISDN/Number) and per application (downlink and uplink traffic)
    # Here we consider both downlink and uplink for each app
    app_traffic = pd.DataFrame()
    
    for app_column in app_columns:
        app_name = app_column.split()[0]  # Extract the app name (e.g., 'Social', 'Google', etc.)
        app_traffic[app_name] = df.groupby('MSISDN/Number')[app_column].sum()

    # Now, find the top 10 most engaged users for each application
    top_10_users = {}
    
    for app_name in app_traffic.columns:
        # Sort by traffic (descending) and take top 10 users
        top_10_users[app_name] = app_traffic[app_name].sort_values(ascending=False).head(10)

    # Create a folder to save results
    output_dir = 'results/aggregated_traffic'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results for each application in a CSV
    for app_name, top_users in top_10_users.items():
        top_users.to_csv(f'{output_dir}/{app_name}_top_10_users.csv', header=True)

    print("Top 10 most engaged users for each application have been saved.")

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to aggregate user traffic per application
aggregate_traffic_per_application(file_path)
