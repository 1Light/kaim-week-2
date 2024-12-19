import pandas as pd
import os
import matplotlib.pyplot as plt

def analyze_user_behavior(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Step 1: Ensure 'IMSI' column exists for user aggregation
    if 'IMSI' not in df.columns:
        print("Error: 'IMSI' column is missing in the dataset.")
        return
    
    # Step 2: Aggregating the relevant information per IMSI (user)
    user_behavior = df.groupby('IMSI').agg(
        number_of_sessions=('Dur. (ms)', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_download_data=('Total DL (Bytes)', 'sum'),
        total_upload_data=('Total UL (Bytes)', 'sum'),
    ).reset_index()

    # Add the total_data_volume column
    user_behavior['total_data_volume'] = user_behavior['total_download_data'] + user_behavior['total_upload_data']

    # Step 3: Checking for any missing values in the aggregated columns
    if user_behavior.isnull().sum().any():
        print(f"Warning: Some missing values found in the aggregated data. Handling missing values by replacing with the mean.")
        user_behavior.fillna(user_behavior.mean(), inplace=True)

    # Step 4: Generating a plot to visualize the total data volume per IMSI (user)
    fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size
    user_behavior.plot(kind='bar', x='IMSI', y='total_data_volume', ax=ax, color='#4CAF50')
    ax.set_title('Total Data Volume per User (IMSI)')
    ax.set_xlabel('User (IMSI)')
    ax.set_ylabel('Total Data Volume (Bytes)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate labels for readability

    # Step 5: Create the 'results/user_behavior' folder if it doesn't exist
    os.makedirs('results/user_behavior', exist_ok=True)

    # Save the plot as an image in the specified folder
    plt.tight_layout()  # Adjust layout to avoid cutting off labels
    plt.savefig('results/user_behavior/total_data_volume_per_user.png', dpi=300)

    print("User behavior analysis completed and saved as an image.")

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to analyze user behavior and save the plot
analyze_user_behavior(file_path)