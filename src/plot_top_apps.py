import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_apps(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define the application columns
    app_columns = [
        'Social Media DL (Bytes)',
        'Google DL (Bytes)',
        'Email DL (Bytes)',
        'Youtube DL (Bytes)',
        'Netflix DL (Bytes)',
        'Gaming DL (Bytes)'
    ]
    
    # Calculate total data usage for each application by summing the columns
    app_usage = df[app_columns].sum().sort_values(ascending=False)

    # Get the top 3 most used applications
    top_3_apps = app_usage.head(3)

    # Plot the top 3 applications
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_3_apps.index, y=top_3_apps.values, color='blue')  # Simple color instead of palette
    plt.title('Top 3 Most Used Applications by Download Data (Bytes)')
    plt.xlabel('Application')
    plt.ylabel('Total Download Bytes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top_3_apps.png')
    plt.close()  # Use plt.close() instead of plt.show() in non-interactive environments
    
    print("Visualization saved: top_3_apps.png")

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to plot top 3 applications
plot_top_apps(file_path)
