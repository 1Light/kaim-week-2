import pandas as pd
import os

def analyze_basic_metrics(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Describe the basic metrics for numerical variables
    print("\nBasic Metrics for Numerical Variables:")
    descriptive_stats = df.describe(include=[float, int]).T
    descriptive_stats['median'] = df.median(numeric_only=True)  # Add median to the statistics
    descriptive_stats['std_dev'] = descriptive_stats['std']  # Include standard deviation for clarity

    # Rename columns for clarity in output
    descriptive_stats = descriptive_stats.rename(columns={
        "mean": "Mean",
        "50%": "Median",
        "std": "Standard Deviation",
        "min": "Minimum",
        "max": "Maximum"
    })

    print(descriptive_stats)

    # Save the descriptive statistics to a CSV file for documentation
    os.makedirs('results', exist_ok=True)
    descriptive_stats.to_csv('results/basic_metrics.csv', index=True)
    print("\nDescriptive statistics saved to 'results/basic_metrics.csv'.")

    # Explain the importance of these metrics
    print("\nImportance of Metrics:")
    print("""
    - Mean: Provides an average value, useful for understanding the central tendency of data.
    - Median: Gives the middle value, useful for datasets with skewed distributions.
    - Standard Deviation: Indicates the spread of the data, helpful in identifying variability.
    - Minimum and Maximum: Show the range of data, important for detecting outliers.
    - Count: Highlights the total number of non-null values, useful for understanding data completeness.
    """)

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to analyze basic metrics
analyze_basic_metrics(file_path)
