import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_analysis(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define the columns related to the analysis
    columns_to_analyze = [
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
        'Google DL (Bytes)', 'Google UL (Bytes)',
        'Email DL (Bytes)', 'Email UL (Bytes)',
        'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
        'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
        'Other DL (Bytes)', 'Other UL (Bytes)'
    ]

    # Select the relevant columns
    data = df[columns_to_analyze]

    # Compute the correlation matrix
    correlation_matrix = data.corr()

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix for Application Data')
    plt.savefig('results/graphs/correlation_matrix.png')
    plt.show()

    # Print out the correlation matrix
    print("Correlation Matrix:\n", correlation_matrix)

    # Interpret the findings
    print("\nInterpretation of Correlation Matrix:")
    print("""
    - Correlation values range from -1 to 1. A value close to 1 indicates a strong positive correlation,
      while a value close to -1 indicates a strong negative correlation. A value close to 0 indicates weak or no correlation.
    - For example:
      - High positive correlation between Social Media DL and Social Media UL indicates that as the data usage for downloading increases, the data usage for uploading also increases.
      - High negative correlation between YouTube DL and Gaming DL may indicate that these two applications are used by different sets of users, leading to an inverse relationship.
      - If there is a low correlation between Google UL and Email UL, it suggests that the usage patterns of these two applications are not strongly related.
    """)

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to conduct correlation analysis
correlation_analysis(file_path)
