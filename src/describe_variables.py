import pandas as pd
import os

def describe_variables(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Describe variables
    print("Data Types and Non-Null Counts:")
    print(df.info())  # Provides data types and non-null counts

    print("\nSummary Statistics of Numerical Variables:")
    print(df.describe())  # Summary statistics for numerical variables

    print("\nUnique Values for Categorical Variables:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"Column: {col}")
        print(df[col].value_counts(), "\n")

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to describe variables
describe_variables(file_path)