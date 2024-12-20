import pandas as pd
import os

class DataDescriber:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def describe_data_types_and_non_null_counts(self):
        """Describe data types and non-null counts."""
        print("Data Types and Non-Null Counts:")
        print(self.df.info())  # Provides data types and non-null counts

    def describe_numerical_variables(self):
        """Provide summary statistics for numerical variables."""
        print("\nSummary Statistics of Numerical Variables:")
        print(self.df.describe())  # Summary statistics for numerical variables

    def describe_categorical_variables(self):
        """Display unique values for categorical variables."""
        print("\nUnique Values for Categorical Variables:")
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"Column: {col}")
            print(self.df[col].value_counts(), "\n")

    def describe(self):
        """Execute the entire variable description process."""
        self.describe_data_types_and_non_null_counts()
        self.describe_numerical_variables()
        self.describe_categorical_variables()

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the DataDescriber class and run the description process
data_describer = DataDescriber(file_path)
data_describer.describe()