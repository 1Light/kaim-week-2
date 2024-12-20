import pandas as pd
import os

class BasicMetricsAnalyzer:
    def __init__(self, file_path):
        # Initialize with the file path and load the data
        self.file_path = file_path
        self.df = None
        self.descriptive_stats = None

    def load_data(self):
        # Load the CSV file into a pandas DataFrame
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")

    def analyze_metrics(self):
        # Ensure all columns are printed in the terminal
        pd.set_option('display.max_columns', None)

        # Describe the basic metrics for numerical variables
        print("\nBasic Metrics for Numerical Variables:")
        self.descriptive_stats = self.df.describe(include=[float, int]).T
        self.descriptive_stats['median'] = self.df.median(numeric_only=True)  # Add median to the statistics
        self.descriptive_stats['std_dev'] = self.descriptive_stats['std']  # Include standard deviation for clarity

        # Rename columns for clarity in output
        self.descriptive_stats = self.descriptive_stats.rename(columns={
            "mean": "Mean",
            "50%": "Median",
            "std": "Standard Deviation",
            "min": "Minimum",
            "max": "Maximum"
        })

        print(self.descriptive_stats)

    def save_results(self):
        # Save the descriptive statistics to a CSV file for documentation
        os.makedirs('results', exist_ok=True)
        self.descriptive_stats.to_csv('results/basic_metrics.csv', index=True)
        print("\nDescriptive statistics saved to 'results/basic_metrics.csv'.")

    def explain_metrics(self):
        # Explain the importance of these metrics
        print("\nImportance of Metrics:")
        print("""
        - Mean: Provides an average value, useful for understanding the central tendency of data.
        - Median: Gives the middle value, useful for datasets with skewed distributions.
        - Standard Deviation: Indicates the spread of the data, helpful in identifying variability.
        - Minimum and Maximum: Show the range of data, important for detecting outliers.
        - Count: Highlights the total number of non-null values, useful for understanding data completeness.
        """)

    def analyze(self):
        # Run the complete analysis: load data, analyze, save, and explain
        self.load_data()
        self.analyze_metrics()
        self.save_results()
        self.explain_metrics()

# Example usage:
if __name__ == "__main__":
    # Define the file path for the dataset
    file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

    # Create an instance of the BasicMetricsAnalyzer class
    analyzer = BasicMetricsAnalyzer(file_path)

    # Perform the analysis
    analyzer.analyze()