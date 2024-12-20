import pandas as pd
import os

class NonGraphicalUnivariateAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.dispersion_stats = pd.DataFrame()

    def calculate_dispersion_stats(self):
        # Select quantitative columns (numerical data) only
        quantitative_columns = self.df.select_dtypes(include=[float, int]).columns
        
        # Calculate variance, standard deviation, and IQR for each quantitative variable
        self.dispersion_stats['Variance'] = self.df[quantitative_columns].var()
        self.dispersion_stats['Standard Deviation'] = self.df[quantitative_columns].std()
        self.dispersion_stats['Interquartile Range (IQR)'] = self.df[quantitative_columns].apply(lambda x: x.quantile(0.75) - x.quantile(0.25))

    def print_dispersion_stats(self):
        # Print the dispersion statistics for each variable
        print("\nDispersion Statistics for Each Quantitative Variable:")
        print(self.dispersion_stats)

    def save_dispersion_stats(self):
        # Save the dispersion statistics to a CSV file for documentation
        os.makedirs('results', exist_ok=True)
        self.dispersion_stats.to_csv('results/dispersion_stats.csv', index=True)
        print("\nDispersion statistics saved to 'results/dispersion_stats.csv'.")

    def print_interpretation(self):
        # Explanation of the dispersion metrics
        print("\nInterpretation of Dispersion Metrics:")
        print(""" 
        - Variance: Measures how spread out the data is. Higher variance indicates greater spread.
        - Standard Deviation: The square root of variance, gives a more interpretable measure of spread in the same units as the data.
        - Interquartile Range (IQR): Measures the spread of the middle 50% of the data. Useful for detecting outliers.
        """)

    def run_analysis(self):
        self.calculate_dispersion_stats()
        self.print_dispersion_stats()
        self.save_dispersion_stats()
        self.print_interpretation()

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the NonGraphicalUnivariateAnalysis class and run the analysis
analysis = NonGraphicalUnivariateAnalysis(file_path)
analysis.run_analysis()
