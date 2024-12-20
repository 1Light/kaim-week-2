import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class BivariateAnalysis:
    def __init__(self, file_path):
        # Initialize with the file path and load the data
        self.file_path = file_path
        self.df = None
        self.total_dl_ul_column = 'Total DL (Bytes)'  # The column name for total data
        self.application_columns = [
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
            'Google DL (Bytes)', 'Google UL (Bytes)', 
            'Email DL (Bytes)', 'Email UL (Bytes)', 
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]

    def load_data(self):
        # Load the CSV file into a pandas DataFrame
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")

    def plot_scatter(self, column):
        # Create a scatter plot to explore the relationship
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[column], y=self.df[self.total_dl_ul_column], color='blue')
        plt.title(f'Relationship between {column} and Total DL+UL Data')
        plt.xlabel(f'{column}')
        plt.ylabel(f'{self.total_dl_ul_column}')
        
        # Save the plot
        os.makedirs('results/graphs', exist_ok=True)
        plt.savefig(f'results/graphs/{column}_vs_{self.total_dl_ul_column}_scatterplot.png')
        plt.close()
        print(f"Scatter plot for {column} vs {self.total_dl_ul_column} saved.")

    def calculate_correlation(self, column):
        # Calculate the correlation coefficient between the two variables
        correlation, p_value = pearsonr(self.df[column].dropna(), self.df[self.total_dl_ul_column].dropna())
        return correlation

    def print_correlation_insight(self, column, correlation):
        # Print correlation insights
        if correlation > 0.7:
            print(f"Insight: There is a strong positive correlation ({correlation:.2f}) between {column} and {self.total_dl_ul_column}.")
        elif correlation < -0.7:
            print(f"Insight: There is a strong negative correlation ({correlation:.2f}) between {column} and {self.total_dl_ul_column}.")
        elif 0.3 < correlation <= 0.7:
            print(f"Insight: There is a moderate positive correlation ({correlation:.2f}) between {column} and {self.total_dl_ul_column}.")
        elif -0.7 < correlation < -0.3:
            print(f"Insight: There is a moderate negative correlation ({correlation:.2f}) between {column} and {self.total_dl_ul_column}.")
        else:
            print(f"Insight: There is a weak or no correlation ({correlation:.2f}) between {column} and {self.total_dl_ul_column}.")

    def explain_analysis(self):
        # Print out general interpretations for bivariate analysis
        print("\nInterpretations of Bivariate Analysis:")
        print("""
        - Scatter plots are used to explore the relationship between two variables.
        - If the points are clustered along a line, it indicates a strong relationship (positive or negative).
        - If the points are spread randomly, it suggests no strong relationship between the variables.
        - A high positive correlation (close to +1) suggests both variables increase together.
        - A high negative correlation (close to -1) suggests that as one variable increases, the other decreases.
        - A correlation near 0 suggests no linear relationship between the variables.
        """)

    def analyze(self):
        # Run the complete bivariate analysis
        self.load_data()

        # Loop through each application column and perform analysis
        for column in self.application_columns:
            self.plot_scatter(column)
            correlation = self.calculate_correlation(column)
            self.print_correlation_insight(column, correlation)

        # Provide general interpretation of the analysis
        self.explain_analysis()

# Example usage:
if __name__ == "__main__":
    # Define the file path for the dataset
    file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

    # Create an instance of the BivariateAnalysis class
    analysis = BivariateAnalysis(file_path)

    # Perform the bivariate analysis
    analysis.analyze()
