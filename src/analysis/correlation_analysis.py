import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.columns_to_analyze = [
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]
        self.data = self.df[self.columns_to_analyze]
    
    def compute_correlation(self):
        # Compute the correlation matrix
        self.correlation_matrix = self.data.corr()

    def visualize_correlation(self):
        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title('Correlation Matrix for Application Data')
        os.makedirs('results/graphs', exist_ok=True)
        plt.savefig('results/graphs/correlation_matrix.png')
        plt.show()

    def interpret_correlation(self):
        # Interpret the findings
        print("\nInterpretation of Correlation Matrix:")
        
        # Identify high positive correlations
        high_positive_corr = self.correlation_matrix[self.correlation_matrix > 0.7]
        for col in high_positive_corr.columns:
            for row in high_positive_corr.index:
                if high_positive_corr.at[row, col] > 0.7 and row != col:
                    print(f"Insight: There is a high positive correlation ({high_positive_corr.at[row, col]:.2f}) between {row} and {col}.")
        
        # Identify high negative correlations
        high_negative_corr = self.correlation_matrix[self.correlation_matrix < -0.7]
        for col in high_negative_corr.columns:
            for row in high_negative_corr.index:
                if high_negative_corr.at[row, col] < -0.7 and row != col:
                    print(f"Insight: There is a high negative correlation ({high_negative_corr.at[row, col]:.2f}) between {row} and {col}.")

        # Identify low correlations
        low_corr = self.correlation_matrix[(self.correlation_matrix > -0.3) & (self.correlation_matrix < 0.3)]
        for col in low_corr.columns:
            for row in low_corr.index:
                if abs(low_corr.at[row, col]) < 0.3 and row != col:
                    print(f"Insight: There is a low correlation ({low_corr.at[row, col]:.2f}) between {row} and {col}.")
        
        print("""\nGeneral Interpretation of Correlation:
        - Correlation values range from -1 to 1.
        - A correlation closer to +1 indicates a strong positive relationship, meaning as one variable increases, the other tends to increase.
        - A correlation closer to -1 indicates a strong negative relationship, meaning as one variable increases, the other tends to decrease.
        - A correlation near 0 indicates little or no linear relationship between the variables.
        - High correlations can help identify related or similar application usage patterns, while low correlations might indicate unrelated usage patterns.
        """)

    def run_analysis(self):
        self.compute_correlation()
        self.visualize_correlation()
        self.interpret_correlation()


# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the CorrelationAnalysis class and run the analysis
analysis = CorrelationAnalysis(file_path)
analysis.run_analysis()