import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

class GraphicalUnivariateAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.output_dir = 'results/graphs'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_plots(self):
        # Select quantitative columns (numerical data) only
        quantitative_columns = self.df.select_dtypes(include=[float, int]).columns

        # Loop through each quantitative column and generate the most suitable plot
        for column in quantitative_columns:
            # Sanitize the column name to remove invalid characters for filenames
            sanitized_column = re.sub(r'[\\/*?:"<>|]', "_", column)
            
            # Generate and save the plot
            self._create_plot(column, sanitized_column)

    def _create_plot(self, column, sanitized_column):
        # Determine the distribution of the data
        plt.figure(figsize=(10, 6))

        if self.df[column].nunique() < 20:
            # Use a boxplot if the number of unique values is small (categorical-like variable)
            sns.boxplot(x=self.df[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            plt.savefig(f'{self.output_dir}/{sanitized_column}_boxplot.png')
            plt.close()
            print(f"Boxplot for {column} saved.")
            print(f"Insight: Boxplot for {column} indicates potential outliers and the spread of data.")
        
        else:
            # Use a histogram if the number of unique values is large (continuous variable)
            sns.histplot(self.df[column], kde=True, bins=30, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(f'{self.output_dir}/{sanitized_column}_histogram.png')
            plt.close()
            print(f"Histogram for {column} saved.")

            # Analyze the distribution of the data
            self._analyze_distribution(column)

    def _analyze_distribution(self, column):
        # Analyze the skewness and kurtosis of the data distribution
        skewness = self.df[column].skew()
        if skewness > 1:
            print(f"Insight: Histogram for {column} shows a right skew (positively skewed).")
        elif skewness < -1:
            print(f"Insight: Histogram for {column} shows a left skew (negatively skewed).")
        else:
            print(f"Insight: Histogram for {column} shows a roughly symmetric distribution.")

        # Check for normality based on KDE
        kurtosis = self.df[column].kurtosis()
        if abs(kurtosis) < 2:
            print(f"Insight: Histogram for {column} suggests a distribution close to normal.")
        else:
            print(f"Insight: Histogram for {column} shows a more peaked or flat distribution.")

    def print_interpretation(self):
        # Print out general interpretations for graphical analysis
        print("\nInterpretations of Graphical Univariate Analysis:")
        print(""" 
        - Histograms are used for continuous variables, showing the frequency distribution of the data. 
          The shape of the histogram gives insight into the data distribution (normal, skewed, bimodal, etc.).
        - Boxplots are used for continuous data as well but provide more insight into the spread of data, 
          highlighting the median, quartiles, and potential outliers.
        - KDE (Kernel Density Estimate) added to histograms helps visualize the probability density function.
        """)

    def run_analysis(self):
        self.generate_plots()
        self.print_interpretation()


# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the GraphicalUnivariateAnalysis class and run the analysis
analysis = GraphicalUnivariateAnalysis(file_path)
analysis.run_analysis()
