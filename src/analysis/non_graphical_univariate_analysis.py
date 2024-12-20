import pandas as pd
import os

def non_graphical_univariate_analysis(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Select quantitative columns (numerical data) only
    quantitative_columns = df.select_dtypes(include=[float, int]).columns

    # Create a DataFrame to store dispersion statistics
    dispersion_stats = pd.DataFrame(index=quantitative_columns)

    # Calculate variance, standard deviation, and IQR for each quantitative variable
    dispersion_stats['Variance'] = df[quantitative_columns].var()
    dispersion_stats['Standard Deviation'] = df[quantitative_columns].std()
    dispersion_stats['Interquartile Range (IQR)'] = df[quantitative_columns].apply(lambda x: x.quantile(0.75) - x.quantile(0.25))

    # Print the dispersion statistics for each variable
    print("\nDispersion Statistics for Each Quantitative Variable:")
    print(dispersion_stats)

    # Save the dispersion statistics to a CSV file for documentation
    os.makedirs('results', exist_ok=True)
    dispersion_stats.to_csv('results/dispersion_stats.csv', index=True)
    print("\nDispersion statistics saved to 'results/dispersion_stats.csv'.")

    # Explanation of the dispersion metrics
    print("\nInterpretation of Dispersion Metrics:")
    print("""
    - Variance: Measures how spread out the data is. Higher variance indicates greater spread.
    - Standard Deviation: The square root of variance, gives a more interpretable measure of spread in the same units as the data.
    - Interquartile Range (IQR): Measures the spread of the middle 50% of the data. Useful for detecting outliers.
    """)

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to conduct non-graphical univariate analysis
non_graphical_univariate_analysis(file_path)
