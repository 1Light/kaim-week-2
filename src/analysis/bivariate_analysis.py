import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def bivariate_analysis(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define the total DL and UL data columns
    total_dl_ul_column = 'Total DL (Bytes)'  # Use the correct column name for total data

    # Define the application-specific columns you want to analyze
    application_columns = [
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
        'Google DL (Bytes)', 'Google UL (Bytes)', 
        'Email DL (Bytes)', 'Email UL (Bytes)', 
        'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
        'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
        'Other DL (Bytes)', 'Other UL (Bytes)'
    ]
    
    # Loop through each application column and plot the relationship with Total DL+UL
    for column in application_columns:
        plt.figure(figsize=(10, 6))
        
        # Create a scatter plot to explore the relationship
        sns.scatterplot(x=df[column], y=df[total_dl_ul_column], color='blue')
        plt.title(f'Relationship between {column} and Total DL+UL Data')
        plt.xlabel(f'{column}')
        plt.ylabel(f'{total_dl_ul_column}')
        
        # Save the plot
        os.makedirs('results/graphs', exist_ok=True)
        plt.savefig(f'results/graphs/{column}_vs_{total_dl_ul_column}_scatterplot.png')
        plt.close()
        print(f"Scatter plot for {column} vs {total_dl_ul_column} saved.")

        # Calculate the correlation coefficient between the two variables
        correlation, p_value = pearsonr(df[column].dropna(), df[total_dl_ul_column].dropna())
        
        # Print correlation insights
        if correlation > 0.7:
            print(f"Insight: There is a strong positive correlation ({correlation:.2f}) between {column} and {total_dl_ul_column}.")
        elif correlation < -0.7:
            print(f"Insight: There is a strong negative correlation ({correlation:.2f}) between {column} and {total_dl_ul_column}.")
        elif 0.3 < correlation <= 0.7:
            print(f"Insight: There is a moderate positive correlation ({correlation:.2f}) between {column} and {total_dl_ul_column}.")
        elif -0.7 < correlation < -0.3:
            print(f"Insight: There is a moderate negative correlation ({correlation:.2f}) between {column} and {total_dl_ul_column}.")
        else:
            print(f"Insight: There is a weak or no correlation ({correlation:.2f}) between {column} and {total_dl_ul_column}.")

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

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to conduct bivariate analysis
bivariate_analysis(file_path)