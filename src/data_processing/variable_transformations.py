import pandas as pd
import os
import numpy as np

class DataTransformer:
    def __init__(self, file_path):
        # Initialize the object with the file path
        self.file_path = file_path
        self.df = None
    
    def load_data(self):
        """Load the CSV file into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
    
    def check_columns(self):
        """Ensure the required columns exist in the dataset"""
        required_columns = ['Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Error: The following columns are missing: {', '.join(missing_columns)}")
            return False
        return True

    def compute_total_data_volume(self):
        """Compute the total data volume for each user"""
        self.df['total_data_volume'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']

    def aggregate_user_data(self):
        """Aggregate data per user (IMSI) to compute total duration and total data volume"""
        user_aggregates = self.df.groupby('IMSI').agg(
            total_duration=('Dur. (ms)', 'sum'),
            total_data_volume=('total_data_volume', 'sum')
        ).reset_index()
        return user_aggregates

    def segment_users_by_decile(self, user_aggregates):
        """Segment users into deciles based on total duration"""
        user_aggregates['duration_decile'] = pd.qcut(user_aggregates['total_duration'], 10, labels=False)
        return user_aggregates

    def compute_data_volume_per_decile(self, user_aggregates):
        """Compute total data volume per decile class"""
        decile_data = user_aggregates.groupby('duration_decile').agg(
            total_data_volume=('total_data_volume', 'sum')
        ).reset_index()
        return decile_data

    def save_results(self, decile_data):
        """Save the decile data to a CSV file"""
        os.makedirs('results', exist_ok=True)
        decile_data.to_csv('results/decile_data.csv', index=False)
        print("Decile data saved to 'results/decile_data.csv'.")

    def run_transformations(self):
        """Run all transformations"""
        self.load_data()
        if self.df is not None and self.check_columns():
            self.compute_total_data_volume()
            user_aggregates = self.aggregate_user_data()
            user_aggregates = self.segment_users_by_decile(user_aggregates)
            decile_data = self.compute_data_volume_per_decile(user_aggregates)
            
            # Print and save the results
            print("\nTotal Data Volume per Decile Class:")
            print(decile_data)
            self.save_results(decile_data)


# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the DataTransformer class
data_transformer = DataTransformer(file_path)

# Run the data transformations
data_transformer.run_transformations()
