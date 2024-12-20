import pandas as pd
import os

class TrafficAggregator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.app_columns = [
            'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
            'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 
            'Other DL (Bytes)', 'Total DL (Bytes)', 
            'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)', 
            'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 
            'Other UL (Bytes)', 'Total UL (Bytes)'
        ]
        self.app_traffic = pd.DataFrame()
        self.top_10_users = {}

    def aggregate_traffic(self):
        """Aggregate traffic per user and per application."""
        for app_column in self.app_columns:
            app_name = app_column.split()[0]  # Extract the app name (e.g., 'Social', 'Google', etc.)
            self.app_traffic[app_name] = self.df.groupby('MSISDN/Number')[app_column].sum()

        # Ensure all columns of the app_traffic DataFrame are printed
        pd.set_option('display.max_columns', None)
        print("Aggregated Traffic Per Application (Bytes):")
        print(self.app_traffic)

    def find_top_10_users(self):
        """Find top 10 most engaged users for each application."""
        for app_name in self.app_traffic.columns:
            top_users = self.app_traffic[app_name].sort_values(ascending=False).head(10)
            self.top_10_users[app_name] = top_users

    def save_results(self):
        """Create a folder and save results for each application."""
        output_dir = 'results/aggregated_traffic'
        os.makedirs(output_dir, exist_ok=True)
        
        for app_name, top_users in self.top_10_users.items():
            top_users.to_csv(f'{output_dir}/{app_name}_top_10_users.csv', header=True)
        
        print("Top 10 most engaged users for each application have been saved.")

    def run(self):
        """Execute the entire traffic aggregation process."""
        self.aggregate_traffic()
        self.find_top_10_users()
        self.save_results()

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the TrafficAggregator class and run the aggregation process
traffic_aggregator = TrafficAggregator(file_path)
traffic_aggregator.run()
