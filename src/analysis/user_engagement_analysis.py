import pandas as pd
import os

class UserEngagementAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.engagement_data = pd.DataFrame()

    def aggregate_metrics(self):
        # Aggregate metrics per customer ID (MSISDN)
        
        # 1. Sessions Frequency (counting the number of sessions per customer)
        sessions_frequency = self.df.groupby('MSISDN/Number').size().reset_index(name='sessions_frequency')

        # 2. Duration of the session (sum of 'Dur. (ms)' per customer)
        session_duration = self.df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='total_session_duration')

        # 3. Total Traffic (download + upload bytes per customer)
        total_traffic = self.df.groupby('MSISDN/Number').agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()

        # Add the total traffic column
        total_traffic['total_traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']

        # Merge all metrics on 'MSISDN/Number'
        self.engagement_data = pd.merge(sessions_frequency, session_duration, on='MSISDN/Number')
        self.engagement_data = pd.merge(self.engagement_data, total_traffic[['MSISDN/Number', 'total_traffic']], on='MSISDN/Number')

    def print_top_customers(self):
        # Report top 10 customers for each engagement metric
        
        # Top 10 by Sessions Frequency
        top_10_sessions = self.engagement_data.nlargest(10, 'sessions_frequency')
        print("Top 10 Customers by Sessions Frequency:")
        print(top_10_sessions[['MSISDN/Number', 'sessions_frequency']])

        # Top 10 by Total Session Duration
        top_10_duration = self.engagement_data.nlargest(10, 'total_session_duration')
        print("\nTop 10 Customers by Total Session Duration:")
        print(top_10_duration[['MSISDN/Number', 'total_session_duration']])

        # Top 10 by Total Traffic
        top_10_traffic = self.engagement_data.nlargest(10, 'total_traffic')
        print("\nTop 10 Customers by Total Traffic:")
        print(top_10_traffic[['MSISDN/Number', 'total_traffic']])

    def run_analysis(self):
        self.aggregate_metrics()
        self.print_top_customers()

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the UserEngagementAnalysis class and run the analysis
analysis = UserEngagementAnalysis(file_path)
analysis.run_analysis()