import pandas as pd
import numpy as np
import os

class CustomerMetricsAggregator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.aggregated_data = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")

    def handle_missing_values(self):
        for column in self.df.columns:
            if self.df[column].dtype in ['float64', 'int64']:
                self.df[column] = self.df[column].fillna(self.df[column].mean())
            elif self.df[column].dtype == 'object':
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        print("\nMissing values handled.")

    def handle_outliers(self):
        for column in self.df.select_dtypes(include=['float64', 'int64']):
            mean = self.df[column].mean()
            std_dev = self.df[column].std()
            lower_bound = mean - 3 * std_dev
            upper_bound = mean + 3 * std_dev
            self.df[column] = np.where(
                (self.df[column] < lower_bound) | (self.df[column] > upper_bound),
                mean,
                self.df[column]
            )
        print("\nOutliers handled.")

    def aggregate_metrics(self):
        # Replace 'MSISDN/Number' or 'IMEI' with the appropriate column name
        group_by_column = 'MSISDN/Number'  # Use this as a proxy for Customer_ID
        self.aggregated_data = self.df.groupby(group_by_column).agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',  # Average TCP retransmission
            'Avg RTT DL (ms)': 'mean',             # Average RTT
            'Handset Type': lambda x: x.mode()[0], # Most common Handset Type
            'Avg Bearer TP DL (kbps)': 'mean'      # Average Throughput
        }).reset_index()
        print("\nMetrics aggregated per customer.")

    def save_results(self):
        os.makedirs('results', exist_ok=True)
        output_file = 'results/aggregated_customer_metrics.csv'
        self.aggregated_data.to_csv(output_file, index=False)
        print(f"\nAggregated data saved to '{output_file}'.")

    def process(self):
        self.load_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.aggregate_metrics()
        self.save_results()

if __name__ == "__main__":
    file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')
    aggregator = CustomerMetricsAggregator(file_path)
    aggregator.process()