import pandas as pd
import os

class ComputeMetrics:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = {}

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")
        print(f"Loaded data shape: {self.df.shape}")  # Prints the shape of the loaded data

    def compute_top_bottom_frequent(self, column_name):
        top_10 = self.df[column_name].nlargest(10).tolist()
        bottom_10 = self.df[column_name].nsmallest(10).tolist()
        most_frequent = self.df[column_name].value_counts().head(10).index.tolist()
        return {
            'Top 10': top_10,
            'Bottom 10': bottom_10,
            'Most Frequent': most_frequent
        }

    def process_metrics(self):
        columns_to_compute = {
            'TCP DL Retrans. Vol (Bytes)': 'TCP Retransmission',
            'Avg RTT DL (ms)': 'RTT',
            'Avg Bearer TP DL (kbps)': 'Throughput'
        }
        
        for column, label in columns_to_compute.items():
            if column in self.df.columns:
                print(f"\nProcessing metrics for {label}...")
                self.results[label] = self.compute_top_bottom_frequent(column)
                print(f"Top 10 values for {label}: {self.results[label]['Top 10']}")
                print(f"Bottom 10 values for {label}: {self.results[label]['Bottom 10']}")
                print(f"Most frequent values for {label}: {self.results[label]['Most Frequent']}")
            else:
                print(f"\nColumn '{column}' not found in the dataset.")

    def save_results(self):
        os.makedirs('results', exist_ok=True)
        output_file = 'results/tcp_rtt_throughput_metrics.csv'

        # Flatten the results for saving
        rows = []
        for metric, data in self.results.items():
            for key, values in data.items():
                for value in values:
                    rows.append({'Metric': metric, 'Category': key, 'Value': value})

        result_df = pd.DataFrame(rows)
        result_df.to_csv(output_file, index=False)
        print(f"\nMetrics saved to '{output_file}'.")

    def process(self):
        self.load_data()
        self.process_metrics()
        self.save_results()

if __name__ == "__main__":
    file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')
    metrics_computer = ComputeMetrics(file_path)
    metrics_computer.process()