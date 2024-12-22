import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

class HandsetAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.results = {}

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")
        print(f"Loaded data shape: {self.df.shape}")  # Prints the shape of the loaded data

    def compute_distribution_throughput(self):
        if 'Handset Type' in self.df.columns and 'Avg Bearer TP DL (kbps)' in self.df.columns:
            throughput_distribution = self.df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False)
            self.results['Throughput Distribution'] = throughput_distribution
            print(f"\nTop 5 handset types with highest average throughput (kbps):")
            print(throughput_distribution.head(5))  # Print the top 5 handset types by average throughput

            # Plotting the distribution
            plt.figure(figsize=(12, 6))
            sns.barplot(x=throughput_distribution.head(10).index, y=throughput_distribution.head(10).values)
            plt.title('Top 10 Handsets by Average Throughput (kbps)')
            plt.xticks(rotation=45)
            plt.ylabel('Average Throughput (kbps)')
            plt.xlabel('Handset Type')
            plt.tight_layout()
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/throughput_distribution.png')
            print("Throughput distribution saved as 'results/throughput_distribution.png'")
        else:
            print("'Handset Type' or 'Avg Bearer TP DL (kbps)' column not found in the dataset.")

    def compute_average_tcp_retransmission(self):
        if 'Handset Type' in self.df.columns and 'TCP DL Retrans. Vol (Bytes)' in self.df.columns:
            tcp_retransmission = self.df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().sort_values(ascending=False)
            self.results['TCP Retransmission'] = tcp_retransmission
            print(f"\nTop 5 handset types with highest average TCP retransmission (Bytes):")
            print(tcp_retransmission.head(5))  # Print the top 5 handset types by average TCP retransmission

            # Plotting the retransmission
            plt.figure(figsize=(12, 6))
            sns.barplot(x=tcp_retransmission.head(10).index, y=tcp_retransmission.head(10).values)
            plt.title('Top 10 Handsets by Average TCP Retransmission (Bytes)')
            plt.xticks(rotation=45)
            plt.ylabel('Average TCP Retransmission (Bytes)')
            plt.xlabel('Handset Type')
            plt.tight_layout()
            plt.savefig('results/tcp_retransmission.png')
            print("TCP retransmission saved as 'results/tcp_retransmission.png'")
        else:
            print("'Handset Type' or 'TCP DL Retrans. Vol (Bytes)' column not found in the dataset.")

    def process(self):
        self.load_data()
        print("\nComputing Throughput Distribution...")
        self.compute_distribution_throughput()

        print("\nComputing Average TCP Retransmission...")
        self.compute_average_tcp_retransmission()

        # Saving results
        for metric, data in self.results.items():
            file_path = f'results/{metric.lower().replace(" ", "_")}.csv'
            data.to_csv(file_path, index=True, header=True)
            print(f"\n{metric} saved to '{file_path}'.")
            print(f"First 5 rows of {metric} data:")
            print(data.head(5))  # Print the first 5 rows of the saved data to give a preview

if __name__ == "__main__":
    file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')
    analysis = HandsetAnalysis(file_path)
    analysis.process()
