import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class UserEngagementClustering:
    def __init__(self, file_path, n_clusters=3):
        self.file_path = file_path
        self.n_clusters = n_clusters
        self.df = None
        self.engagement_data = None
        self.engagement_data_scaled = None
        self.cluster_model = None
        self.cluster_summary = None

    def load_data(self):
        """Loads the dataset from the provided file path."""
        self.df = pd.read_csv(self.file_path)

    def preprocess_data(self):
        """Aggregates the metrics and prepares the data for clustering."""
        # Aggregate metrics per customer ID (MSISDN)
        sessions_frequency = self.df.groupby('MSISDN/Number').size().reset_index(name='sessions_frequency')
        session_duration = self.df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='total_session_duration')
        total_traffic = self.df.groupby('MSISDN/Number').agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()

        # Add the total traffic column
        total_traffic['total_traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']

        # Merge all metrics on 'MSISDN/Number'
        self.engagement_data = pd.merge(sessions_frequency, session_duration, on='MSISDN/Number')
        self.engagement_data = pd.merge(self.engagement_data, total_traffic[['MSISDN/Number', 'total_traffic']], on='MSISDN/Number')

    def normalize_data(self):
        """Normalizes the engagement metrics using Min-Max scaling."""
        scaler = MinMaxScaler()
        engagement_data_metrics = self.engagement_data[['sessions_frequency', 'total_session_duration', 'total_traffic']]
        self.engagement_data_scaled = scaler.fit_transform(engagement_data_metrics)

    def apply_kmeans(self):
        """Applies K-means clustering on the normalized data."""
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.engagement_data['cluster'] = self.cluster_model.fit_predict(self.engagement_data_scaled)

    def summarize_clusters(self):
        """Aggregates non-normalized metrics per cluster to compute min, max, avg, and total."""
        self.cluster_summary = self.engagement_data.groupby('cluster').agg(
            min_sessions_frequency=('sessions_frequency', 'min'),
            max_sessions_frequency=('sessions_frequency', 'max'),
            avg_sessions_frequency=('sessions_frequency', 'mean'),
            total_sessions_frequency=('sessions_frequency', 'sum'),
            
            min_total_session_duration=('total_session_duration', 'min'),
            max_total_session_duration=('total_session_duration', 'max'),
            avg_total_session_duration=('total_session_duration', 'mean'),
            total_total_session_duration=('total_session_duration', 'sum'),
            
            min_total_traffic=('total_traffic', 'min'),
            max_total_traffic=('total_traffic', 'max'),
            avg_total_traffic=('total_traffic', 'mean'),
            total_total_traffic=('total_traffic', 'sum')
        ).reset_index()

    def visualize_clusters(self):
        """Visualizes the non-normalized metrics for each cluster."""
        output_dir = 'results/graphs'
        os.makedirs(output_dir, exist_ok=True)

        metrics = ['sessions_frequency', 'total_session_duration', 'total_traffic']
        
        for metric in metrics:
            # Plot clusters based on each metric
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.engagement_data, x='cluster', y=metric)
            plt.title(f'{metric} by Cluster')
            plt.xlabel('Cluster')
            plt.ylabel(metric)
            
            # Save the plot as PNG
            plt.savefig(f'{output_dir}/{metric}_by_cluster.png')
            plt.close()
            print(f"Plot for {metric} saved.")

    def print_cluster_summary(self):
        """Prints the cluster summary."""
        pd.set_option('display.max_columns', None)
        print("Cluster Summary (Min, Max, Avg, Total for each metric):")
        print(self.cluster_summary)

    def run_clustering(self):
        """Runs the full clustering process."""
        self.load_data()
        self.preprocess_data()
        self.normalize_data()
        self.apply_kmeans()
        self.summarize_clusters()
        self.print_cluster_summary()
        self.visualize_clusters()

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Instantiate the UserEngagementClustering class
clustering = UserEngagementClustering(file_path)

# Run the clustering process
clustering.run_clustering()