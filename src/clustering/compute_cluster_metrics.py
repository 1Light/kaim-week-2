import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

class ClusterAnalysis:
    def __init__(self, file_path, num_clusters=5):
        # Initializes with the file path, number of clusters, and loads the CSV into a DataFrame
        self.file_path = file_path
        self.num_clusters = num_clusters
        self.df = pd.read_csv(file_path)
        self.clustered_data = pd.DataFrame()

    def preprocess_data(self):
        # Select numeric columns for clustering (ignoring object columns)
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns

        # Handle NaN values in the numeric columns (e.g., fill NaNs with the column mean)
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())

    def apply_kmeans_clustering(self):
        # Apply KMeans clustering and assign cluster labels to the DataFrame
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.df['ClusterId'] = kmeans.fit_predict(self.df[numeric_columns])

    def compute_cluster_metrics(self):
        # Compute the minimum, maximum, average, and total for each cluster
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        cluster_metrics = self.df.groupby('ClusterId')[numeric_columns].agg(
            ['min', 'max', 'mean', 'sum']).reset_index()

        return cluster_metrics

    def save_cluster_metrics(self, cluster_metrics):
        # Save the result to a new CSV file
        output_dir = 'results/cluster_metrics'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'cluster_metrics.csv')
        cluster_metrics.to_csv(output_file, index=False)
        print(f"Cluster metrics saved to {output_file}")

    def visualize_clusters(self):
        # Visualize the metrics (example: distribution of Total DL per cluster)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='ClusterId', y='Total DL (Bytes)', data=self.df)
        plt.title('Total Download Bytes per Cluster')
        plt.savefig('results/cluster_metrics/total_dl_per_cluster.png')
        plt.close()
        print("Visualization saved: total_dl_per_cluster.png")

    def run_analysis(self):
        # Runs the entire clustering and visualization process
        self.preprocess_data()
        self.apply_kmeans_clustering()
        cluster_metrics = self.compute_cluster_metrics()
        self.save_cluster_metrics(cluster_metrics)
        self.visualize_clusters()

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the ClusterAnalysis class and run the analysis
analysis = ClusterAnalysis(file_path)
analysis.run_analysis()