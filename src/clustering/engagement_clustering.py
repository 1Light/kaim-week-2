import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class EngagementClusterer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.engagement_columns = [
            'Social Media DL (Bytes)',
            'Google DL (Bytes)',
            'Email DL (Bytes)',
            'Youtube DL (Bytes)',
            'Netflix DL (Bytes)',
            'Gaming DL (Bytes)'
        ]
        self.engagement_data_scaled = None
        self.optimal_k = None
        self.kmeans = None

    def load_data(self):
        """Load the CSV file into a pandas DataFrame."""
        self.df = pd.read_csv(self.file_path)

    def preprocess_data(self):
        """Preprocess the data by handling missing values and scaling the data."""
        # Drop rows with NaN values in the engagement columns
        self.df = self.df.dropna(subset=self.engagement_columns)
        
        # Extract the relevant engagement data for clustering
        engagement_data = self.df[self.engagement_columns]
        
        # Normalize the data using StandardScaler
        scaler = StandardScaler()
        self.engagement_data_scaled = scaler.fit_transform(engagement_data)
    
    def compute_optimal_k(self):
        """Use the elbow method to determine the optimal number of clusters (k)."""
        wcss = []
        for k in range(1, 11):  # Trying k values from 1 to 10
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans.fit(self.engagement_data_scaled)
            wcss.append(kmeans.inertia_)
        
        # Automatically detect the elbow point
        diffs = np.diff(wcss)
        second_diffs = np.diff(diffs)
        self.optimal_k = np.argmin(second_diffs) + 2  # Add 2 to adjust for the 1-based index of the elbow
        print(f"Optimized value of k based on the elbow method: {self.optimal_k}")
    
    def perform_clustering(self):
        """Apply K-Means clustering to the data."""
        self.kmeans = KMeans(n_clusters=self.optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        self.df['ClusterId'] = self.kmeans.fit_predict(self.engagement_data_scaled)

    def visualize_clusters(self):
        """Visualize the clustering results using a scatter plot."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df['Social Media DL (Bytes)'], y=self.df['Google DL (Bytes)'], hue=self.df['ClusterId'], palette='Set1')
        plt.title(f'User Engagement Clusters (k={self.optimal_k})')
        plt.xlabel('Social Media DL (Bytes)')
        plt.ylabel('Google DL (Bytes)')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig(f'user_engagement_clusters_k_{self.optimal_k}.png')
        plt.show()

    def display_cluster_centers(self):
        """Display the cluster centers in the original data scale."""
        scaler = StandardScaler()
        cluster_centers = pd.DataFrame(scaler.inverse_transform(self.kmeans.cluster_centers_), columns=self.engagement_columns)
        print(f"Cluster Centers (Original Data Scale):\n{cluster_centers}")

    def interpret_clusters(self):
        """Interpret the clustering results."""
        print("\nInterpretation of the findings:")
        for i in range(self.optimal_k):
            print(f"\nCluster {i} characteristics:")
            cluster_data = self.df[self.df['ClusterId'] == i]
            for col in self.engagement_columns:
                mean_value = cluster_data[col].mean()
                print(f"  Average {col}: {mean_value:.2f} bytes")

    def run(self):
        """Run the complete clustering process."""
        self.load_data()
        self.preprocess_data()
        self.compute_optimal_k()
        self.perform_clustering()
        self.visualize_clusters()
        self.display_cluster_centers()
        self.interpret_clusters()

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of EngagementClusterer and run the process
clusterer = EngagementClusterer(file_path)
clusterer.run()