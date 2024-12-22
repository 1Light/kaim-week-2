import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ExperienceClustering:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cluster_centers = None
        self.cluster_labels = None
        # Adjusted metrics
        self.metrics = [
            'Avg Bearer TP DL (kbps)',
            'TCP DL Retrans. Vol (Bytes)',
            'Avg RTT Combined (ms)'  # New metric for combined RTT
        ]

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")

    def preprocess_data(self):
        # Calculate combined RTT
        self.df['Avg RTT Combined (ms)'] = (self.df['Avg RTT DL (ms)'] + self.df['Avg RTT UL (ms)']) / 2

        # Select relevant metrics
        self.df = self.df[self.metrics].dropna()

        # Normalize the data
        scaler = StandardScaler()
        self.df_scaled = scaler.fit_transform(self.df)
        print("Data normalized for clustering.")

    def perform_clustering(self, k=3):
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.df_scaled)
        self.cluster_centers = kmeans.cluster_centers_
        self.df['Cluster'] = self.cluster_labels
        print(f"K-means clustering performed with k={k}.")

    def analyze_clusters(self):
        # Calculate cluster metrics
        cluster_analysis = self.df.groupby('Cluster').mean()
        print("\nCluster Analysis:")
        print(cluster_analysis)

        # Save cluster analysis results
        os.makedirs('results', exist_ok=True)
        cluster_analysis.to_csv('results/cluster_analysis.csv')
        print("Cluster analysis saved as 'results/cluster_analysis.csv'.")

        # Visualize cluster centers
        plt.figure(figsize=(12, 6))
        sns.heatmap(cluster_analysis, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Cluster Characteristics')
        plt.tight_layout()
        plt.savefig('results/cluster_characteristics.png')
        print("Cluster characteristics heatmap saved as 'results/cluster_characteristics.png'.")

    def process(self):
        self.load_data()
        self.preprocess_data()
        self.perform_clustering(k=3)
        self.analyze_clusters()

if __name__ == "__main__":
    file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')
    clustering = ExperienceClustering(file_path)
    clustering.process()
