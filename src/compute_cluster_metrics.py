import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def compute_cluster_metrics(file_path, num_clusters=5):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Select numeric columns for clustering (ignoring object columns)
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Handle NaN values in the numeric columns (e.g., fill NaNs with the column mean)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['ClusterId'] = kmeans.fit_predict(df[numeric_columns])

    # Compute the minimum, maximum, average, and total for each cluster
    cluster_metrics = df.groupby('ClusterId')[numeric_columns].agg(
        ['min', 'max', 'mean', 'sum']).reset_index()
    
    # Save the result to a new CSV file
    output_dir = 'results/cluster_metrics'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'cluster_metrics.csv')
    cluster_metrics.to_csv(output_file, index=False)
    print(f"Cluster metrics saved to {output_file}")
    
    # Visualize the metrics (example: distribution of Total DL per cluster)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ClusterId', y='Total DL (Bytes)', data=df)
    plt.title('Total Download Bytes per Cluster')
    plt.savefig(f'{output_dir}/total_dl_per_cluster.png')
    plt.close()
    
    # Additional visualizations can be added similarly for other metrics
    print("Visualization saved: total_dl_per_cluster.png")

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to compute cluster metrics
compute_cluster_metrics(file_path)