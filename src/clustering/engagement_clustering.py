import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_optimal_k(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define the engagement columns
    engagement_columns = [
        'Social Media DL (Bytes)',
        'Google DL (Bytes)',
        'Email DL (Bytes)',
        'Youtube DL (Bytes)',
        'Netflix DL (Bytes)',
        'Gaming DL (Bytes)'
    ]
    
    # Preprocess the data: handle missing values by dropping rows with NaN values
    df = df.dropna(subset=engagement_columns)

    # Extract the engagement metrics data for clustering
    engagement_data = df[engagement_columns]

    # Normalize the data (important for clustering)
    scaler = StandardScaler()
    engagement_data_scaled = scaler.fit_transform(engagement_data)

    # Use the elbow method to determine the optimal k
    wcss = []
    for k in range(1, 11):  # Trying k values from 1 to 10
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(engagement_data_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.show()

    # Automatically detect the elbow point (k where the decrease in WCSS starts to slow down)
    k_values = range(1, 11)
    diffs = np.diff(wcss)
    second_diffs = np.diff(diffs)
    
    # Find the index of the smallest second difference (this often corresponds to the "elbow")
    optimal_k = np.argmin(second_diffs) + 2  # Add 2 to adjust for the 1-based index of the elbow

    print(f"Optimized value of k based on the elbow method: {optimal_k}")
    return engagement_data_scaled, optimal_k

def kmeans_clustering(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define the engagement columns
    engagement_columns = [
        'Social Media DL (Bytes)',
        'Google DL (Bytes)',
        'Email DL (Bytes)',
        'Youtube DL (Bytes)',
        'Netflix DL (Bytes)',
        'Gaming DL (Bytes)'
    ]
    
    # Preprocess the data: handle missing values by dropping rows with NaN values
    df = df.dropna(subset=engagement_columns)

    # Extract the engagement metrics data for clustering
    engagement_data = df[engagement_columns]

    # Normalize the data (important for clustering)
    scaler = StandardScaler()
    engagement_data_scaled = scaler.fit_transform(engagement_data)

    # Determine the optimal k using the elbow method
    engagement_data_scaled, optimal_k = compute_optimal_k(file_path)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    df['ClusterId'] = kmeans.fit_predict(engagement_data_scaled)

    # Visualize the clusters (example: using a scatter plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['Social Media DL (Bytes)'], y=df['Google DL (Bytes)'], hue=df['ClusterId'], palette='Set1')
    plt.title(f'User Engagement Clusters (k={optimal_k})')
    plt.xlabel('Social Media DL (Bytes)')
    plt.ylabel('Google DL (Bytes)')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(f'user_engagement_clusters_k_{optimal_k}.png')
    plt.show()

    # Display the cluster centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=engagement_columns)
    print(f"Cluster Centers (Original Data Scale):\n{cluster_centers}")

    # Add interpretation of the clusters
    print("\nInterpretation of the findings:")
    for i in range(optimal_k):
        print(f"\nCluster {i} characteristics:")
        cluster_data = df[df['ClusterId'] == i]
        for col in engagement_columns:
            mean_value = cluster_data[col].mean()
            print(f"  Average {col}: {mean_value:.2f} bytes")

    # Return the dataframe with cluster IDs
    return df

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to perform clustering and plot results
df_with_clusters = kmeans_clustering(file_path)
