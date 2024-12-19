import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def user_engagement_clustering(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Aggregate metrics per customer ID (MSISDN)
    # 1. Sessions Frequency (counting the number of sessions per customer)
    sessions_frequency = df.groupby('MSISDN/Number').size().reset_index(name='sessions_frequency')

    # 2. Duration of the session (sum of 'Dur. (ms)' per customer)
    session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='total_session_duration')

    # 3. Total Traffic (download + upload bytes per customer)
    total_traffic = df.groupby('MSISDN/Number').agg({
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()

    # Add the total traffic column
    total_traffic['total_traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']

    # Merge all metrics on 'MSISDN/Number'
    engagement_data = pd.merge(sessions_frequency, session_duration, on='MSISDN/Number')
    engagement_data = pd.merge(engagement_data, total_traffic[['MSISDN/Number', 'total_traffic']], on='MSISDN/Number')

    # Normalize each engagement metric using Min-Max scaling
    scaler = MinMaxScaler()
    engagement_data_scaled = engagement_data[['sessions_frequency', 'total_session_duration', 'total_traffic']]
    engagement_data_scaled = scaler.fit_transform(engagement_data_scaled)

    # Apply K-means clustering (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    engagement_data['cluster'] = kmeans.fit_predict(engagement_data_scaled)

    # Aggregate non-normalized metrics per cluster to compute min, max, avg, and total
    cluster_summary = engagement_data.groupby('cluster').agg(
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

    # Print the cluster summary
    print("Cluster Summary (Min, Max, Avg, Total for each metric):")
    print(cluster_summary)

    # Create a folder to save the plots
    output_dir = 'results/graphs'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the non-normalized metrics
    metrics = ['sessions_frequency', 'total_session_duration', 'total_traffic']
    
    for metric in metrics:
        # Plot clusters based on each metric
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=engagement_data, x='cluster', y=metric)
        plt.title(f'{metric} by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(metric)
        
        # Save the plot as PNG
        plt.savefig(f'{output_dir}/{metric}_by_cluster.png')
        plt.close()
        print(f"Plot for {metric} saved.")

    # Print out general interpretations for graphical analysis
    print("\nInterpretations of Cluster Metrics Analysis:")
    print("""
    - Boxplots display the distribution of the engagement metrics across the three clusters.
      Each boxplot shows the median, interquartile range (IQR), and potential outliers.
    - The spread of the boxes for each metric indicates how diverse the engagement is within each cluster.
    - The larger the spread, the more variance there is in that cluster's engagement behavior.
    """)

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to perform user engagement clustering
user_engagement_clustering(file_path)