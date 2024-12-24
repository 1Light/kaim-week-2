import pandas as pd
import os
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import gdown
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

def load_and_clean_data():

    file_ids = {
        "cleaned_data": "1MFUkAlxFDdXUqXDO1uBu9II9KCqiQiL_",  
    }

    # Define local paths to save the data temporarily
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temp_data'))
    os.makedirs(base_path, exist_ok=True)  
    
    # Define file paths for each dataset
    file_path = os.path.join(base_path, 'main_data_source.csv')

    # Download files only if they don't exist
    if not os.path.exists(file_path):
        print("Downloading data...")
        download_from_gdrive(file_ids["cleaned_data"], file_path)

###############################################################################################
################################## User Engagement Analysis ###################################
###############################################################################################

def load_data(file_path):
    """Load the data from the provided file path."""
    return pd.read_csv(file_path)

def aggregate_metrics(df):
    """Aggregate user engagement metrics."""
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

    return engagement_data

def top_customers_by_metric(engagement_data, metric, top_n=10):
    """Get top N customers based on a specified metric."""
    return engagement_data.nlargest(top_n, metric)[['MSISDN/Number', metric]]

def user_engagement_analysis(file_path):
    """Main function to run user engagement analysis."""
    df = load_data(file_path)
    engagement_data = aggregate_metrics(df)

    top_sessions = top_customers_by_metric(engagement_data, 'sessions_frequency')
    top_duration = top_customers_by_metric(engagement_data, 'total_session_duration')
    top_traffic = top_customers_by_metric(engagement_data, 'total_traffic')

    return top_sessions, top_duration, top_traffic

###############################################################################################
####################### Experience, Engagement & Satisfaction Analysis ########################
###############################################################################################

class EngagementExperienceScoring:
    def __init__(self, file_path, db_config):  
        self.file_path = file_path
        self.db_config = db_config  
        self.df = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.metrics = [
            'Bearer Id',
            'Avg Bearer TP DL (kbps)',
            'TCP DL Retrans. Vol (Bytes)',
            'Avg RTT Combined (ms)'
        ]

        # Retrieve the password from the environment variable
        self.password = os.getenv("POSTGRES_PASSWORD")  # Using self.password for PostgreSQL
        print(f"Password fetched: {self.password}")

    def load_data(self):
        self.df = pd.read_csv(self.file_path)  # First load the data
        print("Columns in the dataset:", self.df.columns) 

    def preprocess_data(self):
        self.df['Avg RTT Combined (ms)'] = (self.df['Avg RTT DL (ms)'] + self.df['Avg RTT UL (ms)']) / 2
        self.df = self.df[self.metrics].dropna()
        scaler = StandardScaler()
        self.df_scaled = scaler.fit_transform(self.df)
        print("Data normalized for clustering.")
        print(f"Data shape after preprocessing: {self.df.shape}")

    def perform_clustering(self, k=3):
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.df_scaled)
        self.cluster_centers = kmeans.cluster_centers_
        self.df['Cluster'] = self.cluster_labels
        print(f"K-means clustering performed with k={k}.")
        print(f"Cluster centers:\n{self.cluster_centers}")

    def analyze_clusters(self):
        cluster_analysis = self.df.groupby('Cluster').mean()
        print("\nCluster Analysis:")
        print(cluster_analysis)
        os.makedirs('results', exist_ok=True)
        cluster_analysis.to_csv('results/cluster_analysis.csv')
        plt.figure(figsize=(12, 6))
        sns.heatmap(cluster_analysis, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Cluster Characteristics')
        plt.tight_layout()
        plt.savefig('results/cluster_characteristics.png')

    def calculate_scores(self):
        less_engaged_cluster_center = self.cluster_centers[0]
        self.df['Engagement Score'] = np.linalg.norm(self.df_scaled - less_engaged_cluster_center, axis=1)
        worst_experience_cluster_center = self.cluster_centers[2]
        self.df['Experience Score'] = np.linalg.norm(self.df_scaled - worst_experience_cluster_center, axis=1)
        
        print("Engagement, Experience scores, and User numbering calculated.")
        self.df.to_csv('results/user_scores.csv', index=False)

    def calculate_satisfaction_score(self):
        self.df['Satisfaction Score'] = (self.df['Engagement Score'] + self.df['Experience Score']) / 2
        print("Satisfaction scores calculated.")
        print(f"Example satisfaction scores (first 5 rows):\n{self.df[['Bearer Id', 'Satisfaction Score']].head()}")

    def get_top_satisfied_customers(self, top_n=10):
        top_customers = self.df.sort_values(by='Satisfaction Score', ascending=False).head(top_n)
        print("\nTop Satisfied Customers:")
        print(top_customers[['Bearer Id', 'Satisfaction Score']])
        top_customers.to_csv('results/top_10_satisfied_customers.csv', index=False)

    def build_regression_model(self):
        if 'Satisfaction Score' not in self.df.columns:
            self.calculate_satisfaction_score()

        X = self.df.drop(columns=['Bearer Id', 'Cluster', 'Engagement Score', 
                                  'Experience Score', 'Satisfaction Score'])
        y = self.df['Satisfaction Score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR2 Score: {r2:.2f}")
        joblib.dump(reg_model, 'results/satisfaction_score_model.pkl')
        predictions = X_test.copy()
        predictions['True Satisfaction Score'] = y_test
        predictions['Predicted Satisfaction Score'] = y_pred
        predictions.to_csv('results/satisfaction_predictions.csv', index=False)

    def run_kmeans_on_scores(self, k=2):
        """Run K-means clustering (k=2) on the Engagement and Experience Scores."""
        print("Running K-means clustering on Engagement and Experience Scores...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        
        # Use the Engagement and Experience scores for clustering
        scores_df = self.df[['Engagement Score', 'Experience Score']].dropna()
        self.cluster_labels = kmeans.fit_predict(scores_df)
        self.cluster_centers = kmeans.cluster_centers_

        self.df['Engagement-Experience Cluster'] = self.cluster_labels
        print(f"K-means clustering performed on Engagement and Experience scores with k={k}.")
        
        os.makedirs('results', exist_ok=True)
        self.df.to_csv('results/engagement_experience_clustered.csv', index=False)
        print("Results saved as 'engagement_experience_clustered.csv'.")

    def aggregate_scores_per_cluster(self):
        """Aggregate the average satisfaction and experience scores per cluster."""
        cluster_agg = self.df.groupby('Cluster')[['Satisfaction Score', 'Experience Score']].mean()
        print("\nAverage Satisfaction and Experience Scores per Cluster:")
        print(cluster_agg)
        
        cluster_agg.to_csv('results/average_scores_per_cluster.csv')
        print("Average scores per cluster saved as 'average_scores_per_cluster.csv'.")

    def export_to_postgresql(self):
        """Export the final table to a PostgreSQL database."""
        try:
            # Connect to PostgreSQL
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )

            # Check if the connection is established
            if self.connection:
                self.cursor = self.connection.cursor()

                # Create table if not exists
                create_table_query = """
                CREATE TABLE IF NOT EXISTS user_scores (
                    "Bearer Id" VARCHAR(255), 
                    "Avg Bearer TP DL (kbps)" FLOAT,
                    "TCP DL Retrans. Vol (Bytes)" FLOAT,
                    "Avg RTT Combined (ms)" FLOAT,
                    "Cluster" INT,
                    "Engagement Score" FLOAT,
                    "Experience Score" FLOAT,
                    "Satisfaction Score" FLOAT,
                    "Engagement-Experience Cluster" INT
                )
                """
                self.cursor.execute(create_table_query)
                print("Table `user_scores` created or already exists.")

                # Insert data into PostgreSQL
                insert_query = """
                INSERT INTO user_scores ("Bearer Id", "Engagement Score", "Experience Score", "Satisfaction Score")
                VALUES (%s, %s, %s, %s)
                """

                # Convert relevant columns to native Python float (not np.float64)
                self.df['Bearer Id'] = self.df['Bearer Id'].astype(str)  # Ensure 'Bearer Id' is a string, if it's not already
                self.df['Engagement Score'] = self.df['Engagement Score'].astype(float)
                self.df['Experience Score'] = self.df['Experience Score'].astype(float)
                self.df['Satisfaction Score'] = self.df['Satisfaction Score'].astype(float)

                # Ensure all numeric columns are converted to Python's native float (just in case)
                self.df[self.df.select_dtypes(include=['float64', 'int64']).columns] = \
                    self.df.select_dtypes(include=['float64', 'int64']).astype(float)

                print(self.df.dtypes)  # Check the data types after conversion

                # Iterate through rows and explicitly convert to Python native float
                for _, row in self.df.iterrows():
                    # Insert the data into PostgreSQL
                    self.cursor.execute(insert_query, 
                                        (str(row['Bearer Id']),  # Make sure Bearer Id is a string
                                        float(row['Engagement Score']), 
                                        float(row['Experience Score']), 
                                        float(row['Satisfaction Score'])))

                # Commit changes and close the cursor and connection
                self.connection.commit()
                print("Data successfully inserted into PostgreSQL.")

        except Exception as e:
            print(f"Error while exporting to PostgreSQL: {e}")
        finally:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()

    def run_all(self):
        self.load_data()
        self.preprocess_data()
        self.perform_clustering(k=3)
        self.analyze_clusters()
        self.calculate_scores()
        self.calculate_satisfaction_score()
        self.get_top_satisfied_customers(top_n=10)
        self.build_regression_model()
        self.run_kmeans_on_scores(k=2)
        self.aggregate_scores_per_cluster()
        self.export_to_postgresql()

if __name__ == "__main__":
    load_and_clean_data()
    file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temp_data')), 'main_data_source.csv')
    db_config = {
        'host': 'localhost',
        'user': 'root', 
        'password': os.getenv("POSTGRES_PASSWORD"),  
        'database': 'kaim-week-2'  
    }
    scoring_system = EngagementExperienceScoring(file_path, db_config)
    scoring_system.run_all()