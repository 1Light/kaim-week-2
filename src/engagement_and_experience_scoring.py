import pandas as pd
import os
import mysql.connector
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load environment variables from .env file
load_dotenv()

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
        self.password = os.getenv("MYSQL_PASSWORD")  # Now using self.password
        print(f"Password fetched: {self.password}")

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")
        print(f"Data shape: {self.df.shape}")

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
        print("Engagement and Experience scores calculated.")
        print(f"Example scores (first 5 rows):\n{self.df[['Bearer Id', 'Engagement Score', 'Experience Score']].head()}")
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

    def export_to_mysql(self):
        """Export the final table to a MySQL database."""
        try:
            # Connect to MySQL
            self.connection = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            
            # Check if the connection is established
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()

                # Create table if not exists
                create_table_query = """
                CREATE TABLE IF NOT EXISTS user_scores (
                    `Bearer Id` VARCHAR(255), 
                    `Avg Bearer TP DL (kbps)` FLOAT,
                    `TCP DL Retrans. Vol (Bytes)` FLOAT,
                    `Avg RTT Combined (ms)` FLOAT,
                    `Cluster` INT,
                    `Engagement Score` FLOAT,
                    `Experience Score` FLOAT,
                    `Satisfaction Score` FLOAT,
                    `Engagement-Experience Cluster` INT
                )
                """
                self.cursor.execute(create_table_query)
                print("Table `user_scores` created or already exists.")

                # Insert data into MySQL
                insert_query = """
                INSERT INTO user_scores (Bearer_Id, Engagement_Score, Experience_Score, Satisfaction_Score)
                VALUES (%s, %s, %s, %s)
                """

                # Convert relevant columns to float32
                self.df['Bearer Id'] = self.df['Bearer Id'].astype('float32')
                self.df['Engagement Score'] = self.df['Engagement Score'].astype('float32')
                self.df['Experience Score'] = self.df['Experience Score'].astype('float32')
                self.df['Satisfaction Score'] = self.df['Satisfaction Score'].astype('float32')

                # Convert other numeric columns (e.g., int64 and float64) to float32
                self.df[self.df.select_dtypes(include=['float64', 'int64']).columns] = self.df.select_dtypes(include=['float64', 'int64']).astype('float32')

                print(self.df.dtypes)

                # Iterate through rows and explicitly convert to Python native float
                for _, row in self.df.iterrows():
                    # Explicitly convert to native Python float
                    engagement_score = float(row['Engagement Score'])
                    experience_score = float(row['Experience Score'])
                    satisfaction_score = float(row['Satisfaction Score'])

                    # Insert the data into MySQL
                    self.cursor.execute(insert_query, 
                                        (row['Bearer Id'], 
                                        engagement_score, 
                                        experience_score, 
                                        satisfaction_score))

                self.connection.commit()
                print("Data exported to MySQL database successfully.")

                # Verify by running a SELECT query
                self.cursor.execute("SELECT * FROM user_scores LIMIT 10;")
                result = self.cursor.fetchall()
                print("\nSample data from the database:")
                for row in result:
                    print(row)
            else:
                print("Failed to connect to the database.")
                
        except mysql.connector.Error as err:
            print(f"Error: {err}")
        finally:
            # Ensure that cursor and connection are closed properly
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()

    def process(self):
        self.load_data()
        self.preprocess_data()
        self.perform_clustering(k=3)
        self.analyze_clusters()
        self.calculate_scores()
        self.calculate_satisfaction_score()
        self.get_top_satisfied_customers()
        self.build_regression_model()
        self.run_kmeans_on_scores(k=2)
        self.aggregate_scores_per_cluster()
        self.export_to_mysql() 

if __name__ == "__main__":

    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': os.getenv('MYSQL_PASSWORD'),
        'database': 'kaim-week-2'
    }

    file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')
    scoring = EngagementExperienceScoring(file_path, db_config)  
    scoring.process()