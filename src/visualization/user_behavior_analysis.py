import pandas as pd
import os
import matplotlib.pyplot as plt

class UserBehaviorAnalyzer:
    def __init__(self, file_path):
        """Initialize the class with the dataset path"""
        self.file_path = file_path
        self.df = None
        self.user_behavior = None

    def load_data(self):
        """Load the CSV file into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
    
    def aggregate_user_behavior(self):
        """Aggregate relevant information per IMSI (user)"""
        if 'IMSI' not in self.df.columns:
            print("Error: 'IMSI' column is missing in the dataset.")
            return
        
        self.user_behavior = self.df.groupby('IMSI').agg(
            number_of_sessions=('Dur. (ms)', 'count'),
            total_session_duration=('Dur. (ms)', 'sum'),
            total_download_data=('Total DL (Bytes)', 'sum'),
            total_upload_data=('Total UL (Bytes)', 'sum'),
        ).reset_index()

        # Add the total_data_volume column
        self.user_behavior['total_data_volume'] = self.user_behavior['total_download_data'] + self.user_behavior['total_upload_data']

    def handle_missing_values(self):
        """Handle missing values by replacing them with the mean"""
        if self.user_behavior.isnull().sum().any():
            print(f"Warning: Some missing values found in the aggregated data. Handling missing values by replacing with the mean.")
            self.user_behavior.fillna(self.user_behavior.mean(), inplace=True)

    def display_user_behavior(self):
        """Display the aggregated user behavior data"""
        print("Aggregated User Behavior Data (Total Data Volume per IMSI):")
        print(self.user_behavior[['IMSI', 'total_data_volume']].head())  # Print the first few rows for inspection

    def plot_data_volume(self):
        """Generate and save a plot for total data volume per IMSI"""
        fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size
        self.user_behavior.plot(kind='bar', x='IMSI', y='total_data_volume', ax=ax, color='#4CAF50')
        ax.set_title('Total Data Volume per User (IMSI)')
        ax.set_xlabel('User (IMSI)')
        ax.set_ylabel('Total Data Volume (Bytes)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate labels for readability

        # Create the 'results/user_behavior' folder if it doesn't exist
        os.makedirs('results/user_behavior', exist_ok=True)

        # Save the plot as an image in the specified folder
        plt.tight_layout()  # Adjust layout to avoid cutting off labels
        plt.savefig('results/user_behavior/total_data_volume_per_user.png', dpi=300)
        print("User behavior plot saved as 'total_data_volume_per_user.png'")

    def run(self):
        """Run the complete process of analyzing user behavior"""
        self.load_data()
        if self.df is not None:
            self.aggregate_user_behavior()
            self.handle_missing_values()
            self.display_user_behavior()
            self.plot_data_volume()


# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the UserBehaviorAnalyzer class
user_behavior_analyzer = UserBehaviorAnalyzer(file_path)

# Run the user behavior analysis
user_behavior_analyzer.run()
