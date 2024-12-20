import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class AppUsagePlotter:
    def __init__(self, file_path):
        # Initialize the object with the file path
        self.file_path = file_path
        self.df = None
        self.app_columns = [
            'Social Media DL (Bytes)',
            'Google DL (Bytes)',
            'Email DL (Bytes)',
            'Youtube DL (Bytes)',
            'Netflix DL (Bytes)',
            'Gaming DL (Bytes)'
        ]
    
    def load_data(self):
        """Load the CSV file into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
    
    def calculate_app_usage(self):
        """Calculate total data usage for each application"""
        app_usage = self.df[self.app_columns].sum().sort_values(ascending=False)
        return app_usage
    
    def get_top_apps(self, app_usage, top_n=3):
        """Get the top N most used applications"""
        top_apps = app_usage.head(top_n)
        return top_apps
    
    def plot_top_apps(self, top_apps):
        """Plot and save the top N applications by download data"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_apps.index, y=top_apps.values, color='blue')
        plt.title('Top Most Used Applications by Download Data (Bytes)')
        plt.xlabel('Application')
        plt.ylabel('Total Download Bytes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('top_apps.png')
        plt.close()  # Close plot to avoid display in non-interactive environments
        print("Visualization saved: top_apps.png")
    
    def display_top_apps(self, top_apps):
        """Display the top N most used applications and their usage"""
        print("Top Most Used Applications by Download Data (Bytes):")
        for app, usage in top_apps.items():
            print(f"{app}: {usage} Bytes")
    
    def run(self):
        """Run the complete process of loading, calculating, and plotting"""
        self.load_data()
        if self.df is not None:
            app_usage = self.calculate_app_usage()
            top_apps = self.get_top_apps(app_usage)
            self.display_top_apps(top_apps)
            self.plot_top_apps(top_apps)

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Create an instance of the AppUsagePlotter class
app_usage_plotter = AppUsagePlotter(file_path)

# Run the application data plotting
app_usage_plotter.run()
