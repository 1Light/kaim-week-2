import gdown
import pandas as pd
import os
import shutil

class DataDownloader:
    def __init__(self, file_ids, output_dir):
        self.file_ids = file_ids
        self.output_dir = output_dir

    def download_from_gdrive(self, file_id, output_path):
        """Download a file from Google Drive using gdown."""
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    def download_data(self):
        """Download the necessary data if it doesn't already exist."""
        os.makedirs(self.output_dir, exist_ok=True)  

        # Define file paths for the dataset
        data_path = os.path.join(self.output_dir, 'main_data_source.csv')

        # Download files only if they don't exist
        if not os.path.exists(data_path):
            print("Downloading data...")
            self.download_from_gdrive(self.file_ids["main_data_source"], data_path)

        return data_path

class DataCleaner:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.numeric_columns = [
            'Bearer Id', 'Start ms', 'End ms', 'Dur. (ms)', 'Avg RTT DL (ms)', 
            'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
            'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)', 
            '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 
            'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)', 
            '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)', 
            'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)', 
            'Activity Duration UL (ms)', 'Dur. (ms).1', 'Nb of sec with 125000B < Vol DL', 
            'Nb of sec with 1250B < Vol UL < 6250B', 'Nb of sec with 31250B < Vol DL < 125000B', 
            'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B', 
            'Nb of sec with 6250B < Vol UL < 37500B', 'Nb of sec with Vol DL < 6250B', 
            'Nb of sec with Vol UL < 1250B', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
            'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 
            'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
            'Other DL (Bytes)', 'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)'
        ]

    def clean_data(self):
        """Clean the data (fix data types, handle missing values, etc.)."""
        # Correcting the data types
        self.df['Start'] = pd.to_datetime(self.df['Start'], errors='coerce')
        self.df['End'] = pd.to_datetime(self.df['End'], errors='coerce')
        self.df['IMSI'] = self.df['IMSI'].astype(str)
        self.df['MSISDN/Number'] = self.df['MSISDN/Number'].astype(str)
        self.df['IMEI'] = self.df['IMEI'].astype(str)
        self.df['Last Location Name'] = self.df['Last Location Name'].astype(str)
        self.df['Handset Manufacturer'] = self.df['Handset Manufacturer'].astype(str)
        self.df['Handset Type'] = self.df['Handset Type'].astype(str)

        # Drop rows where 'Start' or 'End' is missing
        self.df = self.df.dropna(subset=['Start', 'End'])

        # Fill missing numeric values with the median 
        # Chose median and not mean or mode because it is less affected by outliers
        self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())

        return self.df

class DataSaver:
    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir

    def clear_directory(self):
        """Clear the output directory if it exists."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print(f"Existing directory {self.output_dir} cleared.")

    def save_cleaned_data(self):
        """Save the cleaned data to a new CSV file."""
        self.clear_directory()  # Clear the directory first
        os.makedirs(self.output_dir, exist_ok=True)  # Create directory if not exists
        cleaned_path = os.path.join(self.output_dir, 'main_data_source.csv')
        self.df.to_csv(cleaned_path, index=False)
        print(f"Cleaned data saved to {cleaned_path}")

def main():
    file_ids = {"main_data_source": "13MnZMUugPm43U_fLTmRAh0FuJF-hr1L3"}
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'main_data_source'))

    # Step 1: Download data
    downloader = DataDownloader(file_ids, base_path)
    data_path = downloader.download_data()

    # Step 2: Clean the data
    cleaner = DataCleaner(data_path)
    cleaned_df = cleaner.clean_data()

    # Step 3: Save the cleaned data
    cleaned_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cleaned_data', 'main_data_source'))
    saver = DataSaver(cleaned_df, cleaned_data_path)
    saver.save_cleaned_data()

# Assign `main` to `prepare_data` for easier imports
prepare_data = main

# Run the main function to execute the whole process
if __name__ == "__main__":
    main()