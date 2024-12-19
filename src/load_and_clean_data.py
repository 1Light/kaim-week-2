import gdown
import pandas as pd
import os

def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

def load_and_clean_data():

    file_ids = {
        "main_data_source": "13MnZMUugPm43U_fLTmRAh0FuJF-hr1L3"
    }

    # Define local paths to save the data temporarily
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'main_data_source'))
    os.makedirs(base_path, exist_ok=True)  
    
    # Define file paths for the dataset
    data_path = os.path.join(base_path, 'main_data_source.csv')

    # Download files only if they don't exist
    if not os.path.exists(data_path):
        print("Downloading data...")
        download_from_gdrive(file_ids["main_data_source"], data_path)

    df = pd.read_csv(data_path)

    # Step 2: Inspect the structure of the data

    """
    # Print the first few rows to verify successful load
    print(data.head())

    # Check the data types of the columns
    print(df.dtypes)

    # Check for missing data
    print(df.isnull().sum())
    """

    # Correcting the data types
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')  # Convert to datetime, invalid parsing will be set to NaT
    df['End'] = pd.to_datetime(df['End'], errors='coerce')      # Convert to datetime, invalid parsing will be set to NaT
    df['IMSI'] = df['IMSI'].astype(str)  # Convert IMSI to string
    df['MSISDN/Number'] = df['MSISDN/Number'].astype(str)  # Convert MSISDN/Number to string
    df['IMEI'] = df['IMEI'].astype(str)  # Convert IMEI to string
    df['Last Location Name'] = df['Last Location Name'].astype(str)  # Convert to string (if it's an identifier)
    df['Handset Manufacturer'] = df['Handset Manufacturer'].astype(str)  # Convert to string (entire model name)
    df['Handset Type'] = df['Handset Type'].astype(str)  # Convert to string (entire type name)

    numeric_columns = ['Bearer Id', 'Start ms', 'End ms', 'Dur. (ms)', 'Avg RTT DL (ms)', 
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
                    'Other DL (Bytes)', 'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)']

    # Since only 1 row from Start and End is missing, drop it
    df = df.dropna(subset=['Start', 'End'])

    # Fill numeric missing values with the median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Define local paths to save the data temporarily
    cleaned_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cleaned_data', 'main_data_source'))
    os.makedirs(cleaned_path, exist_ok=True)  

    # Save the cleaned data to a new CSV file
    cleaned_data_path = os.path.join(cleaned_path, 'main_data_source.csv')
    df.to_csv(cleaned_data_path, index=False)  # Save without the index column

    print(df.dtypes)

load_and_clean_data()