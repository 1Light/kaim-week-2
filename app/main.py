import streamlit as st
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils import user_engagement_analysis, EngagementExperienceScoring

# Set page title first (before any other Streamlit function)
st.set_page_config(page_title="Analysis Dashboard")

# Load environment variables from .env file
load_dotenv()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["User Engagement Analysis", "Engagement, Experience & Satisfaction Analysis"])

# Path to the data file
file_path = 'cleaned_data/main_data_source/main_data_source.csv'

# --- User Engagement Analysis Page ---
if page == "User Engagement Analysis":
    st.title("User Engagement and Experience Analysis")

    # Run the user engagement analysis from utils
    top_sessions, top_duration, top_traffic = user_engagement_analysis(file_path)

    # Display results for top 10 customers by sessions frequency
    st.header("Top 10 Customers by Sessions Frequency")
    st.write(top_sessions)

    # Plot top sessions frequency
    st.subheader("Sessions Frequency Distribution")
    fig, ax = plt.subplots()
    ax.bar(top_sessions['MSISDN/Number'].astype(str), top_sessions['sessions_frequency'])
    ax.set_xlabel('Customer ID (MSISDN/Number)')
    ax.set_ylabel('Sessions Frequency')
    ax.set_title('Top 10 Customers by Sessions Frequency')
    
    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=90, ha='center')  # Rotate labels 90 degrees
    st.pyplot(fig)

    # Display results for top 10 customers by total session duration
    st.header("Top 10 Customers by Total Session Duration")
    st.write(top_duration)

    # Plot total session duration
    st.subheader("Total Session Duration Distribution")
    fig, ax = plt.subplots()
    ax.bar(top_duration['MSISDN/Number'].astype(str), top_duration['total_session_duration'])
    ax.set_xlabel('Customer ID (MSISDN/Number)')
    ax.set_ylabel('Total Session Duration (ms)')
    ax.set_title('Top 10 Customers by Total Session Duration')
    
    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=90, ha='center')  # Rotate labels 90 degrees
    st.pyplot(fig)

    # Display results for top 10 customers by total traffic
    st.header("Top 10 Customers by Total Traffic")
    st.write(top_traffic)

    # Plot total traffic
    st.subheader("Total Traffic Distribution")
    fig, ax = plt.subplots()
    ax.bar(top_traffic['MSISDN/Number'].astype(str), top_traffic['total_traffic'])
    ax.set_xlabel('Customer ID (MSISDN/Number)')
    ax.set_ylabel('Total Traffic (Bytes)')
    ax.set_title('Top 10 Customers by Total Traffic')
    
    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=90, ha='center')  # Rotate labels 90 degrees
    st.pyplot(fig)

# --- Engagement, Experience & Satisfaction Scoring Analysis Page ---
if page == "Engagement, Experience & Satisfaction Analysis":
    st.title("Engagement and Experience Scoring Analysis")

    # Initialize the EngagementExperienceScoring class
    db_config = {
        'host': 'localhost',
        'user': 'root', 
        'password': os.getenv("POSTGRES_PASSWORD"),  
        'database': 'kaim-week-2'  
    }

    # Use the class for further analysis (e.g., clustering and scoring)
    ee_scoring = EngagementExperienceScoring(file_path, db_config)

    # Load data and preprocess
    ee_scoring.load_data()
    ee_scoring.preprocess_data()

    # Perform K-means clustering
    ee_scoring.perform_clustering(k=3)

    # Analyze clusters and calculate scores
    ee_scoring.analyze_clusters()
    ee_scoring.calculate_scores()
    ee_scoring.calculate_satisfaction_score()

    # Reset the index (no longer needed as index column is dropped in load_data())
    ee_scoring.df.reset_index(drop=True, inplace=True)

    # Ensure 'Bearer Id' is treated as a string
    ee_scoring.df['Bearer Id'] = ee_scoring.df['Bearer Id'].astype(str)

    # Display engagement and experience scores for top 10 customers
    top_scored_customers = ee_scoring.df[['Bearer Id', 'Engagement Score', 'Experience Score', 'Satisfaction Score']].sort_values(by='Satisfaction Score', ascending=False).head(10)
    st.write(top_scored_customers)

    # Plot the top scored customers based on Satisfaction Score
    st.subheader("Top 10 Customers by Satisfaction Score")
    fig, ax = plt.subplots()
    ax.bar(top_scored_customers['Bearer Id'], top_scored_customers['Satisfaction Score'])  # Use 'Bearer Id' on x-axis
    ax.set_xlabel('Customer (Bearer Id)')
    ax.set_ylabel('Satisfaction Score')
    ax.set_title('Top 10 Customers by Satisfaction Score')

    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=90, ha='center')  # Rotate labels 90 degrees
    st.pyplot(fig)