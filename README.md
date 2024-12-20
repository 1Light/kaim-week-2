# KAIM Week 2: Telecommunication Data Analysis  

## Project Overview  
This project aims to analyze telecommunication data from TellCo, a mobile service provider in the Republic of Pefkakia. The goal is to identify opportunities for growth and provide actionable recommendations to a potential investor regarding the purchase of TellCo. The analysis focuses on user behavior, engagement, experience, and satisfaction, leveraging advanced data science techniques.  

Key deliverables include:  
- **Reusable Code**: For data preparation and cleaning, implemented with scikit pipelines.  
- **Streamlit Dashboard**: Interactive visualization of insights.  
- **SQL Database**: Used as a feature store for dashboard visualization and model training.  
- **CI/CD Pipeline**: Automated workflows with GitHub Actions.  
- **Dockerized Deployment**: For portability and ease of use.  

## Objectives  
The project is divided into the following sub-objectives:  
1. **User Overview Analysis**  
   - Identify top devices and manufacturers used by customers.  
   - Analyze user behavior on various applications like social media, YouTube, Netflix, etc.  
   - Handle missing values and outliers for robust insights.  
2. **User Engagement Analysis**  
   - Measure engagement through metrics like session frequency, duration, and traffic.  
   - Cluster users based on engagement scores using k-means clustering.  
   - Identify top applications and most engaged users.  
3. **User Experience Analysis** *(Future Scope)*  
4. **User Satisfaction Analysis** *(Future Scope)*  

## Folder Structure  
```
kaim-week-2/
├── src/                  # Core source code for analysis, models, and utilities
├── scripts/              # Python scripts for task automation
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
```  

## Key Features  
- **Data Preparation and Cleaning**  
   - Handle missing values and outliers.  
   - Transform variables for segmentation and analysis.  
- **Exploratory Data Analysis (EDA)**  
   - Univariate, bivariate, and correlation analyses.  
   - Dimensionality reduction using PCA.  
- **Engagement Metrics**  
   - Cluster analysis of user engagement.  
   - Visualization of top applications and metrics.  
- **Dashboard**  
   - A Streamlit-based dashboard for presenting insights interactively.  

## Setup Instructions  
1. Clone the repository:  
   ```bash
   git clone https://github.com/1Light/kaim-week-2.git
   cd kaim-week-2
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run tests:  
   ```bash
   pytest tests/
   ```  
4. Start the dashboard:  
   ```bash
   streamlit run src/dashboard.py
   ```  
5. Build Docker image (optional):  
   ```bash
   docker build -t kaim-week-2 .
   docker run -p 8501:8501 kaim-week-2
   ```  

## Insights and Recommendations  
### User Overview Analysis  
- **Top Devices**: Identify the top 10 handsets and their manufacturers.  
- **Application Behavior**: Aggregated xDR metrics (sessions, duration, data volume) reveal key usage patterns.  

### User Engagement Analysis  
- **Engagement Clusters**: Users are segmented into 3 clusters using k-means clustering.  
- **Top Applications**: Visualize top 3 applications by usage.  

## Contributing  
Contributions are welcome! Please submit a pull request or open an issue for improvements or bug fixes.  

## License  
This project is licensed under the [MIT License](LICENSE).