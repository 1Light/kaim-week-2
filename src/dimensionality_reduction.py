import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Use 'Agg' backend to avoid issues in non-interactive environments (e.g., headless server)
matplotlib.use('Agg')

def dimensionality_reduction(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define the columns related to the analysis
    columns_to_analyze = [
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
        'Google DL (Bytes)', 'Google UL (Bytes)',
        'Email DL (Bytes)', 'Email UL (Bytes)',
        'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
        'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
        'Other DL (Bytes)', 'Other UL (Bytes)'
    ]

    # Select the relevant columns
    data = df[columns_to_analyze]

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    pca_result = pca.fit_transform(data_scaled)

    # Create a DataFrame for the PCA result
    pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])

    # Plot the PCA result
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_df['Principal Component 1'], y=pca_df['Principal Component 2'])
    plt.title('PCA - Reduced Dimensions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # Save the plot to a file
    plt.savefig('results/graphs/pca_reduced_dimensions.png')

    # Explained variance ratio (variance explained by each principal component)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Cumulative explained variance
    print("Cumulative Explained Variance:", pca.explained_variance_ratio_.cumsum())

    # Interpretation
    print("\nInterpretation of PCA Results:")
    
    # High explained variance in the first component
    if pca.explained_variance_ratio_[0] > 0.7:
        print("Insight: The first principal component (PC1) explains more than 70% of the variance, suggesting that this component captures the dominant pattern in the data.")
    else:
        print("Insight: The first principal component (PC1) explains less than 70% of the variance, indicating that the variance is more evenly distributed among the components.")

    # Analyzing the second component's contribution
    if pca.explained_variance_ratio_[1] > 0.3:
        print(f"Insight: The second principal component (PC2) explains {pca.explained_variance_ratio_[1] * 100:.2f}% of the variance, which is relatively significant in capturing the remaining patterns in the data.")
    else:
        print(f"Insight: The second principal component (PC2) explains only {pca.explained_variance_ratio_[1] * 100:.2f}% of the variance, indicating a lesser contribution compared to PC1.")

    # Cumulative variance explained
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    if cumulative_variance[-1] > 0.9:
        print(f"Insight: The first two principal components together explain {cumulative_variance[-1] * 100:.2f}% of the total variance, which is high and suggests that these components capture most of the variance in the data.")
    else:
        print(f"Insight: The first two principal components together explain {cumulative_variance[-1] * 100:.2f}% of the total variance, meaning more components may be needed for a more complete representation.")

    print("""\nGeneral Interpretation of PCA Results:
    - The first principal component (PC1) explains a large portion of the variance in the data. This indicates the most significant pattern across the applications.
    - The second principal component (PC2) explains a smaller portion of the remaining variance, representing the second most important pattern of variability.
    - Together, these two components can capture a significant portion of the total variance in the data, enabling a reduced representation of the original features.
    - By projecting the data onto these two principal components, we achieve dimensionality reduction without losing much information, making analysis more efficient and easier to visualize.
    """)

# Define the file path for the dataset
file_path = os.path.join('cleaned_data', 'main_data_source', 'main_data_source.csv')

# Call the function to perform dimensionality reduction
dimensionality_reduction(file_path)
