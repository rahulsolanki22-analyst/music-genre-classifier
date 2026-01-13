import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH  ="features.csv"

try:
    # Read the CSV file into a DataFrame.
    features_df = pd.read_csv(CSV_PATH)

    print("DataFrame Loaded Succeefully")

    print("\n--- DataFrame Info ---")
    features_df.info()

    print("\n--- DataFrame Statistical Summary ---")
    print(features_df.describe())

    print("\n--- Missing Values Check ---")
    missing_values_count = features_df.isnull().sum()

    print("Numbers of missing values per column:")
    print(missing_values_count)

    if missing_values_count.sum() == 0:
        print("\n--- No missing values found in the DataFrame. No handling is required")
    else:
        print("\nComclusion: The dataset contains missing values. handling is required")

    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]

    #Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,6))

    # Create the count plot
    ax = sns.countplot(x='genre_label',data=features_df,palette='viridis')
    
    ax.set_title('Distribution of Music Genres in the Dataset',fontsize=16)
    ax.set_xlabel('Genre',fontsize=12)
    ax.set_ylabel('Number of Segments',fontsize = 12)

    ax.set_xticklabels(genre_names, rotation = 30)

    #Display the plot
    plt.tight_layout()
    plt.show()


    # Create box plots to compare the distribution of spectral centroid across genres.
    print("\n--- Generating Box Plot for Spectral Centroid ---")
    plt.figure(figsize=(14,7))
    box_ax = sns.boxplot(x='genre_label',y='25',data=features_df,palette='cubehelix')

    box_ax.set_title('Spectral Centroid Distribution Across Genres',fontsize=18)
    box_ax.set_xlabel('Genre',fontsize=14)
    box_ax.set_ylabel('Spectral Centroid',fontsize=14)

    box_ax.set_xticklabels(genre_names,rotation=30,ha='right')

    plt.tight_layout()
    plt.show()


    print("\n--- Generating Violin Plot for first mfcc (column 0) ---")
    plt.figure(figsize=(14,7))
    violin_ax = sns.violinplot(x='genre_label',y='0',data=features_df,palette='Spectral')

    violin_ax.set_title('Fiest MFCC (Timbre/Energy) Distribution Across Genres',fontsize=18)
    violin_ax.set_xlabel('Genre',fontsize=14)
    violin_ax.set_ylabel('MFCC 1 Value',fontsize=14)
    violin_ax.set_xticklabels(genre_names,rotation=30,ha='right')

    plt.tight_layout()
    plt.show()

    print("\n--- Computing Correlation Matrix ---")

    correlation_matrix = features_df.corr()

    print("Correlation matrix computer successfully.")

    #print("Top 5 rows of the correlation matrix:")
    #print(correlation_matrix.head())

    print("\n--- Generating Heatmap of Feature Correlations ---")

    plt.figure(figsize=(18,15))
    sns.heatmap(correlation_matrix,cmap='coolwarm', annot=False)
    plt.title('Correlation Matrix of Music Features',fontsize=20)
    plt.tight_layout()
    plt.show()

    #print("\nFirst 5 rows of the dataset:")
    #print(features_df.head())

except FileNotFoundError:
    print(f"Error:The file at '{CSV_PATH}' was not found.")
    print("Please ensure you have run the 'feature_extractor.py' script first to generate the dataset.")
except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")