import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Paths
# ===============================
DATA_PATH = "parkinsons.data"
STATIC_DIR = "static/eda"

os.makedirs(STATIC_DIR, exist_ok=True)


def dataAnalysis():
    # ===============================
    # Load Dataset
    # ===============================
    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nMissing Values:\n", df.isnull().sum())

    # ===============================
    # Target Variable Distribution
    # ===============================
    plt.figure()
    df['status'].value_counts().plot(kind='bar')
    plt.title("Parkinson’s Disease Distribution")
    plt.xlabel("Status (0 = Healthy, 1 = Parkinson’s)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{STATIC_DIR}/class_distribution.jpg")
    plt.close()

    # ===============================
    # Correlation Heatmap
    # ===============================


    plt.figure(figsize=[15, 8], dpi=100)
    plt.title("Correlation Graph", fontsize=20)
    numerical = df.drop('name', axis=1)
    cmap = sns.color_palette("Blues")
    sns.heatmap(numerical.corr(), annot=True, cmap=cmap)
    plt.savefig(f"{STATIC_DIR}/correlation_heatmap.jpg")
    plt.show()

    # ===============================
    # Feature Distribution Plots
    # ===============================
    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Shimmer',
        'HNR', 'RPDE', 'DFA', 'spread1', 'spread2'
    ]

    for feature in features:
        plt.figure()
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of {feature}")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/{feature.replace(':', '_')}_distribution.jpg")
        plt.close()

    # ===============================
    # Boxplots: Parkinson vs Healthy
    # ===============================
    for feature in features:
        plt.figure()
        sns.boxplot(x='status', y=feature, data=df)
        plt.title(f"{feature} vs Disease Status")
        plt.xlabel("Status (0 = Healthy, 1 = Parkinson’s)")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/{feature.replace(':', '_')}_boxplot.jpg")
        plt.close()

    print("EDA graphs successfully saved in static/ folder")


#dataAnalysis()


