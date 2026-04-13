import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Create EDA directory
os.makedirs("../static/eda", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/Mental Health Dataset.csv")

# Define target variable
TARGET = "Mood_Swings"   # column name as in dataset

# -----------------------------
# 1. Target Variable Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=TARGET, data=df)
plt.title("Mood Swings Distribution")
plt.tight_layout()
plt.savefig("../static/eda/mood_swings_distribution.png")
plt.close()

# -----------------------------
# 2. Target vs Categorical Features
# -----------------------------
categorical_cols = [
    col for col in df.columns
    if df[col].dtype == "object" and col not in [TARGET, "Timestamp"]
]

for col in categorical_cols:
    plt.figure(figsize=(7,4))
    sns.countplot(x=col, hue=TARGET, data=df)
    plt.title(f"Mood Swings vs {col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"../static/eda/mood_swings_vs_{col}.png")
    plt.close()

# -----------------------------
# 3. Target vs Numerical Features
# -----------------------------
numerical_cols = [
    col for col in df.columns
    if df[col].dtype != "object" and col != TARGET
]

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=TARGET, y=col, data=df)
    plt.title(f"Mood Swings vs {col}")
    plt.tight_layout()
    plt.savefig(f"../static/eda/mood_swings_vs_{col}_boxplot.png")
    plt.close()

