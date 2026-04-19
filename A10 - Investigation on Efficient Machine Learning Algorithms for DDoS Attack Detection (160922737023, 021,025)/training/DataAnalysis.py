import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "../data/dataset_sdn.csv"
EDA_DIR = "../static/eda"
TARGET = "label"

os.makedirs(EDA_DIR, exist_ok=True)

sns.set(style="whitegrid")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Handle missing values explicitly (EDA-safe)
df = df.fillna(df.median(numeric_only=True))

# -----------------------------
# 1. Class Distribution
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=TARGET, data=df)
plt.title("Class Distribution (Normal vs DDOS)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/class_distribution.png")
plt.close()

# -----------------------------
# 2. Numeric Feature Distributions
# -----------------------------
numeric_cols = df.drop(columns=[TARGET]).select_dtypes(include=["int64", "float64"]).columns

df[numeric_cols].hist(
    figsize=(16, 12),
    bins=30,
    edgecolor="black"
)

plt.suptitle("Numeric Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/numeric_distributions.png")
plt.close()

# -----------------------------
# 3. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(18, 14))

corr = df[numeric_cols.tolist() + [TARGET]].corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,                 # SHOW numbers
    fmt=".1f",                  # 2 decimal places
    annot_kws={"size": 12},      # Control number size
    linewidths=0.5,
    cbar=True
)

plt.title("Feature Correlation Heatmap", fontsize=18)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig(f"{EDA_DIR}/correlation_heatmap.png", dpi=300)
plt.close()

# -----------------------------
# 4. Feature vs Label (Boxplots)
# -----------------------------
sample_features = numeric_cols[:10]  # limit to top 10 for clarity

plt.figure(figsize=(18, 10))
for i, col in enumerate(sample_features, 1):
    plt.subplot(2, 5, i)
    sns.boxplot(x=TARGET, y=col, data=df)
    plt.title(col)

plt.suptitle("Feature vs Label Distribution (Boxplots)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{EDA_DIR}/feature_vs_label_boxplots.png")
plt.close()

print("✅ EDA complete. Graphs saved in /eda folder.")
