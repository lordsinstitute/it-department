# compare_models.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "../data/dataset_sdn.csv"
TARGET = "label"

REPORT_DIR = "../static/evaluation/classification_reports"
CM_DIR = "../static/evaluation/confusion_matrices"
MODEL_DIR="../models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)


# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Initial shape:", df.shape)

# =============================
# SEPARATE FEATURES & TARGET
# =============================
X = df.drop(columns=[TARGET])
y = df[TARGET]

# =============================
# KEEP ONLY NUMERIC FEATURES (CRITICAL FIX)
# =============================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

dropped_cols = set(X.columns) - set(numeric_cols)
print("\nDropped non-numeric columns:")
print(dropped_cols)

X = X[numeric_cols]

print("\nFinal feature count:", X.shape[1])

# =============================
# HANDLE MISSING VALUES
# =============================
X = X.fillna(X.median())

# =============================
# TRAIN-TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# SCALING (NUMERIC ONLY)
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, f"{MODEL_DIR}/scaler_for_comparison.pkl")
joblib.dump(list(numeric_cols), f"{MODEL_DIR}/feature_order_for_comparison.pkl")

# =============================
# MODELS TO COMPARE
# =============================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    #"SVM_Linear": SVC(kernel="linear", probability=True),
    "KNN": KNeighborsClassifier(),
    "Decision_Tree": DecisionTreeClassifier(random_state=42)
}

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
except ImportError:
    print("XGBoost not installed – skipping.")

# =============================
# TRAIN & EVALUATE
# =============================
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    # =============================
    # CLASSIFICATION REPORT
    # =============================
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{REPORT_DIR}/{name}_report.csv")

    # =============================
    # CONFUSION MATRIX
    # =============================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"{CM_DIR}/{name}_cm.png", dpi=300)
    plt.close()

# =============================
# SUMMARY COMPARISON
# =============================
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
results_df.to_csv("../static/evaluation/model_comparison_results.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="F1 Score",
    y="Model",
    data=results_df,
    palette="viridis"
)
plt.title("DDOS Detection – Model Comparison (F1 Score)")
plt.tight_layout()
plt.savefig("../static/evaluation/model_comparison_plot.png", dpi=300)
plt.show()

print("\n✅ Model comparison completed successfully")
print(results_df)
