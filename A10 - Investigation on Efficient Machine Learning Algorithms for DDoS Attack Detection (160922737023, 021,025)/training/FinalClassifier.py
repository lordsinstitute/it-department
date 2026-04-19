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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =============================
# CONFIGURATION
# =============================
DATA_PATH = "../data/dataset_sdn.csv"
TARGET = "label"

OUTPUT_DIR = "../static/final_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)

# =============================
# SEPARATE FEATURES & TARGET
# =============================
X = df.drop(columns=[TARGET])
y = df[TARGET]

# =============================
# KEEP ONLY NUMERIC FEATURES
# =============================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
X = X[numeric_cols]

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
# SCALING
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# CANDIDATE MODELS
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    #"SVM (Linear)": SVC(kernel="linear", probability=True),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
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
    pass

# =============================
# SELECT BEST MODEL (F1 SCORE)
# =============================
best_model = None
best_model_name = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    f1 = f1_score(y_test, preds)

    print(f"{name} → F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\n✅ Best Model Selected: {best_model_name}")

# =============================
# FINAL EVALUATION
# =============================
y_pred = best_model.predict(X_test_scaled)

y_proba = (
    best_model.predict_proba(X_test_scaled)[:, 1]
    if hasattr(best_model, "predict_proba")
    else None
)

metrics = {
    "Model": best_model_name,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None
}

# =============================
# SAVE METRICS
# =============================
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)

with open(f"{OUTPUT_DIR}/metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

# =============================
# CLASSIFICATION REPORT
# =============================
report = classification_report(y_test, y_pred)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

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
plt.title("Confusion Matrix – Best Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

# =============================
# SAVE FINAL ARTIFACTS
# =============================
joblib.dump(best_model, f"{OUTPUT_DIR}/best_model.pkl")
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
joblib.dump(list(numeric_cols), f"{OUTPUT_DIR}/feature_order.pkl")

print("\n🎯 Final model and metrics saved successfully")
print(metrics_df)
