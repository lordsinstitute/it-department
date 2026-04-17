import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ======================================================
# 1️⃣ Create Output Folders
# ======================================================

stacked_dir = "static/stacked_outputs"
best_dir = "best_stacked_outputs"

os.makedirs(stacked_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)

# ======================================================
# 2️⃣ Load Dataset
# ======================================================

df = pd.read_csv("engine_data.csv")
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

X = df.drop("Engine Condition", axis=1)
y = df["Engine Condition"]

if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ======================================================
# 3️⃣ Train-Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# 4️⃣ Define Models
# ======================================================

rf = RandomForestClassifier(n_estimators=300, random_state=42)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

estimators = [('rf', rf), ('xgb', xgb)]

stack1 = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stack2 = StackingClassifier(
    estimators=estimators,
    final_estimator=SVC(probability=True),
    cv=5
)

stack3 = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=200),
    cv=5
)

models = {
    "Random_Forest": rf,
    "XGBoost": xgb,
    "Stacking_LogReg": stack1,
    "Stacking_SVM": stack2,
    "Stacking_RF": stack3
}

# ======================================================
# 5️⃣ Train, Evaluate and Save Metrics
# ======================================================

metrics_summary = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    metrics_summary[name] = {
        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }

    # Save Classification Report
    report = classification_report(y_test, y_test_pred)
    with open(os.path.join(stacked_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(report)

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(stacked_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # Save Model
    joblib.dump(model, os.path.join(stacked_dir, f"{name}_model.pkl"))

# ======================================================
# 6️⃣ Save All Models Performance Summary
# ======================================================

metrics_df = pd.DataFrame(metrics_summary).T
metrics_df.to_csv(os.path.join(stacked_dir, "all_models_performance_metrics.csv"))

print("\nAll model metrics saved in 'stacked_outputs/'")

# ======================================================
# 7️⃣ Identify Best Model (Based on Test Accuracy)
# ======================================================

best_model_name = metrics_df["Test_Accuracy"].idxmax()
best_metrics = metrics_df.loc[best_model_name]

print("\nBest Model:", best_model_name)
print(best_metrics)

# Save Best Model Metrics
best_metrics.to_frame().to_csv(os.path.join(best_dir, "best_model_metrics.csv"))

# Save Best Model Object
joblib.dump(models[best_model_name], os.path.join(best_dir, "FINAL_BEST_MODEL.pkl"))

print("\nBest model metrics saved in 'best_stacked_outputs/'")
