import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from xgboost import XGBClassifier
import joblib

# ======================================
# 1️⃣ Create Output Folder
# ======================================

output_dir = "xgb_outputs"
os.makedirs(output_dir, exist_ok=True)

# ======================================
# 2️⃣ Load Data
# ======================================

df = pd.read_csv("engine_data.csv")

df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

X = df.drop("Engine Condition", axis=1)
y = df["Engine Condition"]

if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ======================================
# 3️⃣ Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================
# 4️⃣ XGBoost Model + Hyperparameter Tuning
# ======================================

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ======================================
# 5️⃣ Evaluate
# ======================================

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

accuracy = best_model.score(X_test, y_test)

print("\n🔥 XGBoost Test Accuracy:", accuracy)

# Save accuracy
with open(os.path.join(output_dir, "test_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy}")

# ======================================
# 6️⃣ Classification Report
# ======================================

report = classification_report(y_test, y_pred)

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

print("\nClassification Report:\n", report)

# ======================================
# 7️⃣ Confusion Matrix
# ======================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# ======================================
# 8️⃣ ROC Curve
# ======================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# ======================================
# 9️⃣ Feature Importance
# ======================================

plt.figure(figsize=(10,6))
sns.barplot(x=best_model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# ======================================
# 🔟 Save Model
# ======================================

joblib.dump(best_model, os.path.join(output_dir, "engine_condition_xgb.pkl"))

print("\n✅ All XGBoost outputs saved inside 'xgb_outputs/' folder.")
