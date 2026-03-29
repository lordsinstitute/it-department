import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from xgboost import XGBClassifier
import joblib


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7,5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"../static/final_model/confusion_matrix_{model_name}.jpg")
    plt.close()

df = pd.read_csv("../data/Mental Health Dataset.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.drop(columns=["Timestamp"], inplace=True)

TARGET = "Mood_Swings"

X = df.drop(TARGET, axis=1)
y = df[TARGET]
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "../static/final_model/feature_columns.pkl")

label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",   # 🔥 MUCH faster
    n_jobs=-1,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    xgb_model,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    n_jobs=-1,
    return_train_score=False
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

scores = {
    "Model": "XGBoost",
    "Accuracy": round(accuracy_score(y_test, y_pred), 3),
    "Precision": round(precision_score(y_test, y_pred, average="macro"), 3),
    "Recall": round(recall_score(y_test, y_pred, average="macro"), 3),
    "F1 Score": round(f1_score(y_test, y_pred, average="macro"), 3)
}

scores_df = pd.DataFrame([scores])
scores_df.to_csv("../static/final_model/model_scores.csv", index=False)

with open("../static/final_model/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

class_names = y_encoder.classes_
plot_confusion_matrix(y_test, y_pred, class_names, "XGB")

joblib.dump(xgb_model, "../static/final_model/best_model.pkl")

joblib.dump(label_encoders, "../static/final_model/label_encoders.pkl")
joblib.dump(y_encoder, "../static/final_model/target_encoder.pkl")
