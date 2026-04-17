import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("engine_data.csv")

X = df.drop("Engine Condition", axis=1)
y = df["Engine Condition"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

os.makedirs("static/EDA", exist_ok=True)
os.makedirs("static/Performance", exist_ok=True)
os.makedirs("models", exist_ok=True)

best_acc = 0
best_model = None

# EDA
for col in df.columns[:-1]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.savefig(f"static/EDA/{col}.png")
    plt.close()

# Training
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    acc = model.score(X_test, y_test)

    with open(f"static/Performance/{name}_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.savefig(f"static/Performance/{name}_cm.png")
    plt.close()

    if acc > best_acc:
        best_acc = acc
        best_model = model

# Save best model
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
