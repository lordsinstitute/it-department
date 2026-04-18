import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dummy dataset
data = {
    "feature": [10, 50, 100, 200, 300],
    "label": [0, 0, 1, 2, 2]
}

df = pd.DataFrame(data)

X = df[["feature"]]
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "ids_model.pkl")

print("Model trained and saved!")