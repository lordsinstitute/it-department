# ============================================
# Feature Importance using RandomForest
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ============================================
# 1. Load Dataset
# ============================================
df = pd.read_csv("RTA_Dataset.csv")

TARGET = "Accident_severity"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ============================================
# 2. Handle Missing Values
# ============================================
for col in X.columns:
    if X[col].dtype == "object":
        X[col].fillna(X[col].mode()[0], inplace=True)
    else:
        X[col].fillna(X[col].median(), inplace=True)

# ============================================
# 3. Feature Types
# ============================================
categorical_features = X.select_dtypes(include="object").columns.tolist()
numeric_features = X.select_dtypes(exclude="object").columns.tolist()

# ============================================
# 4. Preprocessing
# ============================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# ============================================
# 5. Model
# ============================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", rf)
    ]
)

# ============================================
# 6. Train/Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================
# 7. Train Model
# ============================================
pipeline.fit(X_train, y_train)

# ============================================
# 8. Evaluation
# ============================================
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1 Score:", f1_score(y_test, y_pred, average="macro"))

# ============================================
# 9. Extract Encoded Feature Names
# ============================================
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
encoded_cat_features = ohe.get_feature_names_out(categorical_features)

all_features = np.concatenate([encoded_cat_features, numeric_features])

# ============================================
# 10. Get Feature Importances
# ============================================
importances = pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "encoded_feature": all_features,
    "importance": importances
})

# ============================================
# 11. Map Encoded → Original Feature
# ============================================
def map_original(feature):
    for col in categorical_features:
        if feature.startswith(col + "_"):
            return col
    return feature

importance_df["original_feature"] = importance_df["encoded_feature"].apply(map_original)

# ============================================
# 12. Aggregate Importance
# ============================================
final_importance = (
    importance_df
    .groupby("original_feature")["importance"]
    .sum()
    .reset_index()
)

final_importance["importance_percent"] = (
    final_importance["importance"] / final_importance["importance"].sum()
) * 100

final_importance = final_importance.sort_values(
    by="importance_percent",
    ascending=False
)

print("\nFeature Importance (%):\n")
print(final_importance)

# ============================================
# 13. Plot
# ============================================
plt.figure(figsize=(10, 6))
plt.barh(
    final_importance["original_feature"][:15],
    final_importance["importance_percent"][:15]
)
plt.gca().invert_yaxis()
plt.xlabel("Importance (%)")
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.show()
