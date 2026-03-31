import pandas as pd
import joblib
import os

# =============================
# PATH TO TRAINING DATASET
# =============================
DATA_PATH = "../dataset/preprocessedR.csv"   # Adjust if needed
SAVE_DIR = "../saved_models"

# =============================
# LOAD DATA
# =============================
rawData = pd.read_csv(DATA_PATH)

# Drop target and ID columns (same as training)
X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

# Apply same column processing as training
X.columns = pd.to_datetime(X.columns)
X = X.reindex(X.columns, axis=1)

# =============================
# SAVE FEATURE COLUMNS
# =============================
os.makedirs(SAVE_DIR, exist_ok=True)

feature_columns = X.columns.tolist()

joblib.dump(feature_columns,
            os.path.join(SAVE_DIR, "cnn1d_feature_columns.pkl"))

print("Feature columns saved successfully!")
print("Total features:", len(feature_columns))
