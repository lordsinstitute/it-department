import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------------- LOAD DATA ----------------
df = pd.read_csv("../RTA_Dataset.csv")

# Drop leakage / irrelevant columns
DROP_COLS = [
    'Time', 'Day_of_week', 'Casualty_severity',
    'Sex_of_driver', 'Sex_of_casualty',
    'Educational_level', 'Fitness_of_casuality',
    'Owner_of_vehicle', 'Vehicle_driver_relation'
]

df.drop(columns=DROP_COLS, inplace=True)
df.to_csv("clean_df.csv")

# Target
y = df['Accident_severity']
X = df.drop('Accident_severity', axis=1)

# ---------------- FEATURE TYPES ----------------
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

# ---------------- PREPROCESSING ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# ---------------- MODEL PIPELINE ----------------
pipeline = ImbPipeline(steps=[
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

# ---------------- HYPERPARAMETERS ----------------
param_grid = {
    'model__n_estimators': [200, 300],
    'model__max_depth': [None, 20, 40],
    'model__min_samples_split': [2, 5]
}

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = grid.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
joblib.dump(grid.best_estimator_, "../AccidentSeverityPipeline.pkl")

print("✅ Model trained and saved successfully")
