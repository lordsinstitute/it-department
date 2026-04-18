# =============================
# Import Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# =============================
# GridSearch Function
# =============================
def bestParams(model, param, xtrain, ytrain):
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=42
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    res = grid.fit(xtrain, ytrain)

    print("Best Parameters:", res.best_params_)
    print("Best CV F1 Score:", res.best_score_)

    return res.best_params_, res.best_score_


# =============================
# Model Creation
# =============================
def create_model():

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv("RTA_Dataset.csv")

    # -----------------------------
    # Convert to Binary Target
    # -----------------------------
    df['Accident_severity'] = df['Accident_severity'].astype(str)

    df['Accident_severity'] = df['Accident_severity'].replace({
        'Slight Injury': 'Non-Severe',
        'Serious Injury': 'Severe',
        'Fatal injury': 'Severe',
        '0': 'Non-Severe',
        '1': 'Severe',
        '2': 'Severe'
    })

    print("Final Target Classes:", df['Accident_severity'].unique())

    # -----------------------------
    # Drop Irrelevant Columns
    # -----------------------------
    drop_cols = [
        'Vehicle_driver_relation', 'Work_of_casuality',
        'Fitness_of_casuality', 'Day_of_week',
        'Casualty_severity', 'Time', 'Sex_of_driver',
        'Educational_level', 'Defect_of_vehicle',
        'Owner_of_vehicle', 'Service_year_of_vehicle',
        'Road_surface_type', 'Sex_of_casualty'
    ]

    df.drop(columns=drop_cols, inplace=True)

    # -----------------------------
    # Handle Missing Values
    # -----------------------------
    fill_cols = [
        'Driving_experience', 'Age_band_of_driver',
        'Type_of_vehicle', 'Area_accident_occured',
        'Road_allignment', 'Type_of_collision',
        'Vehicle_movement', 'Lanes_or_Medians',
        'Types_of_Junction'
    ]

    for col in fill_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # -----------------------------
    # Label Encoding
    # -----------------------------
    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    joblib.dump(encoders, "label_encoder.pkl")

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X = df.drop("Accident_severity", axis=1)
    y = df["Accident_severity"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Compute Class Weights
    # -----------------------------
    classes = np.unique(ytrain)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=ytrain
    )
    class_weights = dict(zip(classes, weights))
    print("Class Weights:", class_weights)

    # -----------------------------
    # Grid Search
    # -----------------------------
    base_model = RandomForestClassifier(
        random_state=42,
        class_weight=class_weights
    )

    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [5, 10]
    }

    best_param, best_score = bestParams(
        base_model, params, xtrain, ytrain
    )

    # -----------------------------
    # Final Model
    # -----------------------------
    model = RandomForestClassifier(
        **best_param,
        random_state=42,
        class_weight=class_weights
    )

    model.fit(xtrain, ytrain)

    # -----------------------------
    # Evaluation
    # -----------------------------
    ypred = model.predict(xtest)

    print("\nAccuracy:", accuracy_score(ytest, ypred))

    print(
        classification_report(
            ytest,
            ypred,
            labels=[0, 1],
            target_names=["Non-Severe", "Severe"]
        )
    )

    cm = confusion_matrix(ytest, ypred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=["Non-Severe", "Severe"],
        yticklabels=["Non-Severe", "Severe"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/pimg/cm.jpg")
    plt.close()

    results = xtest.copy()
    results['Actual'] = ytest.values
    results['Predicted'] = ypred

    true_severe = results[
        (results['Actual'] == 1) &
        (results['Predicted'] == 1)
        ]
    true_severe.to_csv('true_severe_samples.csv', index=False)

    print("Number of TRUE Severe cases:", len(true_severe))
    true_severe.head(10)

    true_nonsevere=results[
        (results['Actual'] == 0) &
        (results['Predicted'] == 0)
        ]
    true_nonsevere.to_csv('true_nonsevere_samples.csv', index=False)

    print("Number of TRUE NonSevere cases:", len(true_nonsevere))
    true_nonsevere.head(10)

    # -----------------------------
    # Save Model
    # -----------------------------
    joblib.dump(model, "Accident_model.pkl")

    print("\n✅ Model saved successfully!")

    return "Binary Accident Severity Model Created", round(best_score * 100, 2)


# =============================
# Run
# =============================
create_model()
