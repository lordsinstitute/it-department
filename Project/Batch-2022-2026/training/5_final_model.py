# ==========================================
# ELECTRICITY THEFT DETECTION - CNN1D MODEL
# ==========================================

import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# SETTINGS
# ----------------------------
tf.random.set_seed(1234)

EPOCHS = 20
TEST_SIZE = 0.1

perf_dir = "../static/dl_performance"
save_dir = "../saved_models"

os.makedirs(perf_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)


# =============================
# READ DATA
# =============================
def read_data():

    print("Reading dataset...")

    df = pd.read_csv("../dataset/preprocessedR.csv")

    # Separate CONS_NO
    cons_no = df["CONS_NO"]

    # Target
    y = df["FLAG"]

    # Features
    X = df.drop(["FLAG", "CONS_NO"], axis=1)

    # Convert date columns to datetime (important for correct ordering)
    X.columns = pd.to_datetime(X.columns)

    # Sort columns chronologically
    X = X.sort_index(axis=1)

    print("Dataset shape:", X.shape)

    # Train Test Split (IMPORTANT: include cons_no)
    X_train, X_test, y_train, y_test, cons_train, cons_test = train_test_split(
        X, y, cons_no,
        test_size=TEST_SIZE,
        random_state=0,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, cons_test


# =============================
# SAVE PERFORMANCE
# =============================
def save_performance(y_test, prediction):

    acc = accuracy_score(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)

    print("Accuracy:", acc)
    print("AUC:", auc)

    # Save classification report
    report = classification_report(y_test, prediction)

    with open(os.path.join(perf_dir, "CNN1D_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
        f.write(report)

    # Save confusion matrix image
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("CNN1D - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(perf_dir, "CNN1D_confusion_matrix.png"))
    plt.close()


# =============================
# SAVE TRUE POSITIVES / NEGATIVES
# =============================
def save_correct_predictions(cons_test, y_test, prediction):

    results_df = pd.DataFrame({
        "CONS_NO": cons_test.values,
        "Actual": y_test.values,
        "Predicted": prediction.flatten()
    })

    true_positive = results_df[
        (results_df["Actual"] == 1) &
        (results_df["Predicted"] == 1)
    ].head(5)

    true_negative = results_df[
        (results_df["Actual"] == 0) &
        (results_df["Predicted"] == 0)
    ].head(5)

    final_df = pd.concat([true_positive, true_negative])

    final_df.to_csv(
        os.path.join(perf_dir, "CNN1D_5_correct_predictions_each_class.csv"),
        index=False
    )

    print("Saved 5 True Positives and 5 True Negatives.")


# =============================
# CNN1D MODEL
# =============================
def train_cnn1d():

    X_train, X_test, y_train, y_test, cons_test = read_data()

    # 🔥 SAVE FEATURE COLUMN ORDER (BEFORE reshaping)
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns,
                os.path.join(save_dir, "cnn1d_feature_columns.pkl"))

    print("Feature columns saved.")

    # Convert to numpy and reshape for CNN1D
    X_train_np = X_train.to_numpy().reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test_np = X_test.to_numpy().reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    # Build CNN1D model
    model = Sequential([
        Conv1D(64, kernel_size=7, activation='relu',
               input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print("Training CNN1D model...")

    history = model.fit(
        X_train_np,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        verbose=1
    )

    # Save model
    model.save(os.path.join(save_dir, "cnn1d_model.h5"))
    print("CNN1D model saved.")

    # Predictions
    prediction = (model.predict(X_test_np) > 0.5).astype(int)

    # Save evaluation
    save_performance(y_test, prediction)

    # Save 5 correct predictions
    save_correct_predictions(cons_test, y_test, prediction)

    print("Deep Learning training completed successfully.")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    train_cnn1d()
