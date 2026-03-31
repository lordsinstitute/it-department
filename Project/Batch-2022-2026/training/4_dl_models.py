import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Conv2D
from tensorflow import keras
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, \
    confusion_matrix, precision_recall_fscore_support, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
import os

tf.random.set_seed(1234)

epochs_number = 20
test_set_size = 0.1
oversampling_flag = 0
oversampling_percentage = 0.2

vis_dir = "../static/vis"
perf_dir = "../static/dl_performance"
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(perf_dir, exist_ok=True)


# =============================
# READ DATA
# =============================
def read_data():
    rawData = pd.read_csv('../dataset/preprocessedR.csv')

    y = rawData[['FLAG']]
    X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

    X.columns = pd.to_datetime(X.columns)
    X = X.reindex(X.columns, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y['FLAG'], test_size=test_set_size, random_state=0)

    if oversampling_flag == 1:
        over = SMOTE(sampling_strategy=oversampling_percentage, random_state=0)
        X_train, y_train = over.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


# =============================
# SAVE FINAL RESULTS
# =============================
def save_final_results(y_test, prediction, model_name):

    acc = 100 * accuracy_score(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction, squared=False)
    mae = mean_absolute_error(y_test, prediction)
    auc = 100 * roc_auc_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)

    print(f"{model_name} Accuracy:", acc)

    # ---- Save Classification Report ----
    report = classification_report(y_test, prediction)

    with open(os.path.join(perf_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.2f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"AUC: {auc:.2f}\n\n")
        f.write(report)

    # ---- Save Confusion Matrix ----
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(perf_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()


# =============================
# SAVE EPOCH HISTORY
# =============================
def save_epoch_results(history, model_name):

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(perf_dir, f"{model_name}_epoch_results.csv"), index=False)

    # Plot Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{model_name} Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(perf_dir, f"{model_name}_accuracy_plot.png"))
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{model_name} Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(perf_dir, f"{model_name}_loss_plot.png"))
    plt.close()


# =============================
# ANN
# =============================
def ANN(X_train, X_test, y_train, y_test):
    print("Running ANN...")

    model = Sequential([
        Dense(1000, input_dim=X_train.shape[1], activation='relu'),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=epochs_number,
                        verbose=1)

    save_epoch_results(history, "ANN")

    prediction = (model.predict(X_test) > 0.5).astype(int)
    save_final_results(y_test, prediction, "ANN")


# =============================
# CNN1D
# =============================
def CNN1D(X_train, X_test, y_train, y_test):
    print("Running CNN1D...")

    X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        Conv1D(100, kernel_size=7, activation='relu',
               input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=epochs_number,
                        verbose=1)

    save_epoch_results(history, "CNN1D")

    prediction = (model.predict(X_test) > 0.5).astype(int)
    save_final_results(y_test, prediction, "CNN1D")

    # -------------------------------------
    # 🔥 SAVE TRAINED MODEL
    # -------------------------------------

    save_dir = "../saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Save model (modern format recommended)
    model.save(os.path.join(save_dir, "cnn1d_model.h5"))

    # Save feature column order
    joblib.dump(X_train.columns.tolist(),
                os.path.join(save_dir, "cnn1d_feature_columns.pkl"))

    print("CNN1D model and feature columns saved successfully!")



    # =============================
    # SAVE 5 TRUE POSITIVES & TRUE NEGATIVES
    # =============================

    y_test_np = y_test.to_numpy()

    results_df = pd.DataFrame({
        "CONS_NO": cons_test.values,
        "Actual": y_test_np,
        "Predicted": prediction
    })

    # True Positives (Actual=1, Predicted=1)
    true_positive = results_df[
        (results_df["Actual"] == 1) &
        (results_df["Predicted"] == 1)
        ].head(5)

    # True Negatives (Actual=0, Predicted=0)
    true_negative = results_df[
        (results_df["Actual"] == 0) &
        (results_df["Predicted"] == 0)
        ].head(5)

    final_tp_tn = pd.concat([true_positive, true_negative])

    # Save to CSV
    final_tp_tn.to_csv("../CNN1D_5_correct_predictions_each_class.csv",index=False)

    print("Saved 5 True Positives and 5 True Negatives successfully!")

    #return model


# =============================
# CNN2D
# =============================
def CNN2D(X_train, X_test, y_train, y_test):
    print("Running CNN2D...")

    # Convert to numpy
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    # -----------------------------
    # Add padding to make columns multiple of 7
    # -----------------------------
    remainder = X_train_np.shape[1] % 7
    if remainder != 0:
        padding = 7 - remainder
        X_train_np = np.hstack((X_train_np,
                                np.zeros((X_train_np.shape[0], padding))))
        X_test_np = np.hstack((X_test_np,
                               np.zeros((X_test_np.shape[0], padding))))

    # -----------------------------
    # Reshape dynamically
    # -----------------------------
    weeks = X_train_np.shape[1] // 7

    X_train_reshaped = X_train_np.reshape(X_train_np.shape[0], weeks, 7, 1)
    X_test_reshaped = X_test_np.reshape(X_test_np.shape[0], weeks, 7, 1)

    # -----------------------------
    # Model
    # -----------------------------
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=(weeks, 7, 1)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train_reshaped, y_train,
                        validation_split=0.1,
                        epochs=epochs_number,
                        verbose=1)

    save_epoch_results(history, "CNN2D")

    prediction = (model.predict(X_test_reshaped) > 0.5).astype(int)
    save_final_results(y_test, prediction, "CNN2D")



# =============================
# MAIN
# =============================
X_train, X_test, y_train, y_test = read_data()

#ANN(X_train, X_test, y_train, y_test)
CNN1D(X_train, X_test, y_train, y_test)
#CNN2D(X_train, X_test, y_train, y_test)

print("Deep Learning outputs saved successfully.")
