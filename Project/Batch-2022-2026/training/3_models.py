import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, \
    confusion_matrix, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Conv2D

# -----------------------------
# Create folders
# -----------------------------
vis_dir = "../static/vis"
perf_dir = "../static/performance"
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(perf_dir, exist_ok=True)

tf.random.set_seed(1234)

epochs_number = 1
test_set_size = 0.1
oversampling_flag = 0
oversampling_percentage = 0.2


# =============================
# READ DATA + DISTRIBUTION PLOT
# =============================
def read_data():
    rawData = pd.read_csv('../dataset/preprocessedR.csv')

    y = rawData[['FLAG']]
    X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

    print('Normal Consumers:     ', y[y['FLAG'] == 0].count()[0])
    print('Consumers with Fraud: ', y[y['FLAG'] == 1].count()[0])
    print('Total Consumers:      ', y.shape[0])

    # -------- Distribution Plot --------
    plt.figure()
    y['FLAG'].value_counts().plot(kind='bar')
    plt.title("Distribution of Consumers (Fraud vs Non-Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "distribution_plot.png"))
    plt.close()

    # Convert columns to datetime
    X.columns = pd.to_datetime(X.columns)
    X = X.reindex(X.columns, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y['FLAG'], test_size=test_set_size, random_state=0)

    # -------- Oversampling --------
    if oversampling_flag == 1:
        over = SMOTE(sampling_strategy=oversampling_percentage, random_state=0)
        X_train, y_train = over.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


# =============================
# SAVE RESULTS
# =============================
def results(y_test, prediction, model_name="model"):
    acc = 100 * accuracy_score(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction, squared=False)
    mae = mean_absolute_error(y_test, prediction)
    auc = 100 * roc_auc_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)

    print("Accuracy:", acc)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("AUC:", auc)
    print(cm)

    # -------- Save Classification Report --------
    report = classification_report(y_test, prediction)

    with open(os.path.join(perf_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.2f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"AUC: {auc:.2f}\n\n")
        f.write(report)

    # -------- Save Confusion Matrix --------
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(perf_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()


# =============================
# MODELS
# =============================

def LR(X_train, X_test, y_train, y_test):
    print("Logistic Regression")
    model = LogisticRegression(C=1000, max_iter=1000, solver='newton-cg')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Logistic_Regression")


def DT(X_train, X_test, y_train, y_test):
    print("Decision Tree")
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Decision_Tree")


def RF(X_train, X_test, y_train, y_test):
    print("Random Forest")
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Random_Forest")


def SVM(X_train, X_test, y_train, y_test):
    print("SVM")
    model = SVC(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "SVM")


def ANN(X_train, X_test, y_train, y_test):
    print("Artificial Neural Network")

    model = Sequential()
    model.add(Dense(1000, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs_number, verbose=1)

    prediction = (model.predict(X_test) > 0.5).astype(int)
    results(y_test, prediction, "ANN")


# =============================
# MAIN
# =============================

X_train, X_test, y_train, y_test = read_data()

# Uncomment any model to test
#LR(X_train, X_test, y_train, y_test)
#DT(X_train, X_test, y_train, y_test)
#RF(X_train, X_test, y_train, y_test)
#SVM(X_train, X_test, y_train, y_test)
ANN(X_train, X_test, y_train, y_test)

print("All outputs saved successfully.")
