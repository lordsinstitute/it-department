import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import warnings
from sklearn.metrics import classification_report,confusion_matrix

from pprint import pprint
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import RandomOverSampler

# Suppress all warnings
warnings.filterwarnings("ignore")

# ===============================
# Paths
# ===============================
DATA_PATH = "parkinsons.data"
STATIC_DIR = "static/vis"
SCALER_PATH = "parkinson_scaler.pkl"

os.makedirs(STATIC_DIR, exist_ok=True)


def createModel():
    # ===============================
    # Load Dataset
    # ===============================
    df = pd.read_csv(DATA_PATH)

    df_copy = df
    X = df_copy.drop(['name', 'status'], axis=1)
    y = df_copy["status"]

    Sample = RandomOverSampler(sampling_strategy=0.6)
    X_sam, Y_sam = Sample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_sam, Y_sam, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the range of k values for neighbors
    neighbors_range = [1, 3, 5, 7, 9, 11, 13, 15, 20]  # Example range, adjust as needed

    # Initialize lists to store training and testing accuracy
    train_accuracy = []
    test_accuracy = []

    # Iterate over different values of k
    for k in neighbors_range:
        # Create K Neighbors Classifier with current k value
        best_knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier on the training data
        best_knn_classifier.fit(X_train_scaled, y_train)

        # Calculate training accuracy and append to list
        train_accuracy.append(best_knn_classifier.score(X_train_scaled, y_train))

        # Calculate testing accuracy and append to list
        test_accuracy.append(best_knn_classifier.score(X_test_scaled, y_test))

    # Plot the training and testing accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors_range, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(neighbors_range, test_accuracy, label='Testing Accuracy', marker='o')
    plt.title('KNN: Training and Testing Accuracy vs. Number of Neighbors')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(neighbors_range)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{STATIC_DIR}/KNN_acc.jpg")

    model_pkl_file = "parkinson_classifier_model.pkl"

    with open(model_pkl_file, 'wb') as file:
        pickle.dump(best_knn_classifier, file)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Model and scaler saved successfully")

#createModel()