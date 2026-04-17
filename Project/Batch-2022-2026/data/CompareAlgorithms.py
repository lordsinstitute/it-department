import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
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

os.makedirs(STATIC_DIR, exist_ok=True)


def compareAlgorithms():
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

    # Define models to test
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Decision Trees": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K Neighbors": KNeighborsClassifier(),
        "XGBoost": XGBClassifier()
    }

    #  Metrics
    training_accuracy = {}
    cross_validation_score_train = {}
    precision_train = {}
    recall_train = {}
    f1_train = {}

    # Train, test, and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)

        # Training Accuracy Score
        training_accuracy[name] = model.score(X_train_scaled, y_train)

        # Cross Validation Score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cross_validation_score_train[name] = cv_scores.mean()

        # Precision Score
        y_pred = model.predict(X_train_scaled)
        precision_train[name] = precision_score(y_train, y_pred)

        # Recall Score
        recall_train[name] = recall_score(y_train, y_pred)

        # F1-Score Score
        f1_train[name] = f1_score(y_train, y_pred)

    # Print evaluation metrics for each model
    print("*" * 50)
    print("Training Set")
    print("*" * 50)
    print("")

    for name in models.keys():
        print(f"Model: {name}")
        print(f"Training Accuracy Score: {training_accuracy[name]}")
        print(f"Cross Validation Score: {cross_validation_score_train[name]}")
        print(f"Precision Score: {precision_train[name]}")
        print(f"Recall Score: {recall_train[name]}")
        print(f"F1-Score Score: {f1_train[name]}")
        print("-" * 50)

    metric_names = ["Training Accuracy", "Cross Validation", "Precision", "Recall", "F1-Score"]
    scores = [training_accuracy, cross_validation_score_train, precision_train, recall_train, f1_train]

    # Define colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # Generate colors from the 'tab10' colormap

    # Plot and save each metric individually
    for i, metric in enumerate(metric_names):
        plt.figure(figsize=(6, 4))  # Create a new figure for each metric

        for j, model_name in enumerate(models.keys()):
            plt.bar(model_name, scores[i][model_name], color=colors[j], label=model_name)

        plt.title(metric)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()

        # Save each plot as a JPG file
        plt.savefig(f"{STATIC_DIR}/train_{metric.replace(' ', '_')}.jpg", format="jpg")
        plt.close()  # Close the figure to avoid overlap

    # Define models to test
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Decision Trees": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K Neighbors": KNeighborsClassifier(),
        "XGBoost": XGBClassifier()
    }

    #  Metrics
    testing_accuracy = {}
    cross_validation_score_test = {}
    precision_test = {}
    recall_test = {}
    f1_test = {}

    # Train, test, and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_test_scaled, y_test)

        # Testing Accuracy Score
        testing_accuracy[name] = model.score(X_test_scaled, y_test)

        # Cross Validation Score
        cv_scores_test = cross_val_score(model, X_test_scaled, y_test, cv=5)
        cross_validation_score_test[name] = cv_scores_test.mean()

        # Precision Score
        y_pred = model.predict(X_test_scaled)
        precision_test[name] = precision_score(y_test, y_pred)

        # Recall Score
        recall_test[name] = recall_score(y_test, y_pred)

        # F1-Score Score
        f1_test[name] = f1_score(y_test, y_pred)

    # Print evaluation metrics for each model
    print("*" * 50)
    print("Testing Set")
    print("*" * 50)
    print("")
    # Print evaluation metrics for each model
    for name in models.keys():
        print(f"Model: {name}")
        print(f"Testing Accuracy Score: {testing_accuracy[name]}")
        print(f"Cross Validation Score: {cross_validation_score_test[name]}")
        print(f"Precision Score: {precision_test[name]}")
        print(f"Recall Score: {recall_test[name]}")
        print(f"F1-Score Score: {f1_test[name]}")
        print("-" * 50)

    metric_names = ["Testing Accuracy", "Cross Validation", "Precision", "Recall", "F1-Score"]
    scores = [testing_accuracy, cross_validation_score_test, precision_test, recall_test, f1_test]

    # Define colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # Generate colors from the 'tab10' colormap

    # Define colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # Generate colors from the 'tab10' colormap

    # Plot and save each metric individually
    for i, metric in enumerate(metric_names):
        plt.figure(figsize=(6, 4))  # Create a new figure for each metric

        for j, model_name in enumerate(models.keys()):
            plt.bar(model_name, scores[i][model_name], color=colors[j], label=model_name)

        plt.title(metric)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()

        # Save each plot as a JPG file
        plt.savefig(f"{STATIC_DIR}/test_{metric.replace(' ', '_')}.jpg", format="jpg")
        plt.close()  # Close the figure to avoid overlap

    chosen_models = {
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "K Neighbors": KNeighborsClassifier()
    }

    # Iterate over each model and plot confusion matrix
    for name, model in chosen_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        plot_confusion_matrix(y_test, y_pred, name)

    rf_classifier = RandomForestClassifier()

    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform Grid Search to find the best combination of hyperparameters
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best estimator with the tuned hyperparameters
    best_rf_classifier = grid_search.best_estimator_

    # Make predictions on the testing set
    y_pred = best_rf_classifier.predict(X_test_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    cv_scores_test = cross_val_score(best_rf_classifier, X_test_scaled, y_test, cv=5)
    cross_validation_score = cv_scores_test.mean()

    # Print the evaluation metrics
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("cross_validation_score:", cross_validation_score)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['Healthy', 'Parkinsons'],
                yticklabels=['Healthy', 'Parkinsons'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    #plt.show()



# Define function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Healthy', 'Parkinsons'],
                yticklabels=['Healthy', 'Parkinsons'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"{STATIC_DIR}/cnf_{model_name}.jpg")

#compareAlgorithms()