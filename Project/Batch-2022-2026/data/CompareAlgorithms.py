import os
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

def compAlg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(BASE_DIR, "..")
    VIS_DIR = os.path.join(PROJECT_DIR, "static", "vis")

    # Ensure the output directory exists
    os.makedirs(VIS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(PROJECT_DIR, "fetal_health.csv"))

    # Split data into X and y
    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]
    # Split data into Train and Test sets
    RANDOM_STATE = 42
    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    numerical_cols = list(X_train.columns)
    numerical_cols.remove('histogram_tendency')

    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()

    for col in numerical_cols:
        min_max_scaler: MinMaxScaler = MinMaxScaler()
        scale = min_max_scaler.fit(X_train_norm[[col]])
        X_train_norm[col] = scale.fit_transform(X_train_norm[[col]])
        X_test_norm[col] = scale.transform(X_test_norm[[col]])

    X_train_stand = X_train.copy()
    X_test_stand = X_test.copy()

    for col in numerical_cols:
        scale = StandardScaler().fit(X_train_stand[[col]])
        X_train_stand[col] = scale.transform(X_train_stand[[col]])
        X_test_stand[col] = scale.transform(X_test_stand[[col]])

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_preds = rf.predict(X_test)
    print('Accuracy:', rf.score(X_test, y_test))
    print('F1:', f1_score(y_test, y_preds, average='macro'))
    print(confusion_matrix(y_test, y_preds))

    test = classification_report(y_test, y_preds, target_names=['Normal', 'Suspect', 'Pathologic'])

    # Normalized data
    rf.fit(X_train_norm, y_train)
    y_preds_norm = rf.predict(X_test_norm)
    print('Accuracy:', rf.score(X_test_norm, y_test))
    print('F1:', f1_score(y_test, y_preds_norm, average='macro'))
    print(confusion_matrix(y_test, y_preds_norm))

    # Standardized data
    rf.fit(X_train_stand, y_train)
    y_preds_stand = rf.predict(X_test_stand)
    print('Accuracy:', rf.score(X_test_stand, y_test))
    print('F1:', f1_score(y_test, y_preds_stand, average='macro'))
    print(confusion_matrix(y_test, y_preds_stand))

    #########################################################################
    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    print('Accuracy:', svm_model_linear.score(X_test, y_test))
    print('F1:', f1_score(y_test, svm_predictions, average='macro'))
    print(confusion_matrix(y_test, svm_predictions))

    #########################################################################
    gnb = GaussianNB().fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)

    print('Accuracy:', gnb.score(X_test, y_test))
    print('F1:', f1_score(y_test, gnb_predictions, average='macro'))
    print(confusion_matrix(y_test, gnb_predictions))

    ############################################################################
    SGDClf = SGDClassifier(max_iter=600,
                           tol=1e-3,
                           alpha=10 ** -5,
                           random_state=RANDOM_STATE)

    # Normalized Data
    SGDClf.fit(X_train_norm, y_train)
    y_preds = SGDClf.predict(X_test_norm)
    print('NORMALIZED')
    print('Accuracy:', SGDClf.score(X_test_norm, y_test))
    print('F1:', f1_score(y_test, y_preds, average='macro'))
    print(confusion_matrix(y_test, y_preds))

    # Standardized Data
    print()
    print('STANDARDIZED')
    SGDClf.fit(X_train_stand, y_train)
    y_preds = SGDClf.predict(X_test_stand)
    print('Accuracy:', SGDClf.score(X_test_stand, y_test))
    print('F1:', f1_score(y_test, y_preds, average='macro'))
    print(confusion_matrix(y_test, y_preds))

    # Put models in a dictionary
    models = {"Logistic Regression": LogisticRegression(max_iter=10000),
              "KNN": KNeighborsClassifier(),
              "Random Forest": RandomForestClassifier(),
              "Support Vector Machine": SVC(),
              "Gaussian Naive Bayes": GaussianNB(),
              "SGD": SGDClassifier()}

    model_scores, model_scores_f1 = fit_and_score(models, X_train, X_test, y_train, y_test, VIS_DIR)
    print(model_scores)
    print(model_scores_f1)

    # Plotting accuracies of all the models
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel("\nAccuracy %", fontsize=20)
    plt.xlabel("\nAlgorithms", fontsize=20)

    ax = sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), palette='tab10')
    for i, accuracy in enumerate(model_scores.values()):
        plt.text(i, accuracy + 2, str(accuracy), ha='center', va='bottom', fontsize=10)

    plt.title("Model Accuracy Comparison", fontsize=25)
    plt.savefig(os.path.join(VIS_DIR, 'AlgComp.jpg'))
    plt.clf()

    return model_scores


# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test, VIS_DIR):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    VIS_DIR : absolute path to save visualizations
    """
    # Make a dictionary to keep model scores
    model_scores = {}
    model_scores_f1 = {}

    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Make predictions
        y_preds = model.predict(X_test)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = round(model.score(X_test, y_test) * 100, 2)
        model_scores_f1[name] = f1_score(y_test, y_preds, average='macro')

        # Generate the confusion matrix
        cm = confusion_matrix(y_test, y_preds)

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap="Blues")

        plt.title(f"Confusion Matrix for {name}")
        plt.savefig(os.path.join(VIS_DIR, 'cnf_' + name + '.jpg'))
        plt.clf()

    return model_scores, model_scores_f1