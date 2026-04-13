# Import libraries for numerical computations
import numpy as np
import pandas as pd

# Import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries for data preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

#Import libraries for model evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
)

# Import libraries for machine learning
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Suppress warnings (use with caution)
import warnings
warnings.filterwarnings("ignore")

def plot_confusion_matrix(y_true, y_pred, class_names = None, cmap = "Blues", title = ""):
    """
    Plots a confusion matrix for classification tasks.

    Args:
      y_true (array-like): True labels for the data.
      y_pred (array-like): Predicted labels for the data.
      class_names (list, optional): List of class names for the labels.
      cmap (str, optional): Colormap to use for the heatmap. Defaults to 'Blues'.
      title (str, optional): Title for the confusion matrix plot. Defaults to an empty string.

    Returns:
      None
    """

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
      class_names = np.unique(y_true)

    plt.figure(figsize = (8, 6))
    sns.heatmap(
      cm,
      annot = True,
      fmt = 'd',
      cmap = cmap,
      xticklabels = class_names,
      yticklabels = class_names
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"../static/evaluation/confusion_matrix_{model_name}.jpg")


def pipeline_classification(pipelines):
    """
    Performs classification using cross-validation, evaluates different models,
    and makes predictions for each model on the test set.

    Args:
        pipelines (list): List of tuples containing model names and pipeline objects.

    Returns:
        pandas.DataFrame: A DataFrame containing model names, mean accuracy, standard deviation,
                          and dictionaries with test set predictions and probabilities.
    """

    # Initialize lists to store results
    cv_results = []
    model_names = []
    model_predictions = {}
    model_probabilities = {}

    # Perform cross-validation and store results
    for name, pipe in pipelines:
        pipe.fit(X_train, y_train)

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X_train, y_train, cv=kfold,
                                 scoring='accuracy', n_jobs=-1)
        cv_results.append(scores)
        model_names.append(name)

        # Predictions and probabilities
        model_predictions[name] = pipe.predict(X_test)
        model_probabilities[name] = pipe.predict_proba(X_test)

    # Build DataFrame with mean and std
    results_df = pd.DataFrame({
        'Model': model_names,
        'Mean Accuracy': [scores.mean() for scores in cv_results],
        'Std Accuracy': [scores.std() for scores in cv_results]
    })

    # Plot the results (mean accuracies)
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, results_df['Mean Accuracy'],
            yerr=results_df['Std Accuracy'], capsize=5)
    plt.title('Algorithm Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.savefig("../static/evaluation/comp_alg.jpg")
    plt.close()

    return results_df, model_predictions, model_probabilities


def get_model_scores(models, predictions, y_test, average = "None"):
    """
    Calculates and returns precision, recall, and F1 scores for each model.

    Args:
      models: A list of trained machine learning models.
      predictions: A list of predictions for each model, corresponding to the models list.
      y_test: True labels for the test set.
      average (str, optional): Averaging type for metrics.

    Returns:
      A Pandas DataFrame containing the model scores.
    """

    scores = [{
      'Model': model_name,
      'Accuracy': round(accuracy_score(y_test, y_pred), 3),
      'Precision': round(precision_score(y_test, y_pred, average = average), 3),
      'Recall': round(recall_score(y_test, y_pred, average = average), 3),
      'F1 Score': round(f1_score(y_test, y_pred, average = average), 3)
    } for model_name, y_pred in zip(models, predictions)]

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv("../static/evaluation/model_scores.csv")
    return scores_df


df = pd.read_csv("../data/Mental Health Dataset.csv")
df.dropna(inplace = True)

df.drop_duplicates(inplace = True)

# We are not going to use the Timestamp column in our analysis
df.drop(columns = "Timestamp", inplace = True)

# Create a LabelEncoder object
le = LabelEncoder()

# Apply LabelEncoder to each column
encoded_df = df.apply(le.fit_transform)

X = encoded_df.drop("Mood_Swings", axis = 1)
y = encoded_df["Mood_Swings"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Define the list of pipelines
pipelines = [
    ('DT', Pipeline([('scaler', MinMaxScaler()), ('DT', DecisionTreeClassifier())])),
    ('RF', Pipeline([('scaler', MinMaxScaler()), ('RF', RandomForestClassifier())])),
    ('XGB', Pipeline([('XGB', XGBClassifier())])),
  ]

results_df, model_predictions, model_probabilities = pipeline_classification(pipelines)

# Extract model names
models = results_df.loc[:, "Model"]

# Extract model predictions
predictions = list(model_predictions.values())

# Define labels for the class outputs
labels = df["Mood_Swings"].unique()[::-1]

# Define a dictionary mapping model names to colormaps
cmap_dict = {'DT': 'Blues', 'RF': 'Oranges', 'XGB': 'YlOrRd'}

for model_name, y_pred in zip(models, predictions):
  plot_confusion_matrix(
      y_test,
      y_pred,
      class_names = labels,
      cmap = cmap_dict.get(model_name),
      title = model_name
  )

scores = get_model_scores(models, predictions, y_test, average = "macro")

print(scores)



