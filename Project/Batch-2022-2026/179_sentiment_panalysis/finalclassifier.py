import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib
import os

def createModel():
    print("Loading dataset...")
    dataset = pd.read_csv('data/generic_sentiment_dataset_50k.csv')
    features = dataset.iloc[:, 1].values
    labels = dataset.iloc[:, 2].values

    print("Preprocessing text...")
    processed_features = []
    for sentence in features:
        processed_feature = re.sub(r'\W', ' ', str(sentence))
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        processed_feature = re.sub(r'^b\s+', '', processed_feature)
        processed_feature = processed_feature.lower()
        processed_features.append(processed_feature)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=1500, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(processed_features).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        processed_features, labels, test_size=0.2, random_state=0
    )

    print("Training Random Forest (this may take 1-3 minutes)...")
    rf_classifier = RandomForestClassifier(n_estimators=80, random_state=0)
    rf_classifier.fit(X_train, y_train)

    print("Evaluating...")
    rf_predictions = rf_classifier.predict(X_test)

    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, rf_predictions))
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, rf_predictions))
    print("\n--- ACCURACY ---")
    print(accuracy_score(y_test, rf_predictions))

    os.makedirs(os.path.join('static', 'vis'), exist_ok=True)

    cm = confusion_matrix(y_test, rf_predictions)
    num_classes = len(np.unique(labels))
    class_names = list(range(num_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion Matrix - Random Forest', y=1.1)
    plt.ylabel('Actuals')
    plt.xlabel('Predicted')
    plt.savefig('static/vis/rf_cnf.jpg')
    plt.clf()
    plt.close()

    report = classification_report(y_test, rf_predictions)
    fig = plt.figure(figsize=(10, 5))
    plt.text(0.01, 0.5, report, fontsize=10, family="monospace",
             verticalalignment='center')
    plt.axis('off')
    plt.title("Random Forest Classification Report")
    plt.savefig('static/vis/rf_clf.jpg', bbox_inches='tight')
    plt.close(fig)

    print("Saving model and vectorizer...")
    joblib.dump(rf_classifier, 'finalized_model.sav')
    joblib.dump(vectorizer, 'vectorizer.sav')
    print("Done! finalized_model.sav and vectorizer.sav created.")

createModel()