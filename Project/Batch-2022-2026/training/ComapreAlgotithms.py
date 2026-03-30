import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder, PowerTransformer
from sklearn.metrics import mean_squared_error,accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import skew
import numpy


#Ensemble Technique
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def compareAlgorithms():
    train_df = pd.read_csv("../data/dataset_sdn.csv")
    train_df["rx_kbps"].fillna(train_df["rx_kbps"].mean())
    train_df["tot_kbps"].fillna(train_df["tot_kbps"].mean())

    remove_cols = ["dt", "tx_kbps", "pktperflow", "pktrate"]
    train_df.drop(remove_cols, axis=1, inplace=True)

    train_df = train_df.apply(LabelEncoder().fit_transform)

    X = train_df.drop(columns='label')
    y = train_df['label']

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_params = {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    skf = RepeatedStratifiedKFold(n_splits=5)

    RF = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                            param_distributions=rf_params, cv=skf, n_iter=2, n_jobs=2)

    RF_model = RF.fit(X_train, y_train)
    RF_pred = RF_model.predict(X_test)
    accuracy_score(y_test, RF_pred)

    evaluate_classification(RF_model, "Random Forest", X_train, X_test, y_train, y_test)







def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_accuracy = accuracy_score(y_train, model.predict(np.array(X_train)))
    test_accuracy = accuracy_score(y_test, model.predict(np.array(X_test)))
    # val_accuracy = accuracy_score(y_val, model.predict(np.array(X_test)))
    train_precision = precision_score(y_train, model.predict(np.array(X_train)))
    test_precision = precision_score(y_test, model.predict(np.array(X_test)))
    train_recall = recall_score(y_train, model.predict(np.array(X_train)))
    test_recall = recall_score(y_test, model.predict(np.array(X_test)))

    classification_evals[name] = {
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Train Precision': train_precision,
        'Test Precision': test_precision,
        'Train Recall': train_recall,
        'Test Recall': test_recall
    }

    print("Training Accuracy " + str(name) + ": {:.2f}".format(train_accuracy * 100))
    print("Test Accuracy " + str(name) + ": {:.2f}".format(test_accuracy * 100))
    print("Training Precision " + str(name) + ": {:.2f}".format(train_precision * 100))
    print("Test Precision " + str(name) + ": {:.2f}".format(test_precision * 100))
    print("Training Recall " + str(name) + ": {:.2f}".format(train_recall * 100))
    print("Test Recall " + str(name) + ": {:.2f}".format(test_recall * 100))

    # Plot the confusion matrix
    actual = y_test
    predicted = model.predict(np.array(X_test))
    cm = confusion_matrix(actual, predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    cm_display.plot(ax=ax)