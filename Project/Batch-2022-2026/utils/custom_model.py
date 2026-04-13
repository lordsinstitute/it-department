from typing import List
import pandas as pd
from tqdm.notebook import tqdm_notebook
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_class_weight
import optuna

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score  # for Gini-mean
from sklearn.metrics import roc_curve

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

import pickle, joblib

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 2023


def load_checkpoint(model_path: str):
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
        return loaded_model


class CustomModelling:

    def __init__(
        self, model, x_train: np.array, y_train: np.array, x_test: np.array
    ) -> None:

        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def train(self, features: np.array, target: np.array):
        self.model.fit(features, target)
        return self.model

    def inference(self, x_test: np.array) -> np.array:
        preds = self.model.predict(x_test)
        return preds

    def stratifiedKCV(
        self,
        splits: int,
        seed: int,
        print_reports: bool = True,
        print_confusion: bool = True,
        isCatBoost: bool = False,
    ):
        skfolds = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
        (
            training_scores,
            validation_scores,
            precission_scores,
            recall_scores,
            f1_scores,
        ) = (list(), list(), list(), list(), list())
        test_folds_predictions = list()

        for train_idx, valid_idx in tqdm_notebook(
            skfolds.split(self.x_train, self.y_train),
            total=skfolds.get_n_splits(),
            desc="Stratified K-Fold",
            ascii=" ▖▘▝▗▚▞█",
        ):

            fold_x_train, fold_y_train = (
                self.x_train.iloc[train_idx],
                self.y_train[train_idx],
            )
            fold_x_valid, fold_y_valid = (
                self.x_train.iloc[valid_idx],
                self.y_train[valid_idx],
            )

            fold_model = self.model
            fold_model = self.train(fold_x_train, fold_y_train)
            with open("lgbm_model_best.pkl", "wb") as file:
                pickle.dump(fold_model, file)

            fold_train_predicted = self.inference(fold_x_train)
            fold_valid_predicted = self.inference(fold_x_valid)

            # inference on test data
            fold_test_predicted = self.inference(self.x_test)
            # store the predictions of each fold
            test_folds_predictions.append(fold_test_predicted)

            fold_train_accuracy, _, _, _ = self.compute_scores(
                fold_y_train, fold_train_predicted
            )
            (
                fold_valid_accuracy,
                fold_valid_precision,
                fold_valid_recall,
                fold_valid_f1,
            ) = self.compute_scores(fold_y_valid, fold_valid_predicted)

            training_scores.append(fold_train_accuracy)
            validation_scores.append(fold_valid_accuracy)
            precission_scores.append(fold_valid_precision)
            recall_scores.append(fold_valid_recall)
            f1_scores.append(fold_valid_f1)

        if print_reports:
            print(
                "\n------------------------------------------------------------------------"
            )
            print(f"Mean Accuracy on Training Data:", np.mean(training_scores))
            print(f"Mean Accuracy on Testing Data is:", np.mean(validation_scores))
            print(f"Mean Precision:", np.mean(precission_scores))
            print(f"Mean Recall:", np.mean(recall_scores))
            print(f"Mean F1 Score:", np.mean(f1_scores))

            # displaty classification reports after runnig on all test data
        if print_confusion:
            predictions = cross_val_predict(
                self.model, self.x_train, self.y_train, cv=skfolds
            )
            self.classification_reports(self.y_train, predictions,"LGBM with Hyperparameter Tuning and Stratified K fold Validation")
            self.confusion_matrices(self.y_train, predictions,"LGBM with Hyperparameter Tuning and Stratified K fold Validation")

        # create statified test predictions
        stratified_test_predictions = self.create_stratified_preds(
            test_folds_predictions, isCatBoost
        )

        return stratified_test_predictions

    def create_stratified_preds(
        self, folds_preds: List[np.array], isCatBoost: bool = False
    ) -> pd.DataFrame:
        # store each fold test predictions into a new df
        test_predictions_df = pd.DataFrame()
        for i in range(len(folds_preds)):
            if isCatBoost:
                test_predictions_df = pd.concat(
                    [test_predictions_df, pd.DataFrame(folds_preds[i])],
                    axis=1,
                    ignore_index=True,
                )
            else:
                test_predictions_df = pd.concat(
                    [test_predictions_df, pd.DataFrame(folds_preds[i].transpose())],
                    axis=1,
                    ignore_index=True,
                )

        # and use the pandas.mode function, which returns the most frequent value per row (i.e. per fold)
        stratified_test_predictions = test_predictions_df.mode(axis=1)[0]
        return stratified_test_predictions

    def compute_scores(self, gold_truths: np.array, predictions: np.array):
        accuracy = accuracy_score(gold_truths, predictions, normalize=True)
        precision = precision_score(gold_truths, predictions, average="weighted")
        recall = recall_score(gold_truths, predictions, average="weighted")
        f1 = f1_score(gold_truths, predictions, average="weighted")

        return accuracy, precision, recall, f1

    def classification_reports(self, gold_truths: np.array, predictions: np.array, algorithmName):

        print(f"Classification Report:")
        print(classification_report(gold_truths, predictions))
        clf_rpt=classification_report(gold_truths, predictions)
        plt.title(f"{algorithmName} Classification Report")
        fig = plt.figure(figsize=(10, 5))
        plt.text(0.1, 0.5, clf_rpt, fontsize=12, family="monospace")
        plt.axis('off')  # Remove axes

        # Save the figure
        plt.savefig("static/vis/clf_"+algorithmName+".jpg", bbox_inches='tight')
        plt.close(fig)

    def confusion_matrices(self, gold_truths: np.array, predictions: np.array, algorithmName):
        print(f"Confusion Matrix:")
        cm = confusion_matrix(gold_truths, predictions)
        plt.figure(figsize=(8, 4))
        plt.title(f"{algorithmName} Confusion Matrix")
        sns.heatmap(cm, annot=True, fmt="g", cmap="summer", xticklabels=True, yticklabels=True)
        plt.savefig("static/vis/cm_"+algorithmName+".jpg", bbox_inches='tight')
        plt.clf()

def compute_class_W(df: pd.DataFrame) -> np.array:
    target = df.NObeyesdad
    unique_classes = np.unique(np.array(target))

    unique_classes_list = list(unique_classes)

    class_weights = compute_class_weight(
        "balanced", classes=unique_classes_list, y=target
    )

    class_weights_dict = dict(zip(unique_classes_list, class_weights))
    print(class_weights_dict)

    le = LabelEncoder()
    target_encoded = le.fit_transform(target).astype(np.uint8)

    unique_classes = np.unique(target_encoded)

    unique_classes_list = list(unique_classes)

    class_weights = compute_class_weight(
        "balanced", classes=unique_classes_list, y=target_encoded
    )

    class_weights_dict = dict(zip(unique_classes_list, class_weights))
    print(class_weights_dict)

    return class_weights_dict