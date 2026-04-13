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
import optuna
import pandas as pd
# for model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score # for Gini-mean
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np

from utils.custom_model import CustomModelling
from utils.dataset import ObesityDataset

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


SEED = 42
acc={}

def createModel():
    train_df1 = pd.read_csv('train.csv')
    train_df2 = pd.read_csv('ObesityDataSet.csv')
    test_df = pd.read_csv('test.csv')

    full_train = pd.concat([train_df1, train_df2], axis=0).reset_index(drop=True)
    full_train.drop('id', axis=1, inplace=True)
    full_train.insert(0, 'id', full_train.index + 1)
    full_train.rename(columns={'family_history_with_overweight': 'FamHist'}, inplace=True)
    test_df.rename(columns={'family_history_with_overweight': 'FamHist'}, inplace=True)

    target = full_train.NObeyesdad
    unique_classes = np.unique(np.array(target))

    unique_classes_list = list(unique_classes)

    class_weights = compute_class_weight('balanced',
                                         classes=unique_classes_list,
                                         y=target)

    class_weights_dict = dict(zip(unique_classes_list, class_weights))
    print(class_weights_dict)

    le = LabelEncoder()
    target_encoded = le.fit_transform(target).astype(np.uint8)

    unique_classes = np.unique(target_encoded)

    unique_classes_list = list(unique_classes)

    class_weights = compute_class_weight('balanced',
                                         classes=unique_classes_list,
                                         y=target_encoded)

    class_weights_dict = dict(zip(unique_classes_list, class_weights))
    print(class_weights_dict)

    class_weights = {
        0: 1.1688729874776387,  # Insufficient_Weight
        1: 0.9697239536954586,  # Normal_Weight
        2: 1.0018399264029438,  # Obesity_Type_I
        3: 0.9215796897038082,  # Obesity_Type_II
        4: 0.7475972540045767,  # Obesity_Type_III
        5: 1.2024291497975708,  # Overweight_Level_I
        6: 1.1618065433854907  # Overweight_Level_II
    }

    optunaObesityDataset = ObesityDataset(train_data=full_train.copy(), test_data=test_df.copy())
    finalObesityDataset = ObesityDataset(train_data=full_train.copy(), test_data=test_df.copy())

    opt_string, final_string = "Optuna Dataset", "Final Dataset"
    print(f"\n{opt_string:^70}")
    x_train_opt, y_train_opt, x_valid_opt, y_valid_opt, x_test_opt, test_ids_opt, le_opt = optunaObesityDataset.build_data()
    print(f"\n{final_string:^70}")
    x_train, y_train, x_valid, y_valid, x_test, test_ids, le = finalObesityDataset.build_data(0)

    LGBMClassifierParamsKaggle = {
        "objective": "multiclass",  # Objective function for the model
        "metric": "multi_logloss",  # Evaluation metric
        "boosting_type": "gbdt",  # Gradient boosting type
        "num_class": 7,  # Number of classes in the dataset
        'learning_rate': 0.030962211546832760,  # Learning rate for gradient boosting
        'n_estimators': 500,  # Number of boosting iterations
        'lambda_l1': 0.009667446568254372,  # L1 regularization term
        'lambda_l2': 0.04018641437301800,  # L2 regularization term
        'max_depth': 10,  # Maximum depth of the trees
        'colsample_bytree': 0.40977129346872643,  # Fraction of features to consider for each tree
        'subsample': 0.9535797422450176,  # Fraction of samples to consider for each boosting iteration
        'min_child_samples': 26,  # Minimum number of data needed in a leaf
    }

    lgbm_params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_class": 7,
        'learning_rate': 0.031,
        'n_estimators': 550,
        'lambda_l1': 0.010,
        'lambda_l2': 0.040,
        'max_depth': 20,
        'colsample_bytree': 0.413,
        'subsample': 0.97,
        'min_child_samples': 25,
        'class_weight': 'balanced',

    }

    lgbm_params_ova = {
        "objective": "multiclass_ova",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_class": 7,
        'learning_rate': 0.031,
        'n_estimators': 550,
        'lambda_l1': 0.010,
        'lambda_l2': 0.040,
        'max_depth': 20,
        'colsample_bytree': 0.413,
        'subsample': 0.97,
        'min_child_samples': 25,
        'class_weight': 'balanced',
    }

    text = "Best Model Stratified K-Fold CV on all dataset"
    print(f"\n{text:^70}")
    FinalLGBMCustom = CustomModelling(

        model=LGBMClassifier(**LGBMClassifierParamsKaggle, random_state=SEED, verbose=-1),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test
    )
    stratifiedLGBM_test_predictions = FinalLGBMCustom.stratifiedKCV(splits=10, seed=SEED)

#createModel()