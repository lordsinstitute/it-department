import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.custom_model import CustomModelling
from utils.dataset import ObesityDataset

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


SEED = 42
acc={}

def compAlg():
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

    XGBOptunaBest = {
        'tree_method': "gpu_hist",
        'n_estimators': 1823,
        'max_depth': 6,
        'min_child_weight': 3.848335213066268,
        'learning_rate': 0.04649747693831137,
        'subsample': 0.8260504483747326,
        'gamma': 0.49014261098410755,
        'colsample_bytree': 0.2707031507217031,
        'colsample_bylevel': 0.9294658015206282,
        'colsample_bynode': 0.6231915247330754
    }

    text = "XGBoostClassifer Optuna Tuning"
    print(f"\n{text:^70}")
    optunaXGBCustom = CustomModelling(

        model=xgb.XGBClassifier(**XGBOptunaBest, class_weight=class_weights, random_state=SEED, verbose=-1),
        x_train=x_train_opt,
        y_train=y_train_opt,
        x_test=None
    )
    model = optunaXGBCustom.train(optunaXGBCustom.x_train, optunaXGBCustom.y_train)
    preds = optunaXGBCustom.inference(x_valid_opt)
    accuracy, precision, recall, f1 = optunaXGBCustom.compute_scores(y_valid_opt, preds)
    print("\n------------------------------------------------------------------------")
    print(f"Accuracy on Validation Data is:", accuracy)
    acc["XGBoost-Optuna"]=accuracy*100
    print(f"Precision:", precision)
    print(f"Recall:", recall)
    print(f"F1 Score:", f1)
    optunaXGBCustom.classification_reports(y_valid_opt, preds,"XGBoost-Optuna")
    optunaXGBCustom.confusion_matrices(y_valid_opt, preds, "XGBoost-Optuna")



    LGBMClassifierOptunaBest = {
        'max_depth': 15,
        'n_estimators': 520,
        'learning_rate': 0.012581445996445997,
        'min_child_weight': 1.8172231887139634,
        'min_child_samples': 173,
        'subsample': 0.5302941162865715,
        'subsample_freq': 3,
        'colsample_bytree': 0.6074110342114967,
        'num_leaves': 32
    }

    text = "LGBMClassifier Optuna Tuning"
    print(f"\n{text:^70}")
    optunaLGBMustom = CustomModelling(

        model=LGBMClassifier(**LGBMClassifierOptunaBest, class_weight=class_weights, random_state=SEED, verbose=-1),
        x_train=x_train_opt,
        y_train=y_train_opt,
        x_test=None
    )
    model = optunaLGBMustom.train(optunaLGBMustom.x_train, optunaLGBMustom.y_train)
    preds = optunaLGBMustom.inference(x_valid_opt)
    accuracy, precision, recall, f1 = optunaLGBMustom.compute_scores(y_valid_opt, preds)
    print("\n------------------------------------------------------------------------")
    print(f"Accuracy on Validation Data is:", accuracy)
    acc["LGBM-Optuna"] = accuracy*100
    print(f"Precision:", precision)
    print(f"Recall:", recall)
    print(f"F1 Score:", f1)
    optunaLGBMustom.classification_reports(y_valid_opt, preds, "LGBM-Optuna")
    optunaLGBMustom.confusion_matrices(y_valid_opt, preds, "LGBM-Optuna")

    CatBoostOptunaBest = {'learning_rate': 0.13762007048684638,
                          'depth': 5,
                          'l2_leaf_reg': 5.285199432056192,
                          'bagging_temperature': 0.6029582154263095,
                          'random_seed': SEED,
                          'verbose': False,
                          'task_type': "GPU",
                          'iterations': 1000
                          }
    text = "CatBoost Optuna Tuning"
    print(f"\n{text:^70}")
    optunaCatBoostustom = CustomModelling(

        model=CatBoostClassifier(**CatBoostOptunaBest),
        x_train=x_train_opt,
        y_train=y_train_opt,
        x_test=None
    )
    model = optunaCatBoostustom.train(optunaCatBoostustom.x_train, optunaCatBoostustom.y_train)
    preds = optunaCatBoostustom.inference(x_valid_opt)
    accuracy, precision, recall, f1 = optunaCatBoostustom.compute_scores(y_valid_opt, preds)
    print("\n------------------------------------------------------------------------")
    print(f"Accuracy on Validation Data is:", accuracy)
    acc["Catboost-Optuna"] = accuracy*100
    print(f"Precision:", precision)
    print(f"Recall:", recall)
    print(f"F1 Score:", f1)
    optunaCatBoostustom.classification_reports(y_valid_opt, preds, "Catboost-Optuna")
    optunaCatBoostustom.confusion_matrices(y_valid_opt, preds, "Catboost-Optuna")

    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 5))
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel("Accuracy %")
    plt.xlabel("Algorithms")
    ax = sns.barplot(x=list(acc.keys()), y=list(acc.values()), palette='rainbow')
    for i, accuracy in enumerate(acc.values()):
        ac = round(accuracy, 2)
        plt.text(i, accuracy + 2, str(ac), ha='center', va='bottom', fontsize=10)

    plt.savefig('static/vis/Algcomp.jpg')
    plt.clf()
    return acc


#compAlg()