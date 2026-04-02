import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pickle

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

def createModel():
    data = pd.read_csv('insurance.csv')
    clean_data = {'sex': {'male': 0, 'female': 1},
                  'smoker': {'no': 0, 'yes': 1},
                  'region': {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
                  }
    data_copy = data.copy()
    data_copy.replace(clean_data, inplace=True)

    data_pre = data_copy.copy()

    tempBmi = data_pre.bmi
    tempBmi = tempBmi.values.reshape(-1, 1)
    data_pre['bmi'] = StandardScaler().fit_transform(tempBmi)

    tempAge = data_pre.age
    tempAge = tempAge.values.reshape(-1, 1)
    data_pre['age'] = StandardScaler().fit_transform(tempAge)

    tempCharges = data_pre.charges
    tempCharges = tempCharges.values.reshape(-1, 1)
    data_pre['charges'] = StandardScaler().fit_transform(tempCharges)


    X_ = data_copy.drop('charges', axis=1).values
    y_ = data_copy['charges'].values.reshape(-1, 1)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7,
                                   n_estimators=1200)
    rf_reg.fit(X_train_, y_train_.ravel())

    y_pred_rf_train_ = rf_reg.predict(X_train_)
    r2_score_rf_train_ = r2_score(y_train_, y_pred_rf_train_)

    y_pred_rf_test_ = rf_reg.predict(X_test_)
    r2_score_rf_test_ = r2_score(y_test_, y_pred_rf_test_)

    print('R2 score (train) : {0:.3f}'.format(r2_score_rf_train_))
    print('R2 score (test) : {0:.3f}'.format(r2_score_rf_test_))

    #Pkl_Filename = "rf_tuned.pkl"

    with open("rf_tuned.pkl", 'wb') as file:
        pickle.dump(rf_reg, file)

    msg="Model saved successfully"
    return msg, r2_score_rf_test_*100

#createModel()












