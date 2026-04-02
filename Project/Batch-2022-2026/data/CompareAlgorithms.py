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
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

acc={}

def compAlg():
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

    X = data_pre.drop('charges', axis=1).values
    y = data_pre['charges'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)

    cv_linear_reg = cross_val_score(estimator=linear_reg, X=X, y=y, cv=10)

    y_pred_linear_reg_train = linear_reg.predict(X_train)
    r2_score_linear_reg_train = r2_score(y_train, y_pred_linear_reg_train)

    y_pred_linear_reg_test = linear_reg.predict(X_test)
    r2_score_linear_reg_test = r2_score(y_test, y_pred_linear_reg_test)

    rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_reg_test)))

    print('CV Linear Regression : {0:.3f}'.format(cv_linear_reg.mean()))
    print('R2_score (train) : {0:.3f}'.format(r2_score_linear_reg_train))
    print('R2_score (test) : {0:.3f}'.format(r2_score_linear_reg_test))
    print('RMSE : {0:.3f}'.format(rmse_linear))

    acc["LR"]=r2_score_linear_reg_test*100
    #Support Vecotor Regressor
    X_c = data_copy.drop('charges', axis=1).values
    y_c = data_copy['charges'].values.reshape(-1, 1)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    X_train_scaled = StandardScaler().fit_transform(X_train_c)
    y_train_scaled = StandardScaler().fit_transform(y_train_c)
    X_test_scaled = StandardScaler().fit_transform(X_test_c)
    y_test_scaled = StandardScaler().fit_transform(y_test_c)

    svr = SVR()
    svr = SVR(C=10, gamma=0.1, tol=0.0001)
    svr.fit(X_train_scaled, y_train_scaled.ravel())

    parameters = {'kernel': ['rbf', 'sigmoid'],
                  'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
                  'tol': [0.0001],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    svr_grid = GridSearchCV(estimator=svr, param_grid=parameters, cv=10, verbose=4, n_jobs=-1)
    svr_grid.fit(X_train_scaled, y_train_scaled.ravel())

    print(svr_grid.best_estimator_)
    print(svr_grid.best_score_)
    cv_svr = svr_grid.best_score_

    y_pred_svr_train = svr.predict(X_train_scaled)
    r2_score_svr_train = r2_score(y_train_scaled, y_pred_svr_train)

    y_pred_svr_test = svr.predict(X_test_scaled)
    r2_score_svr_test = r2_score(y_test_scaled, y_pred_svr_test)

    rmse_svr = (np.sqrt(mean_squared_error(y_test_scaled, y_pred_svr_test)))

    print('CV : {0:.3f}'.format(cv_svr.mean()))
    print('R2_score (train) : {0:.3f}'.format(r2_score_svr_train))
    print('R2 score (test) : {0:.3f}'.format(r2_score_svr_test))
    print('RMSE : {0:.3f}'.format(rmse_svr))
    acc["SVR"]=r2_score_svr_test*100

    #Ridge Regression
    steps = [('scalar', StandardScaler()),
             ('poly', PolynomialFeatures(degree=2)),
             ('model', Ridge())]

    ridge_pipe = Pipeline(steps)
    parameters = {'model__alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 2, 5, 10, 20, 25, 35, 43, 55, 100],
                  'model__random_state': [42]}
    reg_ridge = GridSearchCV(ridge_pipe, parameters, cv=10)
    reg_ridge = reg_ridge.fit(X_train, y_train.ravel())

    reg_ridge.best_estimator_, reg_ridge.best_score_

    ridge = Ridge(alpha=20, random_state=42)
    ridge.fit(X_train_scaled, y_train_scaled.ravel())
    cv_ridge = reg_ridge.best_score_

    y_pred_ridge_train = ridge.predict(X_train_scaled)
    r2_score_ridge_train = r2_score(y_train_scaled, y_pred_ridge_train)

    y_pred_ridge_test = ridge.predict(X_test_scaled)
    r2_score_ridge_test = r2_score(y_test_scaled, y_pred_ridge_test)

    rmse_ridge = (np.sqrt(mean_squared_error(y_test_scaled, y_pred_linear_reg_test)))
    print('CV : {0:.3f}'.format(cv_ridge.mean()))
    print('R2 score (train) : {0:.3f}'.format(r2_score_ridge_train))
    print('R2 score (test) : {0:.3f}'.format(r2_score_ridge_test))
    print('RMSE : {0:.3f}'.format(rmse_ridge))

    acc["RR"] = r2_score_ridge_test*100

    #Random Forest Regressor
    """
    reg_rf = RandomForestRegressor()
    parameters = {'n_estimators': [600, 1000, 1200],
                  'max_features': ["auto"],
                  'max_depth': [40, 50, 60],
                  'min_samples_split': [5, 7, 9],
                  'min_samples_leaf': [7, 10, 12],
                  'criterion': ['mse']}

    reg_rf_gscv = GridSearchCV(estimator=reg_rf, param_grid=parameters, cv=10, n_jobs=-1)
    reg_rf_gscv = reg_rf_gscv.fit(X_train_scaled, y_train_scaled.ravel())

    reg_rf_gscv.best_score_, reg_rf_gscv.best_estimator_

    rf_reg = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7,
                                   n_estimators=1200)
    rf_reg.fit(X_train_scaled, y_train_scaled.ravel())

    cv_rf = reg_rf_gscv.best_score_
    y_pred_rf_train = rf_reg.predict(X_train_scaled)
    r2_score_rf_train = r2_score(y_train, y_pred_rf_train)

    y_pred_rf_test = rf_reg.predict(X_test_scaled)
    r2_score_rf_test = r2_score(y_test_scaled, y_pred_rf_test)

    rmse_rf = np.sqrt(mean_squared_error(y_test_scaled, y_pred_rf_test))

    print('CV : {0:.3f}'.format(cv_rf.mean()))
    print('R2 score (train) : {0:.3f}'.format(r2_score_rf_train))
    print('R2 score (test) : {0:.3f}'.format(r2_score_rf_test))
    print('RMSE : {0:.3f}'.format(rmse_rf))
    """
    models = [
        ('Linear Regression', rmse_linear, r2_score_linear_reg_train, r2_score_linear_reg_test, cv_linear_reg.mean()),
        ('Ridge Regression', rmse_ridge, r2_score_ridge_train, r2_score_ridge_test, cv_ridge.mean()),
        ('Support Vector Regression', rmse_svr, r2_score_svr_train, r2_score_svr_test, cv_svr.mean())
        #('Random Forest Regression', rmse_rf, r2_score_rf_train, r2_score_rf_test, cv_rf.mean())
        ]

    predict = pd.DataFrame(data=models,columns=['Model', 'RMSE', 'R2_Score(training)', 'R2_Score(test)', 'Cross-Validation'])

    plt.figure(figsize=(12, 7))
    predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)

    sns.barplot(x='Cross-Validation', y='Model', data=predict, palette='Reds')
    plt.xlabel('Cross Validation Score')
    plt.ylabel('Model')
    plt.savefig('static/vis/cvscore.jpg')

    return acc

#compAlg()















