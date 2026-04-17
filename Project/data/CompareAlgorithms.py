import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
import lightgbm as lgb


#Using this class to make my print statement look bold
class color:
    BOLD = '\033[1m'

def compAlg():
    df = pd.read_csv("housing.csv")
    df = df.sample(frac=1)

    # Replace ocean proximity with values to help our model
    ocean_proximity = {a: b for b, a in enumerate(df['ocean_proximity'].unique())}
    df.replace(ocean_proximity, inplace=True)

    # The feature 'total_bedrooms' has NaN values so I will replace them with the mean of the feature
    df = df.apply(lambda x: x.fillna(x.mean()))

    df[df['median_house_value'] > 450000]['median_house_value'].value_counts().head()
    df = df.loc[df['median_house_value'] < 500001, :]
    df = df[df['population'] < 25000]

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train, y_train)

    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracies = {}

    accuracy = rf.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['Random Forest'] = accuracy_rounded

    print(color.BOLD + "\nAccuracy of Random Forest Regressor is ", accuracy_rounded, '%')

    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = dt.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['Decision Tree'] = accuracy_rounded
    print(color.BOLD + "\nAccuracy of Decision Tree Regressor is ", accuracy_rounded, '%')

    ada = AdaBoostRegressor(random_state=0)

    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)

    accuracy = ada.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['AdaBoost'] = accuracy_rounded

    print(color.BOLD + "\nAccuracy of AdaBoost Regressor is ", accuracy_rounded, '%')

    xg = XGBRegressor()
    xg.fit(X_train, y_train)

    y_pred = xg.predict(X_test)

    accuracy = xg.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['XGBoost'] = accuracy_rounded

    print(color.BOLD + "\nAccuracy of XGBoost Regressor is ", accuracy_rounded, '%')

    gboost = GradientBoostingRegressor(random_state=42)
    gboost.fit(X_train, y_train)
    y_pred = gboost.predict(X_test)
    accuracy = gboost.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['Gradient Boost'] = accuracy_rounded

    print(color.BOLD + "\nAccuracy of Gradient Boost Regressor is ", accuracy_rounded, '%')

    rid = Ridge(alpha=0.1)

    rid.fit(X_train, y_train)

    y_pred = rid.predict(X_test)

    accuracy = rid.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['Ridge Regression'] = accuracy_rounded

    print(color.BOLD + "\nAccuracy of Ridge Regression is ", accuracy_rounded, '%')

    lasso = linear_model.Lasso(alpha=0.3)

    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test)

    accuracy = lasso.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['Lasso Regression'] = accuracy_rounded
    print(color.BOLD + "\nAccuracy of Lasso Regression is ", accuracy_rounded, '%')

    ela = ElasticNet(random_state=0)
    ela.fit(X_train, y_train)
    y_pred = ela.predict(X_test)
    accuracy = ela.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['Elastic Net Regression'] = accuracy_rounded

    print(color.BOLD + "\nAccuracy of Elastic net Regression is ", accuracy_rounded, '%')

    cat = CatBoostRegressor()

    cat.fit(X_train, y_train)

    y_pred = cat.predict(X_test)

    accuracy = cat.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['CatBoost'] = accuracy_rounded

    print(color.BOLD + "\nAccuracy of CatBoost is ", accuracy_rounded, '%')

    lg = lgb.LGBMRegressor()

    lg.fit(X_train, y_train)

    y_pred = lg.predict(X_test)

    accuracy = lg.score(X_test, y_test)
    accuracy_rounded = round(accuracy * 100, 2)
    accuracies['Light GBM'] = accuracy_rounded
    print(color.BOLD + "\nAccuracy of Light GBM is ", accuracy_rounded, '%')

    # Plotting accuracies of all the models
    colors = ["green", "yellow", "black", "magenta", "#0e76a8", "red", "#34558b", "#f0daa4", "#3b3d4b", "#fd823e"]

    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel("\nAccuracy %", fontsize=20)
    plt.xlabel("\nAlgorithms", fontsize=20)
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

    ax = sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
    for i, accuracy in enumerate(accuracies.values()):

        plt.text(i, accuracy + 2, str(accuracy), ha='center', va='bottom', fontsize=10)

    plt.savefig('static/vis/AlgComp.jpg')
    plt.clf()

    return accuracies

    
#compAlg()

