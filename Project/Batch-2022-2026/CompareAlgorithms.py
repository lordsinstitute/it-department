import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#data transformation
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
# Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
r2={}
def compAlg():
    data = pd.read_csv('final_data.csv')
    df = data.drop(
        ['Model Year', 'Make', 'Model', 'Fuel Consumption (City (L/100 km)', 'Fuel Consumption(Hwy (L/100 km))',
         'Fuel Consumption(Comb (mpg))', 'CO2 Emissions(g/km)', 'Smog Rating'], axis=1)

    df = df.rename(columns={'Vehicle Class': 'Vehicle Class', 'Engine Size(L)': 'Engine Size', 'Cylinders': 'Cylinders',
                            'Transmission': 'Transmission', 'Fuel Type': 'Fuel Type',
                            'Fuel Consumption(Comb (L/100 km))': 'Fuel Consumption', 'CO2 Rating': 'CO2 Rating'})
    df['Fuel Type'].fillna((df['Fuel Type'].mode()[0]), inplace=True)

    df['CO2 Rating'].fillna(0, inplace=True)
    new_ratting = []

    for fuel, co2 in zip(df['Fuel Consumption'], df['CO2 Rating']):
        if co2 == 0:
            if 20 <= fuel:
                new_ratting.append(1)
            elif 16.0 <= fuel < 20.0:
                new_ratting.append(2)
            elif 14.0 <= fuel < 16.0:
                new_ratting.append(3)
            elif 12.0 <= fuel < 14.0:
                new_ratting.append(4)
            elif 10.0 <= fuel < 12.0:
                new_ratting.append(5)
            elif 8.0 <= fuel < 10.0:
                new_ratting.append(6)
            elif 7.0 <= fuel < 8.0:
                new_ratting.append(7)
            elif 6.0 <= fuel < 7.0:
                new_ratting.append(8)
            elif 5.0 <= fuel < 6.0:
                new_ratting.append(9)
            elif fuel < 5.0:
                new_ratting.append(10)
        else:
            new_ratting.append(co2)

    df['CO2 Rating'] = new_ratting

    df = df.replace(
        {'Transmission': {'AM8': 'AM', 'AS10': 'AS', 'A8': 'A', 'A9': 'A', 'AM7': 'AM', 'AS8': 'AS', 'M6': 'M', \
                          'AS6': 'AS', 'AS9': 'AS', 'A10': 'A', 'A6': 'A', 'M5': 'M', 'M7': 'M', 'AV7': 'AV',
                          'AV1': 'AV', 'AM6': 'AM', 'AS7': 'AS', 'AV8': 'AV', 'AV6': 'AV', 'AV10': 'AV', 'AS5': 'AS',
                          'A7': 'A'}})

    # from scipy.stats import chi2_contingency
    fuel_type = pd.crosstab(df['Transmission'], df['Fuel Type'])

    Chi_square_statistic, p, dof, expec = chi2_contingency(fuel_type)

    alpha = 0.05
    print("p_value is " + str(p))
    if p <= alpha:
        print('Dependent')
        print('dof is ' + str(dof))
    else:
        print('Independent')
        print('dof is ' + str(dof))

    Class = pd.crosstab(df['Transmission'], df['Vehicle Class'])
    Chi_square_statistic, p, dof, expec = chi2_contingency(Class)

    alpha = 0.05
    print("p_value is " + str(p))
    if p <= alpha:
        print('Dependent')
        print('dof is ' + str(dof))
    else:
        print('Independent')
        print('dof is ' + str(dof))

    order = ['AV', 'AM', 'M', 'AS', 'A']

    od = OrdinalEncoder(categories=[order])

    df["Transmission_X"] = od.fit_transform(df[["Transmission"]])

    order = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 'SUV: Small',
             'SUV: Standard', 'Minivan', \
             'Station wagon: Small', 'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle', \
             'Pickup truck: Standard']

    od = OrdinalEncoder(categories=[order])

    df["Vehicle Class_X"] = od.fit_transform(df[["Vehicle Class"]])

    new_df = df['Fuel Type'].str.get_dummies()

    df = pd.concat([df, new_df], axis=1)
    df.to_csv('df_train.csv')

    x = df.drop(["Fuel Type", 'Fuel Consumption', "Vehicle Class", "Transmission"], axis=1)

    x.to_csv('Xtrain.csv')

    y = df['Fuel Consumption']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=51)

    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest)

    lr = LinearRegression()
    lr.fit(xtrain, ytrain)
    print("training score = ", lr.score(xtrain, ytrain))
    print("testing score = ", lr.score(xtest, ytest))
    ypred = lr.predict(xtest)
    r2['LR']=round(r2_score(ytest, ypred)*100,3)
    print(r2['LR'])
    plt.scatter(ytest, ypred, alpha=0.7)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')  # Diagonal line
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/LR_pred.jpg')
    plt.clf()

    dc = DecisionTreeRegressor(max_depth=4)
    dc.fit(xtrain, ytrain)
    ypred = dc.predict(xtest)
    print("training score = ", dc.score(xtrain, ytrain))
    print("testing score = ", dc.score(xtest, ytest))
    r2['DT'] = round(r2_score(ytest, ypred) * 100, 3)
    print(r2['DT'])
    plt.scatter(ytest, ypred, alpha=0.7)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')  # Diagonal line
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/DT_pred.jpg')
    plt.clf()

    rf = RandomForestRegressor(n_estimators = 60 , min_samples_split = 4, max_features =  'sqrt', max_depth = 10, criterion='squared_error')
    rf.fit(xtrain, ytrain)
    ypred = rf.predict(xtest)
    print("training score = ", rf.score(xtrain, ytrain))
    print("testing score = ", rf.score(xtest, ytest))
    r2['RF'] = round(r2_score(ytest, ypred) * 100, 3)
    print(r2['RF'])
    plt.scatter(ytest, ypred, alpha=0.7)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')  # Diagonal line
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/RF_pred.jpg')
    plt.clf()

    reg = XGBRegressor()
    reg.fit(xtrain, ytrain)
    y_pred = reg.predict(xtest)
    r2['XGB'] = round(r2_score(ytest, y_pred) * 100, 3)
    print('R2_score (test) XGB: {0:.3f}'.format(r2['XGB']))
    plt.scatter(ytest, y_pred, alpha=0.7)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')  # Diagonal line
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/XGB_pred.jpg')
    plt.clf()

    reg = AdaBoostRegressor()
    reg.fit(xtrain, ytrain)
    y_pred = reg.predict(xtest)
    r2['AB'] = round(r2_score(ytest, y_pred) * 100, 3)
    print('R2_score (test) AB: {0:.3f}'.format(r2['AB']))
    plt.scatter(ytest, y_pred, alpha=0.7)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')  # Diagonal line
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title("Actual vs Fuel Consumption")
    plt.savefig('static/vis/AB_pred.jpg')
    plt.clf()

    reg = Lasso(alpha=1.0)
    reg.fit(xtrain, ytrain)
    y_pred = reg.predict(xtest)
    r2['Lasso'] = round(r2_score(ytest, y_pred) * 100, 3)
    print('R2_score (test) Lasso: {0:.3f}'.format(r2['Lasso']))
    plt.scatter(ytest, y_pred, alpha=0.7)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')  # Diagonal line
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/Lasso_pred.jpg')
    plt.clf()

    # Plotting accuracies of all the models
    colors = ["green", "yellow", "black", "magenta", "#0e76a8", "red"]

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel("\nAccuracy %", fontsize=20)
    plt.xlabel("\nAlgorithms", fontsize=20)
    sns.barplot(x=list(r2.keys()), y=list(r2.values()), palette=colors)

    ax = sns.barplot(x=list(r2.keys()), y=list(r2.values()), palette=colors)
    for i, accuracy in enumerate(r2.values()):
        plt.text(i, accuracy + 2, str(accuracy), ha='center', va='bottom', fontsize=10)

    plt.savefig('static/vis/AlgComp.jpg')
    plt.clf()

    return r2


#compAlg()







