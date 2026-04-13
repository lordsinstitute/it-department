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

import os
import pickle as pk



def createModel():
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
    x = df.drop(["Fuel Type", 'Fuel Consumption', "Vehicle Class", "Transmission"], axis=1)
    y = df['Fuel Consumption']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=51)

    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest)

    filename = "scaled_data.sav"
    pk.dump(sc, open(filename, "wb"))  # write binary = wb


    reg = LinearRegression()
    reg.fit(xtrain, ytrain)
    y_pred = reg.predict(xtest)

    filename = "trained_model_lr.sav"
    pk.dump(reg, open(filename, "wb"))


    acc = round(r2_score(ytest, y_pred) * 100, 3)
    msg="Linear Regression Model created Successfully"

    return msg, acc

    """
    trail_inputs = ["Compact", 2.4, 4, "AM", 6.0, "Z"]

    pred=input_converter(trail_inputs)
    print(pred)
    """


def input_converter(trail_inputs):
    loaded_scaler = pk.load(open("scaled_data.sav", "rb"))  # read binary = rb
    loaded_model = pk.load(open("trained_model_lr.sav", "rb"))


    vcl = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 'SUV: Small', 'SUV: Standard',
           'Minivan', 'Station wagon: Small', 'Station wagon: Mid-size', 'Pickup truck: Small',
           'Special purpose vehicle', 'Pickup truck: Standard']
    trans = ['AV', 'AM', 'M', 'AS', 'A']
    fuel = ["D", "E", "X", "Z"]
    lst = []
    for i in range(len(trail_inputs)):
        if (type(trail_inputs[i]) == str):
            if (trail_inputs[i] in vcl):
                lst.append(vcl.index(trail_inputs[i]))
            elif (trail_inputs[i] in trans):
                lst.append(trans.index(trail_inputs[i]))
            elif (trail_inputs[i] in fuel):
                print(fuel.index(trail_inputs[i]))
                if (fuel.index(trail_inputs[i]) == 0):
                    lst.extend([1, 0, 0, 0])
                    break
                elif (fuel.index(trail_inputs[i]) == 1):
                    lst.extend([0, 1, 0, 0])
                    break
                elif (fuel.index(trail_inputs[i]) == 2):
                    lst.extend([0, 0, 1, 0])
                    break
                elif (fuel.index(trail_inputs[i]) == 3):
                    lst.extend([0, 0, 0, 1])
        else:
            lst.append(trail_inputs[i])

    print(lst)
    arr = np.asarray(lst)
    arr = arr.reshape(1, -1)
    arr = loaded_scaler.transform(arr)
    prediction = loaded_model.predict(arr)

    return (f"The Fuel Consumption L/100km is {round(prediction[0], 2)}")

#createModel()

#trail_inputs = ["Compact", 2.4, 4, "AM", 6.0, "Z"]
#trail_inputs = ["SUV: Small", 8, 16, "AS", 4.0, "Z"]
#trail_inputs=["Two-seater",6.5,12,"AM",1.0,"Z"]
#pred=input_converter(trail_inputs)
#print(pred)


