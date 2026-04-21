#import the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import KFold # import KFold
import warnings
warnings.filterwarnings('ignore')


def compAlg():
    df = pd.read_csv('RTA_Dataset.csv')
    # convert the 'Date' column to datetime format
    df['Time'] = pd.to_datetime(df['Time'])
    # dropping columns that can cause imbalance while imputation
    lists = ['Vehicle_driver_relation', 'Work_of_casuality', 'Fitness_of_casuality', 'Day_of_week', 'Casualty_severity',
             'Time', 'Sex_of_driver', 'Educational_level', 'Defect_of_vehicle', 'Owner_of_vehicle',
             'Service_year_of_vehicle', 'Road_surface_type', 'Sex_of_casualty']
    df.drop(columns=lists, inplace=True)

    # fill missing values with mean column values
    df['Driving_experience'].fillna(df['Driving_experience'].mode()[0], inplace=True)
    df['Age_band_of_driver'].fillna(df['Age_band_of_driver'].mode()[0], inplace=True)
    df['Type_of_vehicle'].fillna(df['Type_of_vehicle'].mode()[0], inplace=True)
    df['Area_accident_occured'].fillna(df['Area_accident_occured'].mode()[0], inplace=True)
    df['Road_allignment'].fillna(df['Road_allignment'].mode()[0], inplace=True)
    df['Type_of_collision'].fillna(df['Type_of_collision'].mode()[0], inplace=True)
    df['Vehicle_movement'].fillna(df['Vehicle_movement'].mode()[0], inplace=True)
    df['Lanes_or_Medians'].fillna(df['Lanes_or_Medians'].mode()[0], inplace=True)
    df['Types_of_Junction'].fillna(df['Types_of_Junction'].mode()[0], inplace=True)

    from sklearn.preprocessing import LabelEncoder  # or one hot encoder
    LE = LabelEncoder()
    df = df.apply(LE.fit_transform)  # categorical values to integers

    #Upsampling
    x = df.drop('Accident_severity', axis=1)
    y = df['Accident_severity']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

    # upsampling using smote

    counter = Counter(ytrain)

    print("=============================")

    for k, v in counter.items():
        per = 100 * v / len(ytrain)
        print(f"Class= {k}, n={v} ({per:.2f}%)")

    oversample = SMOTE()
    xtrain, ytrain = oversample.fit_resample(xtrain, ytrain)

    counter = Counter(ytrain)

    print("=============================")

    for k, v in counter.items():
        per = 100 * v / len(ytrain)
        print(f"Class= {k}, n={v} ({per:.2f}%)")

    print("=============================")

    print("Upsampled data shape: ", xtrain.shape, ytrain.shape)

    x = df.drop(columns=["Accident_severity"])
    y = df["Accident_severity"]

    models = {"LG": LogisticRegression(),
              "DT": DecisionTreeClassifier(),
              "SVM": SVC(),
              "KNN": KNeighborsClassifier(),
              "GNB": GaussianNB(),
              "RF": RandomForestClassifier(),
              "AB": AdaBoostClassifier(),
              "GB": GradientBoostingClassifier(),
              }

    acc = modelAccuracy(models, x, y, 1)

    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 5))
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel("Accuracy %")
    plt.xlabel("Algorithms")
    ax = sns.barplot(x=list(acc.keys()), y=list(acc.values()), palette='rainbow')
    ax.bar_label(ax.containers[0])
    # for index, row in groupedvalues.iterrows():
    plt.savefig('static/pimg/AlgComp.jpg')
    plt.clf()
    return acc





# models,x,y,scaleFlag=0,1,2
def modelAccuracy(models,x,y,scaleFlag):
    #train/Test
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
    acc_result={}
    for name,model in models.items():
        #pipeline
        #1.Transformer -> 2.Model
        if(scaleFlag==1):
            model_pipeline=Pipeline([('MinMaxScler',MinMaxScaler()),('model',model)])
        elif(scaleFlag==2):
             model_pipeline=Pipeline([('StandardScaler',StandardScaler()),('model',model)])
        else:
            model_pipeline=Pipeline([('model',model)])
        #training/testing on model pipeline
        model_fit=model_pipeline.fit(xtrain,ytrain)
        ypred=model_fit.predict(xtest)
        acc=accuracy_score(ytest,ypred)
        print("The Accuracy for ",name," is :",acc)
        acc_result[name]=round(acc*100,2)
    return acc_result



#compAlg()