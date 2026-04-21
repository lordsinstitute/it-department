#import the necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')


def create_model():
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

     # or one hot encoder
    #LE = LabelEncoder()
    #df = df.apply(LE.fit_transform)  # categorical values to integers

    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    joblib.dump(encoders, 'label_encoder.pkl')

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

    model = RandomForestClassifier()
    params = {"n_estimators": [100, 200],
              "criterion": ["gini", "entropy"]
              }
    acc=bestParams(model, params, xtrain, ytrain)

    rfc=RandomForestClassifier(n_estimators=200, criterion='gini')
    rfc.fit(xtrain, ytrain)
    ypred = rfc.predict(xtest)

    # Final Evaluation
    print(accuracy_score(ytest, ypred))
    print(classification_report(ytest, ypred))
    joblib.dump(rfc, 'Accident_model.pkl')
    msg = "Model created successfully using Random Forest Classifier after Hyperparameter tuning using GridsearchCV"
    cm = confusion_matrix(ytest, ypred)
    sns.heatmap(cm, annot=True)
    plt.savefig('static/pimg/cm.jpg')
    return  msg,round(acc*100,2)

def bestParams(model,param,xtrain,ytrain):
    #cv
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3)
    grid_cv=GridSearchCV(estimator=model,param_grid=param,cv=cv,scoring="f1_weighted")
    res=grid_cv.fit(xtrain,ytrain)
    print("Best Parameters are ",res.best_params_)
    print("Best Accuracy is ",res.best_score_)
    return res.best_score_


#create_model()

