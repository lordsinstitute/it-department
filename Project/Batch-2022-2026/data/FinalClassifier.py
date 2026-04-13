import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pickle as pk


def createModel():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(BASE_DIR, "..")

    # Assuming your data is stored in a pandas DataFrame
    data = pd.read_csv(os.path.join(PROJECT_DIR, "CTG.csv"))

    # Separate the data for each "NSP" class
    nsp_class1 = data[data["NSP"] == 1]
    nsp_class2 = data[data["NSP"] == 2]
    nsp_class3 = data[data["NSP"] == 3]

    # Oversample classes 2 and 3 to match the number of rows in class 1 (1655 rows)
    nsp_class2_oversampled = nsp_class2.sample(n=1655, replace=True, random_state=42)
    nsp_class3_oversampled = nsp_class3.sample(n=1655, replace=True, random_state=42)

    # Concatenate the oversampled data back together
    oversampled_data = pd.concat([nsp_class1, nsp_class2_oversampled, nsp_class3_oversampled], ignore_index=True)

    oversampled_data = oversampled_data.drop(["FileName", "Date", "SegFile", "b", "e"], axis=1)
    oversampled_data = oversampled_data.dropna()
    oversampled_data.isnull().sum()
    X = oversampled_data[['LBE', 'LB', 'AC', 'FM', 'UC', 'DL',
                          'DS', 'DP', 'DR']]
    Y = oversampled_data[["NSP"]]
    X.to_csv(os.path.join(PROJECT_DIR, 'Xtrain.csv'))

    Scaler = StandardScaler()
    X = Scaler.fit_transform(X)

    scaler_path = os.path.join(PROJECT_DIR, "scaled_data.sav")
    pk.dump(Scaler, open(scaler_path, "wb"))  # write binary = wb

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_preds = rf.predict(X_test)
    print('Accuracy:', rf.score(X_test, y_test))
    print('F1:', f1_score(y_test, y_preds, average='macro'))
    print(confusion_matrix(y_test, y_preds))

    model_path = os.path.join(PROJECT_DIR, "trained_model_rf.sav")
    pk.dump(rf, open(model_path, "wb"))

    msg = "Random Forest Classifier created successfully"
    acc = round(rf.score(X_test, y_test) * 100, 3)

    return msg, acc


# createModel()

def testModel(param):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(BASE_DIR, "..")

    scaler_path = os.path.join(PROJECT_DIR, "scaled_data.sav")
    model_path = os.path.join(PROJECT_DIR, "trained_model_rf.sav")

    loaded_scaler = pk.load(open(scaler_path, "rb"))  # read binary = rb
    loaded_model = pk.load(open(model_path, "rb"))

    arr = np.asarray(param)
    arr = arr.reshape(1, -1)
    arr = loaded_scaler.transform(arr)
    prediction = int(loaded_model.predict(arr)[0])

    msg = ""
    if prediction == 1:
        msg = "No risk"
    elif prediction == 2:
        msg = "Suspecting risk"
    else:
        msg = "Pathological condition exists"

    print(prediction)
    return msg


# testModel()