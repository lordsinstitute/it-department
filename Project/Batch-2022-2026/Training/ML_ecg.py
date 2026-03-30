import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data(path, label_dict):
    data = []
    label = []
    for cat, label_value in label_dict.items():
        pic_list = os.path.join(path, cat)
        for img in os.listdir(pic_list):
            image_path = os.path.join(pic_list, img)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            data.append(image)
            label.append(label_value)
    return np.array(data), np.array(label)

label_dict = {
    'ECG Images of Myocardial Infarction Patients (240x12=2880)': 0,
    'ECG Images of Patient that have History of MI (172x12=2064)': 1,
    'ECG Images of Patient that have abnormal heartbeat (233x12=2796)': 2,
    'Normal Person ECG Images (284x12=3408)': 3
}

path='../ECG Dataset/train'
data,label = load_data(path,label_dict)
data=data.astype('float32')/255.0
num_classes=len(label_dict)
label=keras.utils.to_categorical(label,num_classes)

train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)
train_data_flatten = train_data.reshape(train_data.shape[0], -1)
test_data_flatten = test_data.reshape(test_data.shape[0], -1)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(train_data_flatten, np.argmax(train_label, axis=1))
svm_accuracy = svm_model.score(test_data_flatten, np.argmax(test_label, axis=1))
print("SVM Accuracy:", svm_accuracy)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes, random_state=42)
xgb_model.fit(train_data_flatten, np.argmax(train_label, axis=1))
xgb_accuracy = xgb_model.score(test_data_flatten, np.argmax(test_label, axis=1))
print("XGBoost Accuracy:", xgb_accuracy)

base_estimator = DecisionTreeClassifier(max_depth=5)
adaboost_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
adaboost_model.fit(train_data_flatten, np.argmax(train_label, axis=1))
adaboost_accuracy = adaboost_model.score(test_data_flatten, np.argmax(test_label, axis=1))
print("AdaBoost Accuracy:", adaboost_accuracy)