import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import cv2
#from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.models import load_model
from keras.layers import Input, Dense,Conv2D , MaxPooling2D, Flatten,BatchNormalization,Dropout

from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Save the classification report as an image
def save_report_as_image(report, image_path):
    # Create a figure
    fig = plt.figure(figsize=(10, 5))
    plt.text(0.1, 0.5, report, fontsize=12, family="monospace")
    plt.axis('off')  # Remove axes
    plt.title("CNN Classification Report")
    # Save the figure
    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)



def plot_confusion_matrix(conf_matrix, file_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

def createModel():
    normal_path = "../KneeDS2/Training/0Normal"
    doubtful_path = "../KneeDS2/Training/1Doubtful"
    mild_path = "../KneeDS2/Training/2Mild"
    moderatepath = "../KneeDS2/Training/3Moderate"
    severepath = "../KneeDS2/Training/4Severe"

    normal_folder = os.listdir(normal_path)
    doubtful_folder = os.listdir(doubtful_path)
    mild_folder = os.listdir(mild_path)
    moderate_folder = os.listdir(moderatepath)
    severe_folder = os.listdir(severepath)

    print("Images in Normal Data:", len(normal_folder))
    print("Images in Doubtful Data:", len(doubtful_folder))
    print("Images in Mild Data:", len(mild_folder))
    print("Images in Moderate Data:", len(moderate_folder))
    print("Images in Sever Data:", len(severe_folder))

    data = []

    for img_file in normal_folder:
        image = Image.open("../KneeDS2/Training/0Normal/" + img_file)
        image = image.resize((224, 224))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    for img_file in doubtful_folder:
        image = Image.open("../KneeDS2/Training/1Doubtful/" + img_file)
        image = image.resize((224, 224))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    for img_file in mild_folder:
        image = Image.open("../KneeDS2/Training/2Mild/" + img_file)
        image = image.resize((224, 224))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    for img_file in moderate_folder:
        image = Image.open("../KneeDS2/Training/3Moderate/" + img_file)
        image = image.resize((224, 224))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    for img_file in severe_folder:
        image = Image.open("../KneeDS2/Training/4Severe/" + img_file)
        image = image.resize((224, 224))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    normal_label = [0] * len(normal_folder)
    doubtful_label= [1] * len(doubtful_folder)
    mild_label=[2] * len(mild_folder)
    moderate_label=[3]*len(moderate_folder)
    severe_label=[4]*len(severe_folder)
    Target_label = normal_label + doubtful_label+mild_label+moderate_label+severe_label

    x = np.array(data)
    y = np.array(Target_label)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    x_train_s = x_train / 255
    x_test_s = x_test / 255

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    x_train_s = x_train / 255
    x_test_s = x_test / 255

    model = Sequential()

    model.add(Conv2D(filters=100, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu",
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    history = model.fit(x_train_s, y_train, batch_size=32,
                        epochs=10, validation_data=(x_test_s, y_test))

    model.save('../models/cnn_koa.h5')

    loss, acc = model.evaluate(x_test_s, y_test)
    print("Loss on Test Data:", loss)
    print("Accuracy on Test Data:", acc)

    loss, acc = model.evaluate(x_train_s, y_train)
    print("Loss on Train Data:", loss)
    print("Accuracy on Train Data:", acc)

    y_pred_test = model.predict(x_test_s)
    y_pred_test_label = [1 if i >= 0.5 else 0 for i in y_pred_test]

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CNN model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('../static/vis/cnn_accuracy.jpg')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('../static/vis/cnn_loss.jpg')
    plt.clf()

    print("-----Metrics Evaluation On Test Data -----")
    print()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_label))
    print()
    print("Classification Report:\n", classification_report(y_test, y_pred_test_label))

    # Plot confusion matrix for testing data
    conf_matrix = confusion_matrix(y_test, y_pred_test_label)

    plot_confusion_matrix(conf_matrix, "../static/vis/cnn_confustion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'.")

    # Testing Classification Report
    testing_report = classification_report(y_test, y_pred_test_label)
    save_report_as_image(testing_report, "../static/vis/cnn_clf_rpt.png")


    y_prob = model.predict(x_test_s)

    fpr, tpr, threshold = roc_curve(y_test, y_prob)

    # Compute ROC curve and ROC area
    roc_auc = auc(fpr, tpr)
    print(f"ROC - Area :{roc_auc}")

    # Create a DataFrame
    roc_df = pd.DataFrame({
        'False Positive Rate (FPR)': fpr,
        'True Positive Rate (TPR)': tpr,
        'Thresholds': threshold
    })

    # Save the DataFrame to a CSV file (optional)
    roc_df.to_csv('../cnn_roc_data.csv', index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('../static/vis/cnn_roc.jpg')
    plt.clf()

    y_prob = model.predict(x_test_s)

    precision, recall, threshold = precision_recall_curve(y_test, y_prob)

    # Compute ROC curve and ROC area
    pr_auc = auc(recall, precision)
    print(f"PR - Area :{pr_auc}")

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', where='post', label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc='upper right')
    plt.savefig('../static/vis/cnn_prc.jpg')
    plt.clf()

createModel()