import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


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
path = '../ECG Dataset/train'
data, label = load_data(path, label_dict)
data = data.astype('float32') / 255.0
num_classes = len(label_dict)
label = keras.utils.to_categorical(label, num_classes)
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    filepath="best_ecg_cnn_model.h5",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="max"
)

#history = model.fit(train_data, train_label, epochs=20, batch_size=32, validation_split=0.2)

history = model.fit(
    train_data,
    train_label,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint]
)

test_path = '../ECG Dataset/test'

# Load test data
test_data, test_label = load_data(test_path, label_dict)
test_data = test_data.astype('float32') / 255.0
test_label = keras.utils.to_categorical(test_label, num_classes)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data, test_label)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("../static/evaluation/acc_loss_cnn.jpg")
plt.close()

# Predictions
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_label, axis=1)

class_names = [
    "MI",
    "History of MI",
    "Abnormal Heartbeat",
    "Normal"
]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 14}
)

plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.title("Confusion Matrix - CNN ECG Heart Disease Classification", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig("../static/evaluation/cnf_cnn.jpg")
plt.close()

report = classification_report(
    y_true,
    y_pred,
    target_names=label_dict.keys()
)

print("Classification Report:\n")
print(report)

with open("../static/evaluation/cnn/classification_report_cnn.txt", "w") as f:
    f.write(report)

n_classes = num_classes
y_true_bin = label_binarize(y_true, classes=[0,1,2,3])

plt.figure(figsize=(8,6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f'Class {list(label_dict.keys())[i]} (AUC = {roc_auc:.2f})'
    )

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - ECG Heart Disease Classification')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("../static/evaluation/roc_cnn.jpg")
plt.close()