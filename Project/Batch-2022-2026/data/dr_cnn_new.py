import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# === Helper Functions ===

def load_images(path, max_per_class=1000):
    data, labels = [], []
    for label in os.listdir(path):
        count = 0
        for file in os.listdir(os.path.join(path, label)):
            if file.endswith(".jpg") or file.endswith(".png"):
                if count >= max_per_class:
                    break
                img = Image.open(os.path.join(path, label, file)).convert('RGB')
                img = img.resize((224, 224))
                data.append(np.array(img))
                labels.append(label)
                count += 1
    return np.array(data), labels

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def preprocess_images(x):
    reference = x[0]
    for i in range(len(x)):
        x[i] = apply_clahe(x[i])
        kernel = np.array([[0, -0.01, 0], [-0.01, 3, -0.01], [0, -0.01, 0]])
        x[i] = cv2.filter2D(x[i], -1, kernel)
        x[i] = cv2.morphologyEx(x[i], cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return x

def build_cnn_model(input_shape, num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === Main Pipeline ===

def run_pipeline():
    # Load and preprocess
    x, y = load_images("../colored_images/")
    x = preprocess_images(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Reshape for SMOTE
    x_train_flat = x_train.reshape(len(x_train), -1)
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_flat, y_train)
    x_train_res = x_train_res.reshape(-1, 224, 224, 3)

    # Normalize
    x_train_res = x_train_res / 255.0
    x_test = x_test / 255.0

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_res)
    y_test_enc = le.transform(y_test)
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Label Map:", label_map)

    # One-hot
    y_train_oh = tf.keras.utils.to_categorical(y_train_enc, num_classes=5)
    y_test_oh = tf.keras.utils.to_categorical(y_test_enc, num_classes=5)

    # Model
    model = build_cnn_model((224, 224, 3))
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("../models/best_cnn_dr_new.h5", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]
    history = model.fit(x_train_res, y_train_oh, epochs=25, validation_data=(x_test, y_test_oh), callbacks=callbacks)

    # === Evaluation Plots ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.savefig("../static/vis/cnn_acc.jpg")
    plt.close()

    # === Metrics ===
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_oh, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("../static/vis/cnn_cnfmat.jpg")
    plt.close()

    report = classification_report(y_true, y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt=".2f")
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig("../static/vis/cnn_clfrpt.jpg")
    plt.close()

    # === ROC Curve ===
    fpr, tpr, roc_auc = {}, {}, {}
    y_test_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("../static/vis/cnn_roc_curve.jpg")
    plt.close()

# Run the pipeline
run_pipeline()
