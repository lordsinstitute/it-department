import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg'
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ==============================
# Configuration
# ==============================
DATASET_PATH = "images"
PERFORMANCE_PATH = "static/performance"
os.makedirs(PERFORMANCE_PATH, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 10
EPOCHS_FINE = 10

# ==============================
# Load Dataset
# ==============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Save class names
with open(os.path.join(PERFORMANCE_PATH, "class_names.json"), "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)

# ==============================
# Model Definition
# ==============================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    ]
)

# ==============================
# Callbacks
# ==============================
checkpoint = ModelCheckpoint(
    os.path.join(PERFORMANCE_PATH, "best_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

# ==============================
# Phase 1 Training
# ==============================
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_INITIAL,
    callbacks=callbacks
)

# ==============================
# Fine-Tuning
# ==============================
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

# Save Final Model
model.save(os.path.join(PERFORMANCE_PATH, "final_model.h5"))

# ==============================
# Combine History
# ==============================
history = {}
for key in history1.history.keys():
    history[key] = history1.history[key] + history2.history[key]

with open(os.path.join(PERFORMANCE_PATH, "training_history.json"), "w") as f:
    json.dump(history, f)

# ==============================
# Plot Accuracy & Loss
# ==============================
epochs_range = range(1, len(history["accuracy"]) + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs_range, history["accuracy"], label="Train Accuracy")
plt.plot(epochs_range, history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(os.path.join(PERFORMANCE_PATH, "accuracy_curve.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(epochs_range, history["loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(PERFORMANCE_PATH, "loss_curve.png"))
plt.close()

# ==============================
# Evaluation Metrics
# ==============================
y_true = []
y_pred = []
y_prob = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_prob.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(PERFORMANCE_PATH, "classification_report.csv"))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(PERFORMANCE_PATH, "confusion_matrix.png"))
plt.close()

# ROC Curve (Micro)
y_true_bin = label_binarize(y_true, classes=range(num_classes))
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curve (Micro Average)")
plt.savefig(os.path.join(PERFORMANCE_PATH, "roc_curve.png"))
plt.close()

print("Training and evaluation complete. All outputs saved in static/performance/")