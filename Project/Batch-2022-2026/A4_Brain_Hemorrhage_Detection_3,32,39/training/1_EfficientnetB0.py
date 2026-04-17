import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# 1. CONFIGURATION
# ===============================
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "head_ct")
LABELS_PATH = os.path.join(BASE_DIR, "dataset", "labels.csv")



MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/brain_hemorrhage_model_effnet.h5")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "models/best_model.h5")

np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# 2. LOAD LABELS (FIXED VERSION)
# ===============================
df = pd.read_csv(LABELS_PATH)

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

print("Columns detected:", df.columns)

# Create proper image path (000.png format)
df["filepath"] = df["id"].astype(int).apply(
    lambda x: os.path.join(DATASET_PATH, f"{x:03d}.png")
)

# Create label column
df["label"] = df["hemorrhage"].astype(str)

print(df.head())

# ===============================
# 3. TRAIN / VAL / TEST SPLIT
# ===============================
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# ===============================
# 4. DATA GENERATORS
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filepath",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col="filepath",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col="filepath",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ===============================
# 5. BUILD MODEL (TRANSFER LEARNING)
# ===============================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ===============================
# 6. CALLBACKS
# ===============================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2, verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True)
]

# ===============================
# 7. TRAIN MODEL
# ===============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_SAVE_PATH)

# ===============================
# 8. PLOT ACCURACY & LOSS
# ===============================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("../static/effnet/accuracy_curve.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("../static/effnet/loss_curve.png")
plt.close()

# ===============================
# 9. TEST EVALUATION
# ===============================
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print("Test Accuracy:", test_acc)
print("Test AUC:", test_auc)

# Predictions
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int)
y_true = test_generator.classes

# ===============================
# 10. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("../static/effnet/confusion_matrix.png")
plt.close()

# ===============================
# 11. CLASSIFICATION REPORT
# ===============================
report = classification_report(y_true, y_pred)
print(report)

with open("../static/effnet/classification_report.txt", "w") as f:
    f.write(report)

# ===============================
# 12. ROC CURVE
# ===============================
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.savefig("../static/effnet/roc_curve.png")
plt.close()

# ===============================
# 13. SAVE TRAINING HISTORY
# ===============================
history_df = pd.DataFrame(history.history)
history_df.to_csv("../static/effnet/training_history.csv", index=False)

print("\nTraining Complete. All outputs saved successfully.")
