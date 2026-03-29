import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

# ==============================
# 1. CONFIGURATION
# ==============================
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(BASE_DIR, "head_ct", "TRAIN")
val_dir = os.path.join(BASE_DIR, "head_ct", "VAL")
test_dir = os.path.join(BASE_DIR, "head_ct", "TEST")

OUTPUT_DIR = os.path.join(BASE_DIR, "static","cnn")
MODELS_DIR = os.path.join(BASE_DIR,"models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_cnn_model.h5")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_cnn_model.h5")

# ==============================
# 2. DATA GENERATORS
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=RANDOM_SEED
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ==============================
# 3. BUILD CNN MODEL
# ==============================
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=RMSprop(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 4. CALLBACKS
# ==============================
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.6,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==============================
# 5. TRAIN MODEL
# ==============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop, checkpoint]
)

# Save final model
model.save(FINAL_MODEL_PATH)

# ==============================
# 6. PLOT ACCURACY & LOSS
# ==============================
epochs_range = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, history.history["accuracy"], label='Train')
plt.plot(epochs_range, history.history["val_accuracy"], label='Validation')
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, history.history["loss"], label='Train')
plt.plot(epochs_range, history.history["val_loss"], label='Validation')
plt.title("Loss Curve")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_loss_curve.png"))
plt.close()

# ==============================
# 7. TEST EVALUATION
# ==============================
test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy:", test_acc)

# Predictions
y_probs = model.predict(test_generator)
y_pred = (y_probs > 0.5).astype(int).ravel()
y_true = test_generator.classes

# ==============================
# 8. CLASSIFICATION REPORT
# ==============================
report = classification_report(y_true, y_pred)
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# ==============================
# 9. CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# ==============================
# 10. ROC CURVE
# ==============================
roc_auc = roc_auc_score(y_true, y_probs)

fpr, tpr, _ = roc_curve(y_true, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# ==============================
# 11. SAVE TRAINING HISTORY
# ==============================
import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

print("\nTraining Complete. All outputs saved in 'cnn_outputs' folder.")