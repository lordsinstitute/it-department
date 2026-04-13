# ===============================
# IMPORTS
# ===============================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight


# ===============================
# PATHS
# ===============================
DATASET_DIR = "dataset/test"   # ONLY TRAIN FOLDER
MODEL_DIR   = "models"
PLOT_DIR    = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ===============================
# IMAGE GENERATOR WITH SPLIT
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # 🔥 INTERNAL SPLIT
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class mapping:", train_data.class_indices)


# ===============================
# CLASS WEIGHTS
# ===============================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))


# ===============================
# RESNET MODEL
# ===============================
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Partial fine-tuning
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ===============================
# CALLBACKS
# ===============================
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_resnet_model_test.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ===============================
# TRAIN MODEL
# ===============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)


# ===============================
# ACCURACY & LOSS CURVES
# ===============================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(f"{PLOT_DIR}/accuracy_curve_resnet_test.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(f"{PLOT_DIR}/loss_curve_resnet_test.png")
plt.close()


# ===============================
# VALIDATION PREDICTIONS
# ===============================
y_true = val_data.classes
y_prob = model.predict(val_data).ravel()
y_pred = (y_prob > 0.5).astype(int)


# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Malnutrition"],
    yticklabels=["Normal", "Malnutrition"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{PLOT_DIR}/confusion_matrix_resnet_test.png")
plt.close()


# ===============================
# CLASSIFICATION REPORT
# ===============================
report = classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Malnutrition"]
)

with open(f"{PLOT_DIR}/classification_report_resnet_test.txt", "w") as f:
    f.write(report)


# ===============================
# ROC CURVE
# ===============================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{PLOT_DIR}/roc_curve_resnet_test.png")
plt.close()


print("✅ Training completed successfully")
print("📁 Model saved in models/")
print("📊 Metrics saved in plots/")
