import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================================
# 1️⃣ Create Output Folder
# ======================================

output_dir = "static/dl_model_outputs"
os.makedirs(output_dir, exist_ok=True)

# ======================================
# 2️⃣ Load Dataset
# ======================================

df = pd.read_csv("engine_data.csv")

df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

X = df.drop("Engine Condition", axis=1)
y = df["Engine Condition"]

if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ======================================
# 3️⃣ Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================
# 4️⃣ Scaling
# ======================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================
# 5️⃣ Class Weights
# ======================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(enumerate(class_weights))

# ======================================
# 6️⃣ Build Model
# ======================================

model = Sequential()

model.add(Dense(256, input_dim=X_train.shape[1]))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ======================================
# 7️⃣ Callbacks
# ======================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# ======================================
# 8️⃣ Train
# ======================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=64,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ======================================
# 9️⃣ Evaluate
# ======================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Save accuracy to file
with open(os.path.join(output_dir, "test_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy}\n")
    f.write(f"Test Loss: {loss}\n")

# ======================================
# 🔟 Predictions
# ======================================

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

# ======================================
# 1️⃣1️⃣ Save Classification Report
# ======================================

report = classification_report(y_test, y_pred)

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

print("\nClassification Report:\n")
print(report)

# ======================================
# 1️⃣2️⃣ Save Confusion Matrix
# ======================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# ======================================
# 1️⃣3️⃣ Save Accuracy & Loss Curves
# ======================================

# Accuracy Curve
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
plt.close()

# Loss Curve
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

# ======================================
# 1️⃣4️⃣ Save ROC Curve
# ======================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# ======================================
# 1️⃣5️⃣ Save Model
# ======================================

model.save(os.path.join(output_dir, "engine_condition_model.h5"))

print("\n✅ All outputs saved inside 'model_outputs/' folder.")
