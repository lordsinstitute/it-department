import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

def save_report_as_image(report, image_path):
    # Create a figure
    fig = plt.figure(figsize=(10, 5))
    plt.text(0.1, 0.5, report, fontsize=12, family="monospace")
    plt.axis('off')  # Remove axes
    plt.title("CNN_MobileNet Classification Report")
    plt.tight_layout()
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
    # Set paths
    dataset_dir = '../KneeDS2/Training'  # Change this
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 5
    EPOCHS = 25

    # Data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=25,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Get class indices and labels
    class_indices = train_gen.class_indices  # e.g., {'normal': 0, 'doubtful': 1, ...}
    classes = list(class_indices.values())   # [0, 1, 2, 3, 4]

    # Get class labels for each sample
    train_labels = train_gen.classes
    print(train_labels)

    # Compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    # Convert to dictionary format required by Keras
    class_weight_dict = dict(enumerate(class_weights))
    print("Computed class weights:", class_weight_dict)

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Load MobileNetV2
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint_cb = ModelCheckpoint('../models/cnn_mobilenet_knee_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb],
        class_weight=class_weight_dict
    )

    # Get true labels and predictions from validation generator
    val_gen.reset()
    y_true = []
    y_pred_probs = []

    for _ in range(len(val_gen)):
        X_batch, y_batch = val_gen.next()
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred_probs.extend(model.predict(X_batch))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_indices.keys(),
                yticklabels=class_indices.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("../static/vis/cnn_mob_confusion_matrix.jpg")
    plt.close()

    # --- 2. Classification Report ---
    report = classification_report(y_true, y_pred, target_names=class_indices.keys())
    print(report)
    with open("../plots/classification_report.txt", "w") as f:
        f.write(report)
    save_report_as_image(report, "../static/vis/cnn_mob_clf_rpt.png")

    # --- 3. ROC Curve ---
    y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 6))
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {list(class_indices.keys())[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("../static/vis/cnn_mob_roc_curve.jpg")
    plt.close()

    # --- 4. Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=f"{list(class_indices.keys())[i]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("../static/vis/cnn_mob_prc_curve.jpg")
    plt.close()

    # --- 5. Training vs Validation Accuracy ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label="Training Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("../static/vis/cnn_mob_train_val_accuracy.jpg")
    plt.close()

    # --- 6. Training vs Validation Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("../static/vis/cnn_mob_train_val_loss.jpg")
    plt.close()




def load_and_predict(model_path, image_path, target_size=(224, 224), class_names=None):
    """
    Loads a CNN model and predicts the class of a given X-ray image.

    Parameters:
    - model_path (str): Path to the saved model file (e.g., .h5).
    - image_path (str): Path to the input X-ray image.
    - target_size (tuple): Input size expected by the model, e.g., (224, 224).
    - class_names (list): List of class labels in the same order as the model outputs.

    Returns:
    - prediction_label (str): Predicted class label.
    - confidence (float): Confidence score for the predicted class.
    """
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the image
    image = load_img(image_path, target_size=target_size, color_mode='rgb')
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    # Resolve class label
    if class_names is None:
        class_names = ['normal', 'doubtful', 'mild', 'moderate', 'severe']
    prediction_label = class_names[predicted_class_index]

    # Display result
    plt.imshow(image)
    plt.title(f"Prediction: {prediction_label} ({confidence * 100:.2f}%)")
    plt.axis('off')
    plt.savefig('static/predictions/result.jpg')

    return prediction_label, confidence

def load_and_predict_new(model_path, image_path, target_size=(224, 224), class_names=None):
    """
    Loads a CNN model and predicts the class of a given X-ray image.

    Parameters:
    - model_path (str): Path to the saved model file (e.g., .h5).
    - image_path (str): Path to the input X-ray image.
    - target_size (tuple): Input size expected by the model, e.g., (224, 224).
    - class_names (list): List of class labels in the same order as the model outputs.

    Returns:
    - prediction_label (str): Predicted class label.
    - confidence (float): Confidence score for the predicted class.
    """
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the image
    image = load_img(image_path, target_size=target_size, color_mode='rgb')
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    # Resolve class label
    if class_names is None:
        class_names = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
    prediction_label = class_names[predicted_class_index]

    # Display result
    plt.imshow(image)
    plt.title(f"Prediction: {prediction_label}")
    plt.axis('off')
    plt.savefig('static/predictions/result.jpg')

    return prediction_label, confidence

def TestModel():

    model_path = 'models/cnn_mobilenet_knee_model.h5'
    image_path = 'static/uploads/test.jpg'
    prediction_label, confidence=load_and_predict_new(model_path, image_path)
    return prediction_label, confidence

#createModel()




