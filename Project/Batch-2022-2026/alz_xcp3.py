import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def save_report_as_image(report, image_path):
    fig = plt.figure(figsize=(10, 5))
    plt.text(0.01, 0.5, report, fontsize=12, family='monospace')
    plt.axis('off')
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(conf_matrix, file_name, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def create_model():
    # Paths
    train_folder = '../Combined Dataset/train'
    validate_folder = '../Combined Dataset/test'
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 8
    OUTPUT_SIZE = 4

    # Data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        validate_folder,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_indices = train_generator.class_indices
    class_labels = list(class_indices.keys())

    # Compute class weights
    train_labels = train_generator.classes
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6)
    checkpoint = ModelCheckpoint('../models/best_xcp_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

    # Load MobileNetV2 and fine-tune
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Build model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(OUTPUT_SIZE, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Train
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Evaluate
    validation_generator.reset()
    y_true = validation_generator.classes
    y_pred_probs = model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    model.save('../models/final_xcp_model.h5')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, "../static/vis/xcp_cnf.jpg", class_labels)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    save_report_as_image(report, "../static/vis/xcp_clfrpt.png")

    # ROC Curve
    y_true_bin = label_binarize(y_true, classes=np.arange(OUTPUT_SIZE))
    plt.figure(figsize=(8, 6))
    for i in range(OUTPUT_SIZE):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_labels[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("../static/vis/xcp_roc.jpg")
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for i in range(OUTPUT_SIZE):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=class_labels[i])

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("../static/vis/xcp_prc.jpg")
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label="Train Acc")
    plt.plot(history.history['val_accuracy'], label="Val Acc")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("../static/vis/xcp_train_val_acc.jpg")
    plt.close()

    # Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../static/vis/xcp_train_val_loss.jpg")
    plt.close()


# Run the model
create_model()
