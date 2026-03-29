# Import necessary libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
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
    plt.title("Classification Report")
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


def create_model():
    # Define paths to the training, validation, and test datasets
    train_folder = '../Lung Cancer Dataset/train'
    test_folder = '../Lung Cancer Dataset/test'
    validate_folder = '../Lung Cancer Dataset/valid'

    # Define paths to the specific classes within the train dataset
    normal_folder = '/normal'
    adenocarcinoma_folder = '/adenocarcinoma'
    large_cell_carcinoma_folder = '/large.cell.carcinoma'
    squamous_cell_carcinoma_folder = '/squamous.cell.carcinoma'

    # Load and preprocess data
    IMAGE_SIZE = (350, 350)
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # Define the batch size for training
    batch_size = 8

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=IMAGE_SIZE,
        batch_size=8,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        validate_folder,
        target_size=IMAGE_SIZE,
        batch_size=8,
        class_mode='categorical',
        shuffle=False

    )

    # Get class indices and labels
    class_indices = train_generator.class_indices  # e.g., {'normal': 0, 'doubtful': 1, ...}
    classes = list(class_indices.values())  # [0, 1, 2, 3, 4]

    # Get class labels for each sample
    train_labels = train_generator.classes
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

    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=2, factor=0.5, min_lr=0.000001)
    early_stops = EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=2, mode='auto')
    checkpointer = ModelCheckpoint(filepath='../models/best_inc_lc_model.hdf5', verbose=2, save_best_only=True, save_weights_only=True)
    checkpoint_cb = ModelCheckpoint('../models/inc_lc_model.h5', save_best_only=True, monitor='val_accuracy',
                                    mode='max', verbose=1)

    # Define the number of output classes
    OUTPUT_SIZE = 4
    validation_steps = validation_generator.samples

    # Load a pre-trained model (Xception) without the top layers and freeze its weights
    #pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model = InceptionV3(include_top=False, weights='imagenet', input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = False

    # Create a new model with the pre-trained base and additional layers for classification
    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(OUTPUT_SIZE, activation='softmax'))

    print("Pretrained model used:")
    pretrained_model.summary()

    print("Final model created:")
    model.summary()

    # Compile the model with an optimizer, loss function, and evaluation metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the training and validation data generators
    history = model.fit(
        train_generator,
        steps_per_epoch=25,
        epochs=50,
        callbacks=[learning_rate_reduction, early_stops, checkpointer,checkpoint_cb],
        validation_data=validation_generator,
        class_weight=class_weight_dict
    )

    # Get true labels and predictions from validation generator
    validation_generator.reset()
    y_true = []
    y_pred_probs = []

    for _ in range(len(validation_generator)):
        X_batch, y_batch = validation_generator.next()
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred_probs.extend(model.predict(X_batch))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Final training accuracy =", history.history['accuracy'][-1])
    print("Final testing accuracy =", history.history['val_accuracy'][-1])

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_indices.keys(),
                yticklabels=class_indices.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("../static/vis/inc_cnf.jpg")
    plt.close()

    # --- 2. Classification Report ---
    report = classification_report(y_true, y_pred, target_names=class_indices.keys())
    print(report)
    # with open("../plots/classification_report.txt", "w") as f:
    # f.write(report)
    save_report_as_image(report, "../static/vis/inc_clfrpt.png")

    # --- 3. ROC Curve ---
    y_true_bin = label_binarize(y_true, classes=np.arange(OUTPUT_SIZE))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 6))
    for i in range(OUTPUT_SIZE):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {list(class_indices.keys())[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("../static/vis/inc_roc.jpg")
    plt.close()

    # --- 4. Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    for i in range(OUTPUT_SIZE):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=f"{list(class_indices.keys())[i]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("../static/vis/inc_prc.jpg")
    plt.close()

    # --- 5. Training vs Validation Accuracy ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label="Training Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("../static/vis/inc_train_val_acc.jpg")
    plt.close()

    # --- 6. Training vs Validation Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("../static/vis/inc_train_val_loss.jpg")
    plt.close()



    # Save the trained model
    model.save('../models/effnet_trained_lung_cancer_model.h5')

create_model()