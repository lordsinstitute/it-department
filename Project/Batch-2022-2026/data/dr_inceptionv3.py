import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from itertools import cycle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def Load_Images(path):
    folders = os.listdir(path)
    data = []
    label = []
    for i in folders:
        c = 0
        images = os.listdir(path + '/' + i)
        for j in images:
            if c <= 1000:
                im = Image.open(path + '/' + i + '/' + j)
                ar = np.array(im)
                data.append(ar)
                label.append(i)
                c = c + 1
    return np.array(data), label

def match_histograms(image, reference):
    # Convert the images to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)

    # Split the LAB images into channels
    l, a, b = cv2.split(image_lab)
    l_ref, a_ref, b_ref = cv2.split(reference_lab)

    # Apply histogram matching on the L channel
    l_matched = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)

    # Merge the matched L channel with the original A and B channels
    matched_lab = cv2.merge((l_matched, a, b))

    # Convert the LAB image back to RGB color space
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

    return matched_rgb




def create_model():

    x, y = Load_Images('../colored_images/')
    x.shape, len(y)

    target = pd.Series(y, dtype='category')

    for i in range(len(x)):
        # Convert the image to LAB color space
        lab = cv2.cvtColor(x[i], cv2.COLOR_RGB2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge the CLAHE-enhanced L channel with the original A and B channels
        limg = cv2.merge((cl, a, b))

        # Convert the LAB image back to RGB color space
        x[i] = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Assuming x is your image array
    # Reference image for histogram matching (you can choose any image from your dataset)
    reference_image = x[72]



    # Loop through each image in the array
    for i in range(len(x)):
        x[i] = match_histograms(x[i], reference_image)

    clahe_output = np.power(x[2600] / 255.0, 1) * 255
    plt.imshow(clahe_output)
    plt.show()
    plt.imshow(x[2600])
    kernel = np.array([[0, -0.01, 0],
                       [-0.01, 3, -0.01],
                       [0, -0.01, 0]])

    plt.show()
    sharpened_image = cv2.filter2D(clahe_output, -1, kernel)
    plt.imshow(sharpened_image)
    plt.show()
    kernel_closing = np.ones((2, 2), np.uint8)  # Adjust the kernel size as needed
    closed_image = cv2.morphologyEx(sharpened_image, cv2.MORPH_CLOSE, kernel_closing)

    # Display the closed image
    plt.imshow(closed_image)
    plt.show()
    print(closed_image.shape)
    _, thresholded_image = cv2.threshold(closed_image, 200, 255, cv2.THRESH_BINARY)
    plt.imshow(thresholded_image)
    plt.show()

    for i in range(len(x)):
        x[i] = np.power(x[i] / 255.0, 1) * 255
        kernel = np.array([[0, -0.01, 0],
                           [-0.01, 3, -0.01],
                           [0, -0.01, 0]])

        x[i] = cv2.filter2D(x[i], -1, kernel)
        kernel_closing = np.ones((2, 2), np.uint8)  # Adjust the kernel size as needed
        x[i] = cv2.morphologyEx(x[i], cv2.MORPH_CLOSE, kernel_closing)
        _, x[i] = cv2.threshold(x[i], 200, 255, cv2.THRESH_BINARY)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    #x_train.shape, x_test.shape, y_train.shape, y_test.shape

    if len(x_train.shape) == 4:
        # Flatten each image into a one-dimensional array
        x_train_resampled = x_train.reshape(x_train.shape[0], -1)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_resampled, y_train)

    original_shape = x_train.shape[1:]  # Assuming the original shape is (height, width, channels)
    print(original_shape)
    x_train_resampled = x_train_resampled.reshape(-1, *original_shape)

    # Normalize pixel values to be between 0 and 1
    x_train_resampled = x_train_resampled / 255.0
    x_test = x_test / 255.0

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_resampled)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert labels to one-hot encoding
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded)

    # Preprocess input for InceptionV3
    x_train_resampled = preprocess_input(x_train_resampled)
    x_test = preprocess_input(x_test)

    # Load InceptionV3 base model (excluding top layers)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=original_shape)
    base_model.trainable = False  # Freeze convolutional base

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(5, activation='softmax')(x)

    # Create full model
    model_inception = Model(inputs=base_model.input, outputs=predictions)
    model_inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint("../models/best_inceptionv3_dr.h5", save_best_only=True)
    ]

    # Train the model
    history = model_inception.fit(
        x_train_resampled, y_train_one_hot,
        epochs=25,
        validation_data=(x_test, y_test_one_hot),
        callbacks=callbacks
    )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], marker='*')
    plt.plot(history.history['val_loss'], marker='*')
    plt.title('InceptionV3 Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    print()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], marker='*')
    plt.plot(history.history['val_accuracy'], marker='*')
    plt.title('InceptionV3 Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    print()
    plt.savefig('../static/vis/inv3_acc.jpg')

    y_pred_probs = model_inception.predict(x_test)

    # Convert probabilities to one-hot encoded predictions
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Convert one-hot encoded labels back to categorical labels for the test set
    y_test_labels = np.argmax(y_test_one_hot, axis=1)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred)
    class_labels = [0, 1, 2, 3, 4]
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('InceptionV3 Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../static/vis/inv3_cnfmat.jpg')

    print()

    # Generate classification report
    class_report = classification_report(y_test_labels, y_pred,output_dict=True)
    print("\nClassification Report:")
    print(class_report)
    plt.figure(figsize=(8, 4))
    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("InceptionV3 Classification Report")
    plt.tight_layout()
    plt.savefig('../static/vis/inv3_clfrpt.jpg')
    plt.close()


    # Get predicted probabilities
    y_score = model_inception.predict(x_test)

    # Binarize the true labels
    n_classes = y_score.shape[1]
    y_test_bin = label_binarize(y_test_encoded, classes=list(range(n_classes)))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    plt.figure(figsize=(10, 8))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {label_encoder.inverse_transform([i])[0]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Inception V3 Multiclass ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.savefig('../static/vis/inv3_roc_curve.jpg')
    #plt.show()

    #model_cnn.save("../models/cnn_dr.h5")

create_model()


