# Function to load and preprocess an image for predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
IMAGE_SIZE = (350, 350)

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

def test_model():
    # Load, preprocess, and predict the class of an image
    model = load_model('../models/trained_lung_cancer_model.h5')
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_folder = '../Lung Cancer Dataset/train'
    test_folder = '../Lung Cancer Dataset/test'

    test_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=IMAGE_SIZE,
        batch_size=8,
        class_mode='categorical'
    )

    img_path = '../Lung Cancer Dataset/test/adenocarcinoma/000108 (3).png'
    img = load_and_preprocess_image(img_path, IMAGE_SIZE)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = list(test_generator.class_indices.keys())
    predicted_label = class_labels[predicted_class]

    print(f"The image belongs to class: {predicted_label}")
    
    # Display the image with the predicted class
    # plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
    # plt.title(f"Predicted: {predicted_label}")
    # plt.axis('off')
    # plt.show()
    
    # Repeat the process for additional images
    img_path = '../Lung Cancer Dataset/test/large.cell.carcinoma/000115 (2).png'
    img = load_and_preprocess_image(img_path, IMAGE_SIZE)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = list(test_generator.class_indices.keys())
    predicted_label = class_labels[predicted_class]
    print(f"The image belongs to class: {predicted_label}")
    # plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
    # plt.title(f"Predicted: {predicted_label}")
    # plt.axis('off')
    # plt.show()
    
    img_path = '../Lung Cancer Dataset/test/normal/6.png'
    img = load_and_preprocess_image(img_path, IMAGE_SIZE)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = list(test_generator.class_indices.keys())
    predicted_label = class_labels[predicted_class]
    print(f"The image belongs to class: {predicted_label}")
    # plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
    # plt.title(f"Predicted: {predicted_label}")
    # plt.axis('off')
    # plt.show()

    img_path = '../Lung Cancer Dataset/test/squamous.cell.carcinoma/000111.png'
    img = load_and_preprocess_image(img_path, IMAGE_SIZE)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = list(test_generator.class_indices.keys())
    predicted_label = class_labels[predicted_class]
    print(f"The image belongs to class: {predicted_label}")
    # plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
    # plt.title(f"Predicted: {predicted_label}")
    # plt.axis('off')
    # plt.show()

#test_model()

