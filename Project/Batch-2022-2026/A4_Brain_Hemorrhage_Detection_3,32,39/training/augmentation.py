import numpy as np
import pandas as pd
import os
#import skimage.io
import cv2

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#folder = '../head_ct/TRAIN/NO'
folder = '../head_ct/TRAIN/YES'
loaded = load_images_from_folder(folder)

loaded_new = []
for i in range(80):
    # print(loaded[i].shape)
    res = cv2.resize(loaded[i], dsize=(227, 227), interpolation=cv2.INTER_CUBIC)
    loaded_new.append(res)

loaded_new = np.array(loaded_new)

#path = '../head_ct/AUGMENTED/TRAIN/NO/'
path = '../head_ct/AUGMENTED/TRAIN/YES/'
os.makedirs(path, exist_ok=True)

datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='constant')

x = loaded_new

i = 0
for batch in datagen.flow(x, batch_size=81, save_to_dir=path, save_prefix='aug', save_format='png'):
    i += 1
    if i > 10:
        break

aug_img = load_images_from_folder(path)



