from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import cv2 as cv2
from keras.utils import to_categorical

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
print(ROOT_DIR)

# Paths
train_dir = os.path.join(ROOT_DIR, 'train')
validation_dir = os.path.join(ROOT_DIR, 'validation')

# Train data
dir_none = os.path.join(ROOT_DIR, train_dir, 'none')
dir_happy = os.path.join(ROOT_DIR, train_dir, 'smilie-happy')
dir_sad = os.path.join(ROOT_DIR, train_dir, 'smilie-sad')

amount_none = len(os.listdir(dir_none))
amount_happy = len(os.listdir(dir_happy))
amount_sad = len(os.listdir(dir_sad))

print('train amount none: ' + str(amount_none))
print('train amount happy: ' + str(amount_happy))
print('train amount sad: ' + str(amount_sad))

# Validation data
dir_none_validation = os.path.join(ROOT_DIR, validation_dir, 'none')
dir_happy_validation = os.path.join(ROOT_DIR, validation_dir, 'smilie-happy')
dir_sad_validation = os.path.join(ROOT_DIR, validation_dir, 'smilie-sad')

amount_none_validation = len(os.listdir(dir_none_validation))
amount_happy_validation = len(os.listdir(dir_happy_validation))
amount_sad_validation = len(os.listdir(dir_sad_validation))

print('validation amount none: ' + str(amount_none_validation))
print('validation amount happy: ' + str(amount_happy_validation))
print('validation amount sad: ' + str(amount_sad_validation))

# GET IMAGES
total_train = amount_none + amount_happy + amount_sad
total_val = amount_none_validation + amount_happy_validation + amount_sad_validation

batch_size = 64
epochs = 15
IMG_HEIGHT = 26
IMG_WIDTH = 26

IMGS = []
IMGS_CLASSES = []

IMGS_VAL = []
IMGS_CLASSES_VAL = []

def addImages(root_folder, class_name_folder, classes, imgs, imgs_classes):
    target_folder = os.path.join(ROOT_DIR, root_folder, class_name_folder)
    images = os.listdir(target_folder)

    for img_filename in images: 
        path = os.path.join(target_folder, img_filename)
        print(path)
        imgs.append(cv2.imread(path))
        imgs_classes.append(classes.index(class_name_folder))

# Classification categories
image_classes = ['none', 'smilie-happy', 'smilie-sad']

# Train
addImages('train','none', image_classes, IMGS, IMGS_CLASSES)
addImages('train','smilie-happy', image_classes, IMGS, IMGS_CLASSES)
addImages('train','smilie-sad', image_classes, IMGS, IMGS_CLASSES)

print('imgs: ' + str(len(IMGS)))
print('imgs_classes: ' + str(len(IMGS_CLASSES)))
print('imgs:classes: ' + str(IMGS_CLASSES))

# Validation
addImages('validation','none', image_classes, IMGS_VAL, IMGS_CLASSES_VAL)
addImages('validation','smilie-happy', image_classes, IMGS_VAL, IMGS_CLASSES_VAL)
addImages('validation','smilie-sad', image_classes, IMGS_VAL, IMGS_CLASSES_VAL)

print('imgs_val: ' + str(len(IMGS_VAL)))
print('imgs_classes_val: ' + str(len(IMGS_CLASSES_VAL)))

print('before: ' + str(IMGS[0]))

IMGS = np.asarray(IMGS)
IMGS = IMGS.astype('float32')
IMGS /= 255.0

IMGS_VAL = np.asarray(IMGS_VAL)
IMGS_VAL = IMGS_VAL.astype('float32')
IMGS_VAL /= 255.0

IMGS_CLASSES = np.asarray(IMGS_CLASSES)
IMGS_CLASSES_VAL = np.asarray(IMGS_CLASSES_VAL)


IMGS_CLASSES = to_categorical(IMGS_CLASSES, len(image_classes))
IMGS_CLASSES_VAL = to_categorical(IMGS_CLASSES_VAL, len(image_classes))




# Create model
model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH , len(image_classes))),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])


# Compile model

print("======================")
print(IMGS_VAL)

model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_new.fit(IMGS, IMGS_CLASSES, epochs=100, batch_size=32, verbose=1, validation_split=0.2, validation_data=(IMGS_VAL, IMGS_CLASSES_VAL))
test_results = model_new.evaluate(IMGS_VAL, IMGS_CLASSES_VAL, verbose=1)

model_new.save('model.h5')

print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')


# Visualize the model
'''
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Generator for our training data
train_image_generator = ImageDataGenerator(rescale=1./255)

# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical') 
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])
'''

