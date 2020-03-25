from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def getImages(path, classes):
    folder = os.listdir(path)
    classes_counter = 0
    images = []
    images_classes = []

    for x in range (0,len(folder)):
        myPicList = os.listdir(path+"/"+ str(classes[classes_counter]))
        for pic in myPicList:
            img_path = path+"/" + str(classes[classes_counter]) + "/" + pic
            img = cv2.imread(img_path)
            print('Image shape ' + img_path + ' => ' + str(img.shape))
            images.append(img)
            images_classes.append(classes_counter)
        classes_counter +=1

    images_data = np.array(images)
    images_data = images_data.astype('float32')
    images_data /= 255

    return images_data, images_classes
 
def createModel(classes, images_dimension):
    classes_amount = len(np.unique(classes))
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=images_dimension))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_amount, activation='softmax'))
 
    return model



images_dimension=(26,26,3)
labels = [0,1,2]
images, images_classes = getImages('training-images', labels)

X_train, X_test, Y_train, Y_test = train_test_split(images, images_classes, test_size=0.2)  # if 1000 images split will 200 for testing
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2) # if 1000 images 20% of remaining 800 will be 160 for validation

model = createModel(labels, images_dimension)
batch_size = 20
epochs = 100
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_validation, Y_validation))
model.evaluate(X_test,Y_test,verbose=0)