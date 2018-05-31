import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import os
data = pd.read_csv("./data/driving_log.csv")
X = data[['center', 'left', 'right']].values
y = data['steering'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)

def choose(center, left, right, angle):
    choice = np.random.choice(3)
    if choice == 0:
        return imageo(left), angle + 0.2
    elif choice == 1:
        return imageo(right), angle - 0.2
    return imageo(center), angle


def imageo(file):
    return mpimg.imread("./data/"+file.strip())



def translate(image, angle, rangex, rangey):
    transx = rangex * (np.random.rand() - 0.5)
    transy = rangey * (np.random.rand() - 0.5)
    angle += transx * 0.002
    transm = np.float32([[1, 0, transx], [0, 1, transy]])
  
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, transm, (width, height))
    return image, angle


def flip(image, angle):
  
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = - angle
    return image, angle

def preprocess(image):
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def shadow(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def augument(center, left, right, angle, rangex=100, rangey=10):
    image, angle = choose( center, left, right, angle)
    image, angle = flip(image, angle)
    image, angle = translate(image, angle, rangex, rangey)
    image = shadow(image)
    image = brightness(image)
    return image, angle



def brightness(image):
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def generator(image_paths, angles, batch_size, training):
    images = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            angle = angles[index]
            # argumentation
            if training and np.random.rand() < 0.6:
                image, angle = augument(center, left, right, angle)
            else:
                image = imageo(center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = angle
            i = i + 1
            if i == batch_size:
                break
        yield images, steers

import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Cropping2D,Flatten,Dense, Conv2D, MaxPooling2D, Dropout, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(66,200,3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2,2),  init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.1))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2),  init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2),  init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, 3, activation='elu',  init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, 3, activation='elu',  init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1,activation='linear', init='he_normal'))
model.summary()

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

model.compile(loss='mse', optimizer=Adam(lr=1e-3))
model.fit_generator(generator( X_train, y_train, 64, True), 24000,
validation_data=generator(X_valid, y_valid, 64, False), nb_val_samples=1024, nb_epoch=28, verbose = 1)


import json

model.save_weights('model.h5')
    # Save model architecture as json file
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
