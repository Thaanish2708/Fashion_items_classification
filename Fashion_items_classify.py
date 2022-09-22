import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print(x_train)

# image_index = 88 # You may select anything up to 60,000
# print(y_train[image_index]) # The label is 8
# plt.imshow(x_train[image_index], cmap='Greys')

# print('Training data shape : ', x_train.shape, y_train.shape)

# print('Testing data shape : ', x_test.shape, y_test.shape)

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train[0:60000, 0:28 , 0:28 , 0:1]
y_train = y_train[0:60000,]
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

from sklearn.model_selection import train_test_split

x_train,valid_x,y_train,valid_y = train_test_split(x_train, y_train)

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,  BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
# Creating a Sequential Model and adding the layers
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
fashion_model.add(MaxPooling2D(pool_size=(2, 2)))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, kernel_size=(3,3)))
fashion_model.add(MaxPooling2D(pool_size=(2, 2)))
fashion_model.add(Dropout(0.25))
fashion_model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
fashion_model.add(Dense(128, activation=tf.nn.relu))
fashion_model.add(Dropout(0.4))
fashion_model.add(Dense(10,activation=tf.nn.softmax))

fashion_model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


fashion_train=fashion_model.fit(x_train, y_train, validation_data=(valid_x, valid_y),epochs=5)
 # network evaluation
test_eval=fashion_model.evaluate(x_test, y_test)   

# test sample data
image_index = 6890
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = fashion_model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
