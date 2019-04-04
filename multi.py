import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Activation
from keras.optimizers import SGD
#from PIL import Image
#import os
#import imageio

# Generate dummy data
import numpy as np

data_input=keras.preprocessing.image.ImageDataGenerator(validation_split=0.8)
input_training_generator=data_input.flow_from_directory('data/inputset',subset='training',target_size=(100,100))
input_validation_generator=data_input.flow_from_directory('data/inputset',subset='validation',target_size=(100,100))
data_output=keras.preprocessing.image.ImageDataGenerator(validation_split=0.8)
output_training_generator=data_output.flow_from_directory('data/outputset',subset='training',target_size=(100,100))
output_validation_generator=data_output.flow_from_directory('data/outputset',subset='validation',target_size=(100,100))

#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(10000, activation='relu', input_shape=(100,100,3)))
model.add(Dropout(0.1))
model.add(Dense(10000, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10000, activation='softmax'))
#model.add(Reshape((100, 100)))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(np.array(input_training_generator), np.array(output_training_generator),
          epochs=20,
          batch_size=32)
score = model.evaluate(np.array(input_validation_generator), np.array(output_validation_generator), batch_size=32)