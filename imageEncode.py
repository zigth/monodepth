import keras
import scipy.misc
from keras.models import Sequential,Model
from keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, Reshape, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from PIL import Image

#import imageio

# Generate dummy data
import numpy as np

data_input=keras.preprocessing.image.ImageDataGenerator(validation_split=0.8)
input_training_generator=data_input.flow_from_directory('data/inputset',subset='training',target_size=(27,8))
input_validation_generator=data_input.flow_from_directory('data/inputset',subset='validation',target_size=(27,8))
data_output=keras.preprocessing.image.ImageDataGenerator(validation_split=0.8)
output_training_generator=data_output.flow_from_directory('data/outputset',subset='training',target_size=(27,8))
output_validation_generator=data_output.flow_from_directory('data/outputset',subset='validation',target_size=(27,8))

print("images loaded")

#input_training_generator = to_categorical(input_training_generator)
#input_validation_generator = to_categorical(input_validation_generator)
#output_training_generator = to_categorical(output_training_generator)
#output_validation_generator = to_categorical(output_validation_generator)

print("image reshape 1")

#input_training_generator = input_training_generator.reshape(input_training_generator.shape[0],3,27,8)
#input_validation_generator = input_validation_generator.reshape(input_validation_generator.shape[0],3,27,8)
#output_training_generator = output_training_generator.reshape(output_training_generator.shape[0],1,27,8)
#output_validation_generator = output_validation_generator.reshape(output_validation_generator.shape[0],1,27,8)

print("image reshape 2")

input_layer = Input((1,28,28))

x=Conv2D(10,5,activation='relu')(input_layer)
x=MaxPooling2D(2)(x)
x=Conv2D(20,2,activation='relu')(x)
x=MaxPooling2D(2)(x)
encoded=x
x=UpSampling2D(2)(x)
x=Conv2DTranspose(20,2,activation='relu')(x)
x=UpSampling2D(2)(x)
x=Conv2DTranspose(10,5,activation='relu')(x)
x=Conv2DTranspose(1,3,activation='sigmoid')(x)

print("layer definition")

model=Model(input=input_layer, output=x)
model.summary()

print("model creation")

model.compile(loss='binary_crossentropy', optimizer='adam')

print("model compilation")

model.fit(input_training_generator,output_training_generator,batch_size=32, epochs=1, validation_data=(input_validation_generator,output_validation_generator))
model.save("testex.h5")

print("done")

