import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, Reshape, Activation, ZeroPadding2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
#from PIL import Image
import os
#import imageio

# Generate dummy data
import numpy as np

import tensorflow as tf
print(tf.test.is_built_with_cuda())

#data_input=keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
#input_training_generator=data_input.flow_from_directory('data/inputset',subset='training',target_size=(27,8))
#input_validation_generator=data_input.flow_from_directory('data/inputset',subset='validation',target_size=(27,8))
#data_output=keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
#output_training_generator=data_output.flow_from_directory('data/outputset',subset='training',target_size=(27,8))
#output_validation_generator=data_output.flow_from_directory('data/outputset',subset='validation',target_size=(27,8))

#input_training_generator = input_training_generator.reshape(87,648)
#input_validation_generator = input_validation_generator.reshape(21,648)
#output_training_generator = output_training_generator.reshape(87,216)
#output_validation_generator = output_validation_generator.reshape(21,216)

data_generator = ImageDataGenerator(validation_split=0.2,
                               rotation_range=15,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               rescale=1. / 255)

train_set_size=83457
val_set_size=20864
batch_size=16

def generator_training_inout(generator, dirIn, dirOut, batch_size, img_height, img_width):
    genI = generator.flow_from_directory(dirIn, subset='training',
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)

    genO = generator.flow_from_directory(dirOut, subset='training',
                                          target_size=(img_height, img_width),
                                          color_mode="grayscale",
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)
    while True:
        X = genI.next()
        Y = genO.next()
        yield X[0], Y[0]

def generator_validation_inout(generator, dirIn, dirOut, batch_size, img_height, img_width):
    genI = generator.flow_from_directory(dirIn, subset='validation',
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)

    genO = generator.flow_from_directory(dirOut, subset='validation',
                                          target_size=(img_height, img_width),
                                          color_mode="grayscale",
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)
    while True:
        X = genI.next()
        Y = genO.next()
        yield X[0], Y[0]


training_generator = generator_training_inout(data_generator,'data/input','data/output',batch_size,144,256)
validation_generator = generator_validation_inout(data_generator,'data/input','data/output',batch_size,144,256)

#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

print("image loaded")

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(144, 256, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))


model.add(ZeroPadding2D(((1, 1), (1, 1))))
model.add(Conv2DTranspose(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D(((1, 1), (1, 1))))
model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(ZeroPadding2D(((1, 1), (1, 1))))
model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Conv2D(1, (3, 3), activation='relu'))
model.add(Cropping2D(cropping=((0, 1), (0, 1))))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.01, decay=1e-4)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

print(model.summary())
mycallback = keras.callbacks.ModelCheckpoint(filepath = 'models/mymodel3_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv_logger = keras.callbacks.CSVLogger('mymodel3.log',append=True)

#score = model.evaluate(np.array(input_validation_generator), np.array(output_validation_generator), batch_size=32)
if os.path.exists('models/mymodel3_03-0.05.h5'):
    model = keras.models.load_model("models/mymodel3_03-0.05.h5")

if os.path.exists('mymodel3.h5'):
    model=keras.models.load_model("mymodel3.h5")
else:
    model.fit_generator(
		training_generator,
        steps_per_epoch=train_set_size // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=val_set_size // batch_size,
        callbacks=[mycallback,csv_logger])
    model.save('models/mymodel3.h5')


img_path = "data/input/scene/ims_10002196_93457.png"
img = image.load_img(img_path, target_size=(144, 256))
#print(type(img))

x = image.img_to_array(img)/255.
#print(type(x))
#print(x.shape)
plt.imshow(x)
plt.show()

x = np.expand_dims(x, axis=0)
#print(x.shape)


img_depth = model.predict(x).reshape(144,256)
print(img_depth)

plt.imshow((img_depth))
plt.show()