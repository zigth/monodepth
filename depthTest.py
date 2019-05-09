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


if os.path.exists('models/mymodel.h5'):
    model=keras.models.load_model("models/mymodel.h5")



fig = plt.figure(figsize=(12,15))
		
img_path = "data/input/scene/ims_10002382_42314.png"
img = image.load_img(img_path, target_size=(144, 256))
#print(type(img))

x = image.img_to_array(img)/255.
#print(type(x))
#print(x.shape)
fig.add_subplot(3,1,1)
plt.imshow(x)
#plt.show()

x = np.expand_dims(x, axis=0)
#print(x.shape)

fig.add_subplot(3,1,2)
img_depth = model.predict(x).reshape(144,256)
print(img_depth)

plt.imshow(img_depth,cmap='gray')
#plt.show()

img_path = "data/output/depth/imd_10002382_42314.png"
img = image.load_img(img_path, target_size=(144, 256))
#print(type(img))

x = np.array(image.img_to_array(img)/255.)
#print(type(x))
#print(x.shape)
fig.add_subplot(3,1,3)
plt.imshow(x,cmap='jet')
plt.show()