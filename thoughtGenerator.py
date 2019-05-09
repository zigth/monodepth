import keras
from itertools import permutations
import numpy as np
from keras.preprocessing.text import text_to_word_sequence, hashing_trick
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Activation
from keras.optimizers import SGD

dir1 = "data/training/useful.txt"
dir2 = "data/training/useless.txt"

f1=open(dir1,"r")
f2=open(dir2,"r")

text=f1.read()
words = set(text_to_word_sequence(text))
vocab_size1 = len(words)

text=f2.read()
words = set(text_to_word_sequence(text))
vocab_size2 = len(words)

vocab_size = max(vocab_size1,vocab_size2)

model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(14,)))
model.add(Dropout(0.1))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.load_weights("thought_model.h5")

inp=input('enter a sentence: ')
words = inp.split()

perm = permutations(words)

for i in perm:
    preinp = hashing_trick(str(i), round(vocab_size * 1.3), hash_function='md5')
    readyinp = pad_sequences(np.array([preinp]),maxlen=14)
    output = model.predict_classes(readyinp)
    #print(output)
    if output[0]==0:
        print(i)
