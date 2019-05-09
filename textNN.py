import keras
import random
import numpy as np
from keras.preprocessing.text import text_to_word_sequence,hashing_trick
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Activation
from keras.utils import to_categorical
from keras.optimizers import SGD


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

dir1 = "data/training/useful.txt"
dir2 = "data/training/useless.txt"
dir3 = "data/validation/useful.txt"
dir4 = "data/validation/useless.txt"

train_set_size=file_len(dir1)+file_len(dir2)
val_set_size=file_len(dir3)+file_len(dir4)

f1=open(dir1,"r")
f2=open(dir2,"r")

text=f1.read()
words = set(text_to_word_sequence(text))
vocab_size1 = len(words)

text=f2.read()
words = set(text_to_word_sequence(text))
vocab_size2 = len(words)

vocab_size = max(vocab_size1,vocab_size2)

print(vocab_size)

f1.close()
f2.close()

def generator_inout(d1, d2, cat1, cat2, batch_size, vocab):
    c1=True
    c2=True
    file1 = open(d1, "r")
    file2 = open(d2, "r")
    while True:
        batch_input = []
        batch_output = []
        for i in range(0,batch_size):
            if c1 and c2:
                if random.randint(0,1)>0:
                    text=file1.readline()
                    cat=cat1
                    if text == "" :
                        c1=False
                else:
                    text = file2.readline()
                    cat = cat2
                    if text == "":
                        c2 = False
            else:
                if c1:
                    text = file1.readline()
                    cat = cat1
                    if text == "":
                        c1 = False
                if c2:
                    text = file2.readline()
                    cat = cat2
                    if text == "":
                        c2 = False
            if not(c1) and not(c2):
                file1.close()
                file2.close()
                file1 = open(d1, "r")
                file2 = open(d2, "r")
                c1=True
                c2=True

            input = hashing_trick(text, round(vocab * 1.3), hash_function='md5')
            output = cat

            batch_input += [input]
            batch_output += [output]

        batch_x = pad_sequences(np.array(batch_input),maxlen=14)
        batch_y = to_categorical(np.array(batch_output),num_classes=2)

        yield batch_x, batch_y

training_generator = generator_inout(dir1,dir2,1,0,32,vocab_size)
validation_generator = generator_inout(dir3,dir4,1,0,32,vocab_size)

#print(next(training_generator))

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

print("model compiled")


model.fit_generator(
    training_generator,
    steps_per_epoch=train_set_size // 32,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=val_set_size // 32)

model.save('thought_model.h5')

"""testcase = ""
input = hashing_trick(testcase, round(vocab_size * 1.3), hash_function='md5')
readyinp = pad_sequences(np.array([input]),maxlen=14)

output = model.predict_classes(readyinp)
prediction = output[0]
print("Class: ",prediction)"""





