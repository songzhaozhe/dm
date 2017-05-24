# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json
from sklearn.preprocessing import scale
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Data sets

path = "./data/m0000"
save_path = "./models/"
input_size = 1

class dataset():
    def __init__(self, data, target):
        self.data = data
        self.target = target

def load_dataset():
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith(".out"):
            files.append(os.path.join(path, file_name))
    files.sort()
    train_list = []
    test_list = []
    all_list = []
    for i in range(len(files)):
        tmp = np.load(files[i])
        #print(tmp.shape)
        all_list.append(tmp)

    print("concatenating matrix....")
    all_matrix = np.concatenate(all_list, axis=0)
    tot_row = all_matrix.shape[0]
    print("slicing matrix...")
    all_data = all_matrix[:,-2:-1]
    all_target = all_matrix[:,-1]
    print(np.mean(all_data, axis = 0))

    all_data = scale(all_data, axis = 0)


    train_set = dataset(all_data[0:int(tot_row*0.7)], all_target[0:int(tot_row*0.7)])
    test_set = dataset(all_data[int(tot_row*0.7):], all_target[int(tot_row*0.7):])

    return train_set, test_set

model = Sequential()
model.add(Dense(32, input_dim = input_size, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
adam = Adam(lr = 0.001)

model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
train_set, test_set = load_dataset()
model.fit(train_set.data, train_set.target , epochs=2, batch_size=128)

score = model.evaluate(test_set.data, test_set.target, batch_size=128)
print(score)

save_file = os.path.join(path, 'model.json')
save_weights_file = os.path.join(path, 'model.h5')
model_json = model.to_json()
with open(save_file, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(save_weights_file)
print("Saved model to disk")
# print('accuracy on test set: %.2f%%' % )