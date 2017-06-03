# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Masking, Dropout
from keras.models import model_from_json
from sklearn.preprocessing import scale
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, TensorBoard, CSVLogger
#from visual_callbacks import AccLossPlotter
#model = Sequential()
#model.add(Dense(32, input_dim=784))
#model.add(Activation('relu'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# Data sets

path = "./data/m0000"
save_path = "./models_20_2/"
input_size = 8
time_step = 20
epoch_num = 500
batch_size = 512


class dataset():
    def __init__(self, data, target):
        self.data = data
        self.target = target

class indexset():
	def __init__(self,file_index, row_index):
		self.file_index = file_index
		self.row_index = row_index

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
    all_data = all_matrix[:,:-1]
    #print ('all_Data',len(all_data[0]))
    all_target = all_matrix[:,-1]
    #print(np.mean(all_data, axis = 0))

    all_data = scale(all_data, axis = 0)
    train_data = np.array(all_data[0:int(tot_row*0.7),:]).reshape(int(tot_row*0.7),time_step,input_size)
    train_target = np.array(all_target[0:int(tot_row*0.7)]).reshape(int(tot_row*0.7),time_step,1)
    test_data = np.array(all_data[int(tot_row*0.7):,:]).reshape(tot_row-int(tot_row*0.7),time_step,input_size)
    test_target = np.array(all_target[int(tot_row*0.7):]).reshape(tot_row-int(tot_row*0.7),time_step,1)

    train_set = dataset(train_data, train_target)
    test_set = dataset(test_data, test_target)
    return train_set, test_set

def prepare_data_for_generator(files):
	all_list = []
	len_list = []
	index_list = []
	feature_list = []
	label_list = []
	for i in range(len(files)):
		tmp = np.load(files[i])
        #print(tmp.shape)
		for j in range(len(tmp)-time_step+1):
			index_list.append(indexset(i,j))
		len_list.append(len(tmp))
		all_list.append(tmp[:,:-1])
		label_list.append(tmp[:,-1])
	all_matrix = np.concatenate(all_list, axis=0)
	#print("slicing matrix...")
	all_data = scale(all_matrix, axis = 0)
	#min_max_scaler = MinMaxScaler()   
	#all_data = min_max_scaler.fit_transform(all_data_scaled)
	index = 0
	for i in range(len(files)):
		feature_list.append(all_data[index:index+len_list[i],:])
		index = index + len_list[i]

	feature_list = np.array(feature_list)
	label_list = np.array(label_list)

	return index_list, feature_list, label_list

def data_generator(index_list, feature_list, label_list, batch_size = 128, shuffle = True):

    if shuffle:
    	random.shuffle(index_list)
    maxlen = len(index_list)
    current = 0
    batch_features = np.zeros([batch_size,time_step,input_size])
    batch_label = np.zeros([batch_size,1])
    while True:
    	#batch_features = []
    	#batch_label = []
    	if current + batch_size <= maxlen:
    		for i in range(current,current+batch_size):
    			fi = index_list[i].file_index
    			ri = index_list[i].row_index
    			batch_features[i%batch_size] = feature_list[fi][ri:ri+time_step,:]
    			batch_label[i%batch_size] = label_list[fi][ri+time_step-1]
    		current = current + batch_size
    	else:
    		nextbatch = current + batch_size - maxlen;
    		for i in range(current,maxlen):
    			fi = index_list[i].file_index
    			ri = index_list[i].row_index
    			batch_features[i%batch_size] = feature_list[fi][ri:ri+time_step,:]
    			batch_label[i%batch_size] = label_list[fi][ri+time_step-1]
    		if shuffle:
    			random.shuffle(index_list)
    		current = 0
    		for i in range(current, nextbatch):
    			fi = index_list[i].file_index
    			ri = index_list[i].row_index
    			batch_features[batch_size - nextbatch + i%batch_size] = feature_list[fi][ri:ri+time_step,:]
    			batch_label[batch_size - nextbatch + i%batch_size] = label_list[fi][ri+time_step-1]
    		current = nextbatch
    	#print(batch_features[0,0])
    	yield batch_features, batch_label



data_files = []
for file_name in os.listdir(path):
    if file_name.endswith(".out"):
        data_files.append(os.path.join(path, file_name))
data_files.sort()
train_len = int(len(data_files)*0.7)
train_files = data_files[:train_len]
test_files = data_files[train_len:]
model = Sequential()
model.add(Masking(mask_value= 0,	input_shape=(time_step, input_size)))
model.add(LSTM(16, input_shape = (time_step,input_size), activation = 'relu',return_sequences=True))
model.add(LSTM(32, activation = 'relu',return_sequences=True))
model.add(LSTM(64, activation = 'relu',return_sequences=True))
model.add(LSTM(32, activation = 'relu',return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
adam = Adam(lr = 0.001)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = History()
tensorboard = TensorBoard()
csv_logger = CSVLogger('./trainlog.csv')
#plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
checkpoint = ModelCheckpoint(os.path.join(save_path,'weights.{epoch:02d}-{acc:.4f}.h5'), monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
#train_set, test_set = load_dataset()
#model.fit(train_set.data, train_set.target , epochs=2, batch_size=128, shuffle = False)

index_list, feature_list, label_list = prepare_data_for_generator(train_files)
epochstep = int(3*len(index_list)/epoch_num/batch_size)
model.fit_generator(data_generator(index_list, feature_list, label_list,batch_size),epochs=epoch_num, steps_per_epoch=epochstep, callbacks=[history, tensorboard, csv_logger, checkpoint])
#score = model.evaluate(test_set.data, test_set.target, batch_size=128)
#print(score)

index_list, feature_list, label_list = prepare_data_for_generator(train_files)
epochstep = int(len(index_list)/batch_size)
score = model.evaluate_generator(data_generator(index_list, feature_list, label_list,batch_size,shuffle=False), epochstep)
print(score)

save_file = os.path.join(save_path, 'model.json')
save_weights_file = os.path.join(save_path, 'model.h5')
model_json = model.to_json()
with open(save_file, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(save_weights_file)
print("Saved model to disk")
# print('accuracy on test set: %.2f%%' % )
