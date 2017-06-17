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
from keras.models import model_from_json, load_model
from sklearn.preprocessing import scale
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, TensorBoard, CSVLogger

path = "./data/m0000"
save_path = "./models/"
input_size = 8
time_step = 20
epoch_num = 500
batch_size = 512

class indexset():
	def __init__(self,file_index, row_index):
		self.file_index = file_index
		self.row_index = row_index

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
		feature_list.append(all_data[index:index+len_list[i],:input_size])
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

model_path = os.path.join(save_path, "weights.390-0.6015.h5")
model = load_model(model_path)

index_list, feature_list, label_list = prepare_data_for_generator(test_files)
epochstep = int(len(index_list)/batch_size)
score = model.evaluate_generator(data_generator(index_list, feature_list, label_list,batch_size,shuffle=False), epochstep)
print(score)