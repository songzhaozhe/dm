#!/usr/bin/python

from __future__ import division
import os
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
input_size = 8
input_time = 5
path = "./data/m0000"

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
        tmp_new = np.zeros([tmp.shape[0]-input_time+1, input_size * input_time + 1])
        for j in range(tmp.shape[0]- input_time +1):
            for k in range(input_time):
                tmp_new [j, k*input_size:k*input_size+input_size] = tmp[j+k, :input_size]
            tmp_new[j,-1] = tmp[j+input_time-1, -1]
        #print(tmp.shape)
        all_list.append(tmp_new)

    print("concatenating matrix....")
    all_matrix = np.concatenate(all_list, axis=0)
    tot_row = all_matrix.shape[0]
    print("slicing matrix...")
    all_data = all_matrix[:,:input_size]
    all_target = all_matrix[:,-1]
    print(np.mean(all_target, axis = 0))
    for i in range(all_target.shape[0]):
        if (all_target[i] > 1):
            print("originally warning")
    all_data = scale(all_data, axis = 0)


    train_set = dataset(all_data[0:int(tot_row*0.7)], all_target[0:int(tot_row*0.7)])
    test_set = dataset(all_data[int(tot_row*0.7):], all_target[int(tot_row*0.7):])

    return train_set, test_set
[train, test] = load_dataset()


train_X = train.data
train_Y = train.target
print(train_X.shape)
print(train_Y.shape)
for i in range(train_Y.shape[0]):
    if train_Y[i] > 1:
        print("warning! %d"%train_Y[i])

test_X = test.data
test_Y = test.target

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 12
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 2

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 15
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# # do the same thing again, but output probabilities
# param['objective'] = 'multi:softprob'
# bst = xgb.train(param, xg_train, num_round, watchlist)
# # Note: this convention has been changed since xgboost-unity
# # get prediction, this is in 1D array, need reshape to (ndata, nclass)
# pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 2)
# pred_label = np.argmax(pred_prob, axis=1)
# error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
# print('Test error using softprob = {}'.format(error_rate))