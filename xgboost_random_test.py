#!/usr/bin/python

from __future__ import division
import os
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
input_size = 8
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
        #print(tmp.shape)
        all_list.append(tmp)
        for j in range(tmp.shape[0]):
            if (tmp[j,-1] > 1):
                print(files[i],j)

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
    #all_data = scale(all_data, axis = 0)


    train_X, test_X, train_Y, test_Y = train_test_split(all_data, all_target, test_size=0.30, random_state=42)

    return train_X, train_Y, test_X, test_Y
[train_X, train_Y, test_X, test_Y] = load_dataset()


xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.05
param['max_depth'] = 20
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 2

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 20
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 2)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softprob = {}'.format(error_rate))