# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import argparse

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Masking, Dropout
from keras.models import model_from_json
from sklearn.preprocessing import scale
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, TensorBoard, CSVLogger
import reg_models


parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model_name', type=str, default="LSTMModel", choices=['BaselineModel','LSTMModel', 'LogisticModel'],
                            help='model for training')
parser.add_argument('-dir', '--data_dir', type=str, default="./data/m0000",
                            help='path to the dataset')
parser.add_argument('-o', '--output_dir', type=str, default=None,
                            help='path to the output dir')
parser.add_argument('-ts', '--time_step', type=int, default=20,
                            help='how many ticks as input (useless for BaselineModel)')
parser.add_argument('-bs', '--batch_size', type=int, default=512,
                            help='batch size')
parser.add_argument('-tt', '--traverse_time', type=int, default=3,
                            help='how many times to traverse the train dataset')
parser.add_argument('-ep', '--epoch_num', type=int, default=50,
                            help='how many epoch')
parser.add_argument('-id', '--gpu_id', type=int, default=0,
                            help='how many epoch')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Data sets
model_name = args.model_name
path = args.data_dir
if args.output_dir != None:
    save_path = args.output_dir
else:
    save_path = "./model/"+model_name+"/test/"

input_size = 8
epoch_num = args.epoch_num
batch_size = args.batch_size
train_times = args.traverse_time
if model_name == "BaselineModel":
    one_feature = True
    time_step = 1
else:
    time_step = args.time_step


if not os.path.exists(save_path):
    os.makedirs(save_path)


class dataset():
    def __init__(self, data, target):
        self.data = data
        self.target = target


class indexset():
    def __init__(self, file_index, row_index):
        self.file_index = file_index
        self.row_index = row_index


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


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
        # print(tmp.shape)
        all_list.append(tmp)

    print("concatenating matrix....")
    all_matrix = np.concatenate(all_list, axis=0)
    tot_row = all_matrix.shape[0]
    print("slicing matrix...")
    all_data = all_matrix[:, :-1]
    # print ('all_Data',len(all_data[0]))
    all_target = all_matrix[:, -1]
    # print(np.mean(all_data, axis = 0))

    all_data = scale(all_data, axis=0)
    train_data = np.array(all_data[0:int(tot_row * 0.7), :]).reshape(int(tot_row * 0.7), time_step, input_size)
    train_target = np.array(all_target[0:int(tot_row * 0.7)]).reshape(int(tot_row * 0.7), time_step, 1)
    test_data = np.array(all_data[int(tot_row * 0.7):, :]).reshape(tot_row - int(tot_row * 0.7), time_step, input_size)
    test_target = np.array(all_target[int(tot_row * 0.7):]).reshape(tot_row - int(tot_row * 0.7), time_step, 1)

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
        # print(tmp.shape)
        for j in range(len(tmp) - time_step + 1):
            index_list.append(indexset(i, j))
        len_list.append(len(tmp))
        all_list.append(tmp[:, :-1])
        label_list.append(tmp[:, -1])
    all_matrix = np.concatenate(all_list, axis=0)
    # print("slicing matrix...")
    all_data = scale(all_matrix, axis=0)
    # min_max_scaler = MinMaxScaler()
    # all_data = min_max_scaler.fit_transform(all_data_scaled)
    index = 0
    for i in range(len(files)):
        feature_list.append(all_data[index:index + len_list[i], :input_size])
        index = index + len_list[i]

    feature_list = np.array(feature_list)
    label_list = np.array(label_list)

    return index_list, feature_list, label_list


def train_data_generator(index_list, feature_list, label_list, batch_size=128, shuffle=True):
    if shuffle:
        random.shuffle(index_list)
    maxlen = len(index_list)
    current = 0
    batch_features = np.zeros([batch_size, time_step, input_size])
    batch_label = np.zeros([batch_size, 1])
    while True:
        # batch_features = []
        # batch_label = []
        if current + batch_size <= maxlen:
            for i in range(current, current + batch_size):
                fi = index_list[i].file_index
                ri = index_list[i].row_index
                batch_features[i % batch_size] = feature_list[fi][ri:ri + time_step, :]
                batch_label[i % batch_size] = label_list[fi][ri + time_step - 1]
            current = current + batch_size
        else:
            nextbatch = current + batch_size - maxlen;
            for i in range(current, maxlen):
                fi = index_list[i].file_index
                ri = index_list[i].row_index
                batch_features[i % batch_size] = feature_list[fi][ri:ri + time_step, :]
                batch_label[i % batch_size] = label_list[fi][ri + time_step - 1]
            if shuffle:
                random.shuffle(index_list)
            current = 0
            for i in range(current, nextbatch):
                fi = index_list[i].file_index
                ri = index_list[i].row_index
                batch_features[batch_size - nextbatch + i % batch_size] = feature_list[fi][ri:ri + time_step, :]
                batch_label[batch_size - nextbatch + i % batch_size] = label_list[fi][ri + time_step - 1]
            current = nextbatch
        # print(batch_features[0,0])
        yield batch_features, batch_label


def test_data_generator(index_list, feature_list, label_list, batch_size=128, shuffle=True):
    if shuffle:
        random.shuffle(index_list)
    maxlen = len(index_list)
    current = 0
    batch_features = np.zeros([batch_size, time_step, input_size])
    batch_label = np.zeros([batch_size, 1])
    while True:
        # batch_features = []
        # batch_label = []
        if current + batch_size <= maxlen:
            for i in range(current, current + batch_size):
                fi = index_list[i].file_index
                ri = index_list[i].row_index
                batch_features[i % batch_size] = feature_list[fi][ri:ri + time_step, :]
                batch_label[i % batch_size] = label_list[fi][ri + time_step - 1]
            current = current + batch_size
        else:
            nextbatch = current + batch_size - maxlen;
            for i in range(current, maxlen):
                fi = index_list[i].file_index
                ri = index_list[i].row_index
                batch_features[i % batch_size] = feature_list[fi][ri:ri + time_step, :]
                batch_label[i % batch_size] = label_list[fi][ri + time_step - 1]
            if shuffle:
                random.shuffle(index_list)
            current = 0
            for i in range(current, nextbatch):
                fi = index_list[i].file_index
                ri = index_list[i].row_index
                batch_features[batch_size - nextbatch + i % batch_size] = feature_list[fi][ri:ri + time_step, :]
                batch_label[batch_size - nextbatch + i % batch_size] = label_list[fi][ri + time_step - 1]
            current = nextbatch
        # print(batch_features[0,0])
        yield batch_features, batch_label


data_files = []
for file_name in os.listdir(path):
    if file_name.endswith(".reg_out"):
        data_files.append(os.path.join(path, file_name))
data_files.sort()

model_instance = find_class_by_name(model_name, [reg_models])()
model = model_instance.create_model((time_step, input_size))

history = History()
tensorboard = TensorBoard()
csv_logger = CSVLogger('./trainlog.csv')
# plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
checkpoint = ModelCheckpoint(os.path.join(save_path, 'weights.{epoch:02d}-{acc:.4f}.h5'), monitor='val_acc', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', period=1)
# train_set, test_set = load_dataset()
# model.fit(train_set.data, train_set.target , epochs=2, batch_size=128, shuffle = False)

index_list, feature_list, label_list = prepare_data_for_generator(data_files)
train_len = int(len(index_list) * 0.7)
train_index_list = index_list[:train_len]
test_index_list = index_list[train_len:]

epochstep = int(train_times * len(train_index_list) / epoch_num / batch_size)
model.fit_generator(train_data_generator(train_index_list, feature_list, label_list, batch_size), epochs=epoch_num,
                    steps_per_epoch=epochstep, callbacks=[history, tensorboard, csv_logger, checkpoint])
# score = model.evaluate(test_set.data, test_set.target, batch_size=128)
# print(score)

epochstep = int(len(test_index_list) / batch_size)
score = model.evaluate_generator(
    test_data_generator(test_index_list, feature_list, label_list, batch_size, shuffle=False), epochstep)
print(score)

# save_file = os.path.join(save_path, 'model.json')
save_file = os.path.join(save_path, 'model.h5')
# model_json = model.to_json()
# with open(save_file, "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
model.save(save_file)
print("Saved model to disk")
# print('accuracy on test set: %.2f%%' % )
