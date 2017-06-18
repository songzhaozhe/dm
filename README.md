# Data Mining - A price trend predicter on the stock exchange

## Introduction

This is the final project of Data Mining for IEEE class in 2014 grade in SJTU. A price trend predicter on the stock exchange is provided based on the given dataset. several models are trained and the best accuracy achieved 62.5% with LSTM model.

# Detailed usage

## Dependent

* Python = 2.7
* [NumPy](http://www.numpy.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [scikit-learn](http://scikit-learn.org/)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/)

## Pre-work

### Data Preparation
Make sure you have the dataset provided in this class. move and uncompress the 'm0000' dataset to ./data (recommended) or any other directory you like.
```
mkdir data
mv path/to/m0000_20160101-20170410.rar ./data/
rar x data/m0000_20160101-20170410.rar
```

### Data Pre-processing
Do the data pre-processing by ruuning:
```
python process_data.py
```
If you have save your data to other directory, running:
```
python process_data.py -dir yourpath
```

### Data Pre-processing (for MSE trick)
In this task, we tried to convert the labeling question into a regression one and use MSE as loss function for training. To do so, another data pre-processing should be done by running:
```
python process_data_reg.py
```
If you have save your data to other directory, running:
```
python process_data_reg.py -dir yourpath
```

## Training models

### Baseline, Logistic and LSTM models
These models can be trained by running `train.py` with diffrent arguments. For example, you can train LSTM model with default parameter by running:
```
python train.py -m LSTMModel
```
You can also select the model and modify the training parameters with command line arguments as follow:
* -m [BaselineModel|LSTMModel|LogisticModel] - the model you wish to train
* -dir ... path to the dataset - don't required if you save your data in recommended way
* -o ... the directory you wish to save the training result. default directory is ./model/model_name/test/
* -ts 20 ... time step - how many ticks you wish to use to predict the price trend. this parameter is unmeaning for BaselineModel. default is 20
* -bs 512 ... batch size - larger values usually speed up training however increase the memory usage. default is 512
* -tt 3 ... traverse time - how many time the train dataset will be traverse. An overfitting problem will happened if it's too big. default is 3
* -ep 50 ... epoch number - our training use generator to create the data stream for training. larger epoch number will have more outputed training loss and acc information while the sample number in each epoch will be small. default is 50
* -id 0 ... GPU id - required for computer with multi-GPU to choose which GPU to use. default is 0

### MSE trick
If you want to use the MSE trick for training, make sure you have done the pre-processing for it.Then, run `train_reg.py` instead of `train.py`. The command line arguments they used are same.

### XGBoost
XGBoost model can be trained by running:
```
python xgboost_train.py
```
You can also modify the parameters of XGBoost with command line arguments as follow:
* -dir ... path to the dataset - don't required if you save your data in recommended way
* -ts 5 ... time step - how many ticks you wish to use to predict the price trend. default is 5
* -eta 0.1 ... the shrinkage rate, which is equal to learning rate in other model. a float number from 0 to 1. default is 0.1
* -maxd 12 ... max depth for one tree. A large depth may lead to overfitting problem. default is 12
* -nth 4 ... number of tread used for training. default is 4
* -round 15 ... number of training round. default is 15
* -s 1 ... 0 for print the training log and 1 for not. default is 1

# Contributors
Zhaozhe Song, Wenhao Qu
