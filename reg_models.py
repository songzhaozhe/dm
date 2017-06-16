"""Contains the base class for models."""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Masking, Dropout, Flatten
from keras.models import model_from_json
from keras.optimizers import Adam
import keras.backend as K
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()
def custom_acc(y_true, y_pred):
    # count = 0
    # for i in range (y_true.shape[0]):
    #     if y_pred[i] >= 0 and y_true[i] >= 0:
    #         count = count + 1
    #     if y_pred[i] < 0 and y_true[i] < 0:
    #         count = count + 1
    # return count * 1.0 / y_true.shape[0]
    return K.mean(K.equal(K.sign(y_true), K.sign(y_pred)))

class LSTMModel(BaseModel):
    def create_model(self, input):
        lstm_size = 64
        model = Sequential()
        model.add(LSTM(16, input_shape=input, activation='relu', return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dense(1, activation = None))
        adam = Adam(lr=0.001)
        model.compile(optimizer=adam,
                      loss='mean_squared_error',
                      metrics=[custom_acc])
        return model

class LogisticModel(BaseModel):
    def create_model(self, input):
        model = Sequential()
        model.add(Flatten(input_shape = input))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation=None))
        adam = Adam(lr=0.001)
        model.compile(optimizer=adam,
                      loss='mean_squared_error',
                      metrics=[custom_acc])
        return model