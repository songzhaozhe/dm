"""Contains the base class for models."""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Masking, Dropout
from keras.models import model_from_json
from keras.optimizers import Adam
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()


class LSTMModel(BaseModel):
    def create_model(self, input):
        lstm_size = 64
        model = Sequential()
        model.add(LSTM(lstm_size, input_shape = input, activation='relu', return_sequences=True))
        model.add(LSTM(lstm_size, activation='relu', return_sequences=True))
        #model.add(LSTM(lstm_size, activation='relu', return_sequences=True))
        #model.add(LSTM(lstm_size, activation='relu', return_sequences=False))
        #model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        adam = Adam(lr=0.001)
        model.compile(optimizer=adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model