import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU

from models.base_model import BaseModel

# remove tf warning messages
tf.logging.set_verbosity(tf.logging.ERROR)

# todo: just a template - not actually working since decided to use RFregressor. see Notebooks for LSTM results
class LSTM(BaseModel):

    def __init__(self, data: pd.DataFrame = None):

        super(LSTM, self).__init__()
        self.target_scaler = None
        self.epochs = None
        self.batch_size = None
        self.verbose = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.stack_arr = None

        if data:
            self.data = data
        self.data = self.data.sort_values('time_period_end')
        self.time_window = self.project_conf["time_window"]

    def architecture(self, epochs: int = 15, batch_size: int = 64, verbose: bool = True):

        n_timesteps, n_features, n_outputs = self.X_train.shape[1], self.X_train.shape[2], 1

        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(Dense(100, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(1))
        model.add(LeakyReLU(alpha=0.1))

        model.compile(loss='mean_absolute_error', optimizer='adam', metrics='mean_absolute_error')

        self.model = model

    def train(self):

        # split data into data and target
        data = self.data.drop(self.target_feature, axis=1).values

        # make model
        self.architecture()

        # train model
        _ = model.fit(self.X_train,
                      self.y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      verbose=self.verbose)

        return self.model

    def eval(self):

        self.prep_data()
        self.train_test_split()
        model = self.train()
        # evaluate
        mae_scaled = model.evaluate(self.X_test,
                       self.y_test,
                       batch_size=self.batch_size,
                       verbose=self.verbose)

        predictions = model.predict(self.X_test)
        predictions_unscaled = self.target_scaler.inverse_transform(np.array(predictions))
        actuals = self.target_scaler.inverse_transform(np.array(self.y_test).reshape(-1, 1))

        mae = self.get_mae(actuals, predictions_unscaled)

        return mae, model

    def scale_values(self):

        target_scaler = None

        for feat in self.data.columns:
            scaler = MinMaxScaler()
            self.data[feat] = scaler.fit_transform(np.array(self.data[feat]).reshape(-1, 1))
            if feat == self.target_feature:
                target_scaler = deepcopy(scaler)  # save scaler for target, to transform back to original later
        self.target_scaler = target_scaler

    def window_stack(self, width: int = 60):
        stack = [self.data.iloc[i:width + i] for i in range(0, len(self.data) - width + 1)]
        return stack

    def window_target(self, width: int = 60):
        targets = np.array([self.data.iloc[width + i]['price_open'] for i in range(0, len(self.data) - width)])
        return targets

    def prep_data(self):

        self.scale_values()

        # get 60-sec time windows
        stacks = self.window_stack(width=self.time_window)[:-1]  # since last window's 9999 doesn't have nxt 10000 idx
        self.actuals = self.window_target(width=self.time_window)

        assert len(stacks) == len(self.actuals), "The stacks and targets aren't of equal length."

        # convert to 3D array
        stack_arr = [np.array(x) for x in stacks]
        self.stack_arr = np.array(stack_arr)
        features = stacks[-1].columns

        print('data shape: ', stack_arr.shape)
        print('features: ', features.values)

        n_data, window_width, n_features = stack_arr.shape
        assert n_data == len(stacks), "The n_data stack_arr[0] is incorrect shape."
        assert window_width == self.time_window, "The window_width stack_arr[1] is incorrect shape."
        assert n_features == len(features), "The n_features stack_arr[2] is incorrect shape."

    def train_test_split(self):

        # Treat each row as a "sequence".
        train_idx, test_idx = train_test_split(range(0, len(self.stack_arr)),
                                               test_size=1 / 3,
                                               random_state=99
                                               )

        X_train = self.stack_arr[train_idx]
        y_train = np.array([self.actuals[i] for i in train_idx])
        assert len(X_train) == len(y_train), "X_train and y_train aren't of same len."

        X_test = self.stack_arr[test_idx]
        y_test = np.array([self.actuals[i] for i in test_idx])
        assert len(X_test) == len(y_test), "X_train and y_train aren't of same len."

        assert len(X_train) + len(X_test) == len(self.stack_arr), \
            "Total length of train & test don't add up to original."

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


model = LSTM()
LSTM.eval()