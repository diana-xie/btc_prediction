""" Random Forest Regressor, which was the best model out of the 3 for predicting BTC after 60-sec window"""

import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from models.base_model import BaseModel


class RFregressor(BaseModel):

    def __init__(self, data: pd.DataFrame = None):
        super(RFregressor, self).__init__()
        if data:
            self.data = data

    def scale_values(self):

        target_scaler = None

        # scale features
        for feat in self.data.columns:
            scaler = MinMaxScaler()
            self.data[feat] = scaler.fit_transform(np.array(self.data[feat]).reshape(-1, 1))
            if feat == self.target_feature:
                target_scaler = deepcopy(scaler)  # save scaler for target, to transform back to original later
        self.target_scaler = target_scaler

    def architecture(self):
        """ set model """
        self.model = RandomForestRegressor(bootstrap=True, max_depth=80, max_features=3, min_samples_leaf=3,
                                      min_samples_split=8, n_estimators=200)

    def train_test_split(self):
        """
        perform train test split
        :return: train, test sets
        """
        self.train, self.test = train_test_split(self.data, test_size=1 / 3, random_state=99)

    def train(self):
        """
        train model
        :return:
        """

        self.architecture()
        self.scale_values()
        self.train_test_split()
        self.model.fit(self.train[self.features], self.train[self.target_feature])

    def eval(self):
        """
        eval model
        :return:
        """

        self.train()
        predictions_unscaled = self.get_prediction(self.test)

        # get actual values - unscaled
        self.test_labels = self.test[self.target_feature]  # scaled values
        actuals = self.target_scaler.inverse_transform(np.array(self.test_labels).reshape(-1, 1))  # now unscaled

        mae = mean_absolute_error(predictions_unscaled, actuals)  # get error
        print("MAE for RFregressor is: ", mae)

        return mae

    def get_prediction(self, test):
        """
        get prediction, real-time inference
        :param test: test set to perform inference/pred on
        :return: prediction, in BTC unit
        """

        self.test_features = test[self.features]

        # make prediction
        predictions = self.model.predict(self.test_features)
        # rescale back to original
        predictions_unscaled = self.target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return predictions_unscaled
