import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error
import sys
import os

from conf.conf_loader import conf_object
from data.data_preprocessing import preprocess_data

sys.path.insert(0, os.path.dirname(__file__))


class BaseModel:

    def __init__(self):

        try:
            self.data = pd.read_pickle(os.path.join(os.path.dirname(__file__), "data.pkl"))
        except Exception as ex:
            logging.info("Couldn't load data.pkl, so preprocessing data to generate.")
            self.data = preprocess_data()

        self.project_conf = conf_object.project_conf
        self.model = None

        self.target_feature = self.project_conf["target_feature"]
        self.features = list(self.data.columns)
        self.features.remove(self.target_feature)

        self.actuals = self.data[self.target_feature].values

    def get_mae(self, actuals, predictions):

        mae = mean_absolute_error(actuals, predictions)
        print("MAE: ", mae)
        return mae