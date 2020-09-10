""" Train the model amongst the options provided"""
import pandas as pd
import logging
import pickle
import os

from conf.conf_loader import conf_object
from utils import fix_path


def train_model(request_dict: dict = None):
    """
    train model among options specified in project_conf.json.
    :param request_dict: request posted via API
    :return: mae, after saving updated model
    """

    model = None
    if request_dict:
        data = pd.DataFrame(request_dict["bitcoin_last_minute"], index=[0])
    else:
        logging.info("Train mode.")

    model_name = conf_object.project_conf["model"]

    if model_name == 'rfregressor':
        from models.rfregressor import RFregressor
        model = RFregressor()

    if model_name == 'neuralnet':
        from models.neural_net import NeuralNet
        model = NeuralNet(data=data)

    if model_name == 'lstm':
        from models.lstm import LSTM
        model = LSTM(data=data)

    mae = model.eval()

    # save model
    with open(os.path.join(fix_path(), 'models/model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    return mae