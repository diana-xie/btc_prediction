import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data.data_preprocessing import preprocess_data


def fix_path():
    """
    ensures that if objects living locally are loaded, correct path is used
    :return:
    """
    path_name = os.path.dirname(__file__).split('/')
    if path_name[-1] == 'scripts':
        path_name = path_name[:-1]
    path_name = '/'.join(path_name)
    return path_name


def process_request(observation: dict):
    """
    format request from API into form taken by model
    :param observation: request from API
    :return: formatted request
    """
    observation = pd.DataFrame(observation["bitcoin_last_minute"], index=[0])
    observation = preprocess_data(observation)
    return observation
