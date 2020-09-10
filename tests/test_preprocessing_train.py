import sys
import os

from data.data_preprocessing import preprocess_data
from train import train_model

sys.path.insert(0, os.path.dirname(__file__))


def test_preprocessing_train():
    """
    make sure preprocessing and training steps work
    :return:
    """

    data = preprocess_data()
    data = {'bitcoin_last_minute': data.to_dict()}  # convert to "API" request format
    assert data is not None, "Error in preprocess_data()."

    mae = train_model(request_dict=data)
    assert mae is not None, "Error in train_model()."
