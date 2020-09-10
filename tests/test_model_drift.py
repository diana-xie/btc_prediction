import unittest
import sys
import os
import numpy as np
from copy import deepcopy

from data.data_preprocessing import preprocess_data
from conf.conf_loader import conf_object

sys.path.insert(0, os.path.dirname(__file__))


def test_model_drift():
    """
    catch model drift. meant to show when test fails and returns error msg, but could be set to break instead
    :return: error message, or success message
    """

    error_msg = 'Successfully ran model drift-volume anomaly tests.'

    target_feature = conf_object.project_conf["target_feature"]

    # fake data of the "previous" data
    data_previous_dummy = preprocess_data()

    # fake data of the incoming "new" data that has drifted
    data_drifted_dummy = deepcopy(data_previous_dummy)
    data_drifted_dummy[target_feature] = np.zeros(len(data_previous_dummy))

    mu1 = data_previous_dummy[target_feature].mean()
    mu2 = data_drifted_dummy[target_feature].mean()

    std = data_previous_dummy[target_feature].std()
    drift_threshold = std*3

    try:
        assert abs(mu1-mu2) < drift_threshold, "Model drift detected!"
    except Exception as ex:
        error_msg = "This is supposed to break since dummy data mean is 0 BTC. " \
                    "Returning error message for testing purposes."

    return error_msg
