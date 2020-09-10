""" Runs the endpoints for BTC predict, train, unit tests """

import tensorflow as tf
from flask import Flask, jsonify, request
import os

import logging
import pkg_resources
import pandas as pd

from tests.test_conf import test_conf
from tests.test_preprocessing_train import test_preprocessing_train
from tests.test_model_drift import test_model_drift
from train import train_model
from utils import fix_path, process_request

# remove tf warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))


@app.route('/', methods=['GET'])
def server_is_up():
    # print("success")
    return 'API is up.'


@app.route('/train', methods=['POST'])  # POST
def train_api():
    observation = request.json
    mae = train_model(observation)
    return 'Model has been trained and saved. MAE is {}'.format(mae)


@app.route('/predict', methods=['POST'])  # POST
def predict_api():

    try:
        model = pd.read_pickle(os.path.join(fix_path(), "models/model.pkl"))
        logging.info("RFregressor version: ", pkg_resources.get_distribution("scikit-learn"))

        # observation = observation.encode()  # this code is for scenario where data is encoded as str in POST
        # observation = pickle.loads(base64.b64decode(observation))
        # request = open('request.json', 'rb')  # todo - comment out if not testing locally
        observation = request.json

        observation = process_request(observation=observation)
        pred = model.get_prediction(observation)

        return jsonify({"bitcoin prediction": str(pred)})

    except Exception as ex:
        logging.error("No model was found, so run /train")


""" unit tests"""


@app.route('/test_conf', methods=['GET'])
def unit_tests_conf():
    test_conf()
    return 'Successfully ran conf test.'


@app.route('/test_preprocess_train', methods=['GET'])
def unit_tests_preprocess():
    test_preprocessing_train()
    return 'Successfully ran preprocessing and train tests.'


@app.route('/test_drift', methods=['GET'])
def unit_tests_drift():
    msg = test_model_drift()
    return msg


if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=port)