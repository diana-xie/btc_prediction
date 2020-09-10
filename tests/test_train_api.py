""" Optional unit test for API, which is meant to run as a unittest suite"""
import json
import unittest
from unittest.mock import patch
import requests

from data.data_preprocessing import preprocess_data


class TestAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.data = preprocess_data()
        self.data = {'bitcoin_last_minute': self.data.to_dict()}  # convert to "API" request format

    @patch('requests.post')
    def test_post(self, mock_post):
        self.data = preprocess_data()
        self.data = {'bitcoin_last_minute': self.data.to_dict()}  # convert to "API" request format
        info = self.data
        url = 'http://0.0.0.0:5000/train'
        resp = requests.post(url, data=json.dumps(info), headers={'Content-Type': 'application/json'})
        mock_post.assert_called_with(url, data=json.dumps(info), headers={'Content-Type': 'application/json'})


TestAPI().test_post()