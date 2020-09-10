# import unittest  # todo: unittestts wasn't working with API, so performing simple assert checks instead
import sys
import os

from conf.conf_loader import conf_object

sys.path.insert(0, os.path.dirname(__file__))

REQUIRED_FIELDS = ["model", "time_features", "lookback_features", "file_path", "time_window", "target_feature"]


def test_conf():
    """
    test that required fields in config are present
    :return:
    """

    # check that required fields are in the project_conf.json & that the json loaded correctly
    assert len(set(REQUIRED_FIELDS)) == len(set(conf_object.project_conf.keys())), \
        "Required fields in project_conf.json aren't present"