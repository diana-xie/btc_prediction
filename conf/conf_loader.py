import logging
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


class SetupConf:

    def __init__(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), 'project_conf.json')) as stream:
                self.project_conf = json.load(stream)
        except Exception as ex:
            logging.error("Error loading project_conf.json: {}".format(ex))
            raise ex


conf_object = SetupConf()
