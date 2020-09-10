import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime
sys.path.insert(0, os.path.dirname(__file__))

from conf.conf_loader import conf_object


def info_date(data: pd.DataFrame):
    """
    print info on date features
    :param data: bitcoin data
    :return:
    """

    try:
        length_trade_day = (data['time_period_end']-data['time_period_start']).unique()
        length_trade_day = length_trade_day/np.timedelta64(1, 's')
        print('Length of time increments: ', length_trade_day.max(), ' sec')
        print('# unique time increments: ', len(set(length_trade_day)), '\n')
        if len(set(length_trade_day)) > 1:
            logging.info("Length of time increments is unequal.")
        print("Max date: ", data['time_period_end'].max())
        print("Min date: ", data['time_period_end'].min())

    except Exception as ex:
        logging.error("Error in info_date(): {}".format(ex))
        raise ex


def preprocess_time(data: pd.DataFrame,
                    ) -> pd.DataFrame:
    """
    preprocess date features
    :param data: bitcoin data
    :return:
    """

    try:
        data['time_period_start'] = pd.to_datetime(data['time_period_start'])
        data['time_period_end'] = pd.to_datetime(data['time_period_end'])
        data['hour'] = [x.hour for x in data['time_period_end']]  # make hour of day a feature, in case useful
        data = data.sort_values('time_period_end')

        info_date(data=data)

        data.drop(conf_object.project_conf["time_features"],
                  axis=1,
                  inplace=True
                  )  # drop unused time features

        return data

    except Exception as ex:
        logging.error("Error in preprocess_time(): {}".format(ex))
        raise ex


def preprocess_data(data: pd.DataFrame = None):

    try:
        if data is None:
            # get data
            data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), conf_object.project_conf["file_path"]),
                index_col=0
            )
            # preprocessing - time
            data = preprocess_time(data=data)
        else:
            data['hour'] = datetime.datetime.now().hour

        return data

    except Exception as ex:
        logging.error("Error in preprocess_data(): {}".format(ex))
        raise ex


if __name__ == "__main__":
    data = preprocess_data()
