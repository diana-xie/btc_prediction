from data.data_preprocessing import preprocess_data
from data.data_lookback import lookback
from conf.conf_loader import conf_object

import logging

data = preprocess_data()
data = lookback(dataset=data,
                    features=conf_object.project_conf["lookback_features"],
                    timesteps=conf_object.project_conf["time_window"]
                    )

data.to_pickle('data.pkl')
logging.info("Finished preprocessing.")