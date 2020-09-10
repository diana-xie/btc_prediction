# btc_prediction

App deployed with Heroku, Docker, Flask, Python: [https://penn-challenge.herokuapp.com/](https://penn-challenge.herokuapp.com/)

<ml-engineering-challenge>

```bash
├── ml-engineering-challenge
│   ├── `analysis` - <i>contains EDA notebooks</i>
│   │   ├── bitcoin-predictor.ipynb
│   │   ├── lstm.ipynb
│   │   ├── randomforest.ipynb
│   │   ├── xgboost.ipynb
│   ├── `conf`
│   │   ├── project_conf.json - <i>contains model parameters, which can be configured by user</i>
│   │   ├── conf_loader.py
│   ├── data
│   │   ├── data_lookback.py
│   │   ├── data_main.py
│   │   ├── data_preprocessing.py
│   └── models
│   │   ├── base_model.py
│   │   ├── lstm.py
│   │   ├── neural_net.py
│   │   ├── rfregressor.py
│   │   ├── model.pkl - <i>last saved model, based off bitcoin.csv</i>
│   └── tests
│   │   ├── test_conf.py
│   │   ├── test_model_drift.py
│   │   ├── test_preprocessing_train.py
│   │   ├── test_train_api.py
├── Dockerfile
├── `server.py` - <i>contains endpoints such as `/predict`, `/train`, and calling unit tests</i>
├── train.py
├── utils.py
```

