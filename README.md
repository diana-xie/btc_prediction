# btc_prediction

App deployed with Heroku, Docker, Flask, Python: [https://penn-challenge.herokuapp.com/](https://penn-challenge.herokuapp.com/)

<ml-engineering-challenge>

```bash
├── ml-engineering-challenge
│   ├── `analysis` - contains EDA notebooks
│   │   ├── bitcoin-predictor.ipynb
│   │   ├── lstm.ipynb
│   │   ├── randomforest.ipynb
│   │   ├── xgboost.ipynb
│   ├── `conf`
│   │   ├── project_conf.json - contains model parameters, which can be configured by user
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
│   │   ├── model.pkl - last saved model, based off bitcoin.csv
│   └── tests
│   │   ├── test_conf.py
│   │   ├── test_model_drift.py
│   │   ├── test_preprocessing_train.py
│   │   ├── test_train_api.py
├── Dockerfile
├── `server.py` - contains endpoints such as `/predict`, `/train`, and calling unit tests
├── train.py
├── utils.py
```

