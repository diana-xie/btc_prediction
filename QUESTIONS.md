# btc_prediction

<b>How do we write good tests (unit, integration) for this machine learning service?</b>

Ensure the following (see “tests” subdirectory) and each of their endpoints in server.py:
-	Required params and fields are present
-	Model is working on sample data
-	Preprocessing is working on sample data
-	Model drift is detected, or unusual volume of data is detected

<b>Can we add an endpoint just to check that the API is up?</b>

Endpoint “/” will display “API is up” if API is up.

<b>The model is up and running. How do we know if it's being used?</b>

Currently the Heroku logs will indicate when a request has been made.

<b>How often does it error out? How many 5xx? How many 4xx?</b>

In the first hour of local & Heroku run, no errors were reported when running the endpoints.

<b>How accurate is the model? We can see how accurate the training and validation sets are, but can we include an endpoint to measure how accurate it was in the last 60 seconds?</b>

On bitcoin.csv, the error is ~ 0.10 MAE, which means that the BTC prediction deviates from the actual by 0.10 BTC. For comparison, the average BTC is ~7188 BTC.

Running ‘/predict’ will generate the prediction for time t = 61, based on model pretrained bitcoin.csv. Although an endpoint isn’t included for accuracy in last 60 seconds, my approach would be:

1.	Create an endpoint for calling model.eval() (in rfregressor.py): Ex. ‘/evaluate’
2.	Save ‘/predict’ prediction at t = 61. 
3.	When time t = 62, obtain the BTC value at t = 61 through another endpoint. Ex. endpoint ‘/latest_btc’.
4.	Post to ‘/evaluate’ with t = 61 BTC from ‘latest_btc’. Then get the MAE between predicted BTC at t = 61 and actual BTC at t = 61.
5.	‘/evaluate’ returns the MAE from 4.

<b>What version is the API on and how do we know? Is this the same as the model version? Do we need both?</b>
 
Running predict will print the version of Random Forest Regressor (i.e. scikit-learn version). As long as the model is compatible with API, which is tested through the unit tests, we do not necessarily need both. 

<b>The jupyter notebook is a nice format for quick work, but we don't want to have to run this on our laptop each time we train. Can you create a job for it to run in a different compute environment?</b>

The endpoints in the code will perform train, predict, etc. independently of the Jupyter Notebooks. ‘/train’ for example will train the model and save it in the container environment.

Jupyter Notebooks now contained in separate “analysis” subdirectory and can be run/experimented independently of the code.

<b>Can you make the model more accurate? Some things you could consider:</b>

Please see Notebooks in “analysis” folder. My conclusion was that so far, Random Forest Regressor performed best out of the 3 models I tested (RFregressor, adjusted Neural Net, and LSTM). I have implemented RFregressor in the code, while leaving a template for option to turn on NN or LSTM.

<b>Is the neural network architecture correct?</b>

The example neural net wasn’t entirely correct – had to replace last ReLU layer with Leaky ReLU, as it was facing the “dying ReLu” problem and all predictions were 0. The dying ReLU had also been responsible for the RMSE ~ 100% (~7200 BTC) error reported by the example. With Leaky ReLU, error was reduced to ~ 2-5 BTC MAE from the actual BTC value.

Also replaced RMSE with MAE for interpretability – we know the absolute value of how far our predictions deviated from actual BTC. And this also will not distinguish between negative or positive error, so that we are measuring the absolute deviation.

<b>Is a different model appropriate?</b>

Yes, it turns out that with first-pass grid search Random Forest Regressor beat my adjusted neural net and LSTM. However, it’s possible the neural net architectures were not optimal for the task. For example in one of the visualizations in LSTM, I show that the reason it underperforms RFregressor is because it has trouble with the bimodal distribution of BTC values in our dataset.

<b>Do we have the right loss metric?</b>

The current loss for RFregressor is MSE – this would be appropriate for reducing MAE which also deals with distance from the actual vs. predicted. For LSTM, the loss metric was MAE since the evaluation metric was MAE.

<b>Do we have the right optimizer?</b>

I used Adam optimizer for LSTM, because of its popularity for faster convergence to optima and better avoidance of being stuck in saddle points. However, other optimizers can be tested in the future to see if this improves the error, given my epochs set at 15.

<b>Do we have the right feature set? More features? Less features? Better features?</b>

Adding features could potentially improve performance. However, it should be noted that it’s possible it could reduce performance depending on what features are added. For example, I tested 180 and 360 sec lookback windows, which increased the number of features (ex. “time_open_-180”). This actually made the error larger (i.e. worse error) when predicting BTC.

<b>Do we need to go longer than a 60 second lookback? Shorter?</b>

I tested 180 and 360 sec lookback window, which made the error larger (i.e. worse error). Potentially, shorter lookback windows could actually improve the results. This will depend on model hyperparameter tuning and grid search as well.

<b>Is there any scaling or preprocessing we're missing?</b>

I used a MinMaxScaler and StandardScaler (MinMax gave better error). For future direction, I would experiment with implementing an individual scaler for each time window. So that if in a 60-second period, the BTC suddenly surges, the prediction at t = 61 would reflect the stats anchored to this 60-second period alone.

<b>Once we've trained it in batch several times and we're comfortable with the stability of the model we've chosen, we'll want to be able to train online. Can you create a /train endpoint where we can send data to retrain the model.</b>

/train endpoint is implemented in server.py. This retrains and saves the model each time it is called.

<b>How many concurrent executions is this API capable of handling? Is it thread safe? If we have a /train endpoint and a /predict endpoint will they collide? If we train and predict at the same time, which version of the model is the prediction coming from?</b>

Concurrent execution is possible, as long as there is already a model saved.

 /train independently trains and saves a model. So calling /train and /predict - hypothetically /predict will use the current saved model version that has not yet been replaced/updated by the simultaneously run /train. 

Once /train finishes running, it will replace the current model with the updated model version. This updated model will not be used by /predict until /predict is called after updated model is saved. 

<b>Tensorflow throws a lot of Future warnings and CPU warnings at runtime. Can we fix the code so these warnings go away?</b>

Suppressed warnings by using tf.compat.v1.logging.set_verbositity() to make sure that any incompatibility warnings are suppressed during runtime.
