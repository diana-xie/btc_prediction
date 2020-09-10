import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU

from models.base_model import BaseModel


# todo: just a template - not actually working since decided to use RFregressor. see Notebooks for NeuralNet results
class NeuralNet(BaseModel):

    def __init__(self, data: pd.DataFrame = None):

        super(NeuralNet, self).__init__()
        if data:
            self.data = data

    def architecture(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.data.shape[1]))
        model.add(Dense(16))
        model.add(Dense(1))
        model.add(LeakyReLU(alpha=0.1))

        model.compile(
            loss='mse',
            optimizer=Adam(lr=0.01),  # is this the best optimizer/learning rate?
            metrics=['mean_squared_error', 'mean_absolute_error']  # does accuracy make sense in this context?
        )

        self.model = model

    def train(self):

        # split data into data and target
        data = self.data.drop(self.target_feature, axis=1).values

        # make model
        self.architecture()

        # callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='auto',
            restore_best_weights=True
        )

        # train model
        _ = self.model.fit(
            data,
            self.actuals,
            validation_split=.3,
            epochs=20,
            verbose=1
        )

        return self.model

    def eval(self):

        # train model
        _ = self.train()
        # predict & eval
        predictions = self.model.predict(self.data)
        mae = self.get_mae(self.actuals, predictions)
        return mae
