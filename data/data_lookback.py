import pandas as pd


def lookback(dataset: pd.DataFrame,
             features: list,
             timesteps: int = 60
             ):
    # this uses the shift method of pandas dataframes to shift all of the columns down one row
    # and then append to the original dataset
    data = dataset
    for i in range(1, timesteps):
        step_back = dataset[features].shift(i).reset_index()
        step_back.columns = ['index'] + [f'{column}_-{i}' for column in dataset[features].columns if column != 'index']
        data = data.reset_index().merge(step_back, on='index', ).drop('index', axis=1)

    return data.dropna()
