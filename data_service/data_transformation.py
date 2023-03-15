from fitter import Fitter, get_common_distributions
import pandas as pd
import numpy as np

def _data_normalisation(df):
    """
    1.Creating daily changes
    2.Taking ln from dc
    """
    df.replace(0,1,inplace =True)
    df = (df / df.shift()).fillna(value=1)
    for x in df.columns:
        val = []
        for y in df[x].values:
            val.append(np.log(y))
        df[x] = val
    return df

def _distribution_type(df):

    # Fitting common distributions
    f = Fitter(df, distributions=get_common_distributions())
    f.fit()
    f.get_best(method='sumsquare_error')
    dist_type = (next(iter(f.get_best(method='sumsquare_error'))))
    return dist_type

def _data_denormalisation(array, df, n_future, testY):
    """
    1.Deloging the values
    2.Getting initial values to create closes
    """

    array_copy = array.copy()

    nat = pd.DataFrame(array_copy)
    for z in range(len(nat.index)):
        for f in range(len(nat.columns)):
            nat.iloc[z, f] = np.exp(nat.iloc[z, f])
    array_copy = nat.values
    in_val = df.iloc[-len(testY) - n_future:-n_future]
    for x in range(n_future):
        in_val[x] = np.ones(len(in_val))
    for x in range(len(in_val)):
        for y in range(n_future):
            inp = in_val.iloc[x, y]
            out = pd.DataFrame(array_copy).iloc[x, y]
            in_val.iloc[x, y + 1] = out * inp
    array_copy = in_val.drop(columns=['Close']).values
    return array_copy

def _split_data(df, future, past):
    n_future = future
    n_past = past
    dataset_train = df.iloc[:int(df.shape[0] * 0.8), :]
    dataset_test = df.iloc[int(df.shape[0] * 0.8) - n_past - 1:]
    def fractioning(df):
        train_x, train_y = [], []
        df_training = df.reset_index(drop=True)
        for x in range(n_past, len(df_training) - n_future + 1):
            input_val = df_training.loc[x - n_past:x, :].values
            output_val = df_training.iloc[x + 1:x + n_future + 1, 0].values
            if len(output_val) < n_future:
                break
            train_x.append(input_val)
            train_y.append(output_val)
        return train_x, train_y

    train_x, train_y = fractioning(dataset_train)
    test_x, test_y = fractioning(dataset_test)

    trainX = np.array(train_x, dtype=float)
    trainY = np.array(train_y, dtype=float)
    testX = np.array(test_x, dtype=float)
    testY = np.array(test_y, dtype=float)

    return trainX, trainY, testX, testY

