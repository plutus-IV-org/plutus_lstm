from fitter import Fitter, get_common_distributions
import pandas as pd
import numpy as np


def _data_normalisation(df, is_targeted: bool = False):
    """
    1. Creating daily changes
    2. Taking ln from dc
    """
    if is_targeted:
        column_to_add = df['Close'].rename('Target')

    add_timestamp = False
    if 'timestamp_int' in df.columns:
        df_to_add = df['timestamp_int']
        df = df.drop(columns=['timestamp_int'])
        add_timestamp = True

    df.replace(0, 1, inplace=True)
    df = (df / df.shift()).fillna(1)

    # Use vectorized operations for logarithms
    df = np.log(df)

    if is_targeted:
        df = pd.concat([df, column_to_add], axis=1)
    if add_timestamp:
        df = pd.concat([df, df_to_add], axis=1)

    return df


def _distribution_type(df):
    # Fitting common distributions
    f = Fitter(df, distributions=get_common_distributions())
    f.fit()
    f.get_best(method='sumsquare_error')
    dist_type = (next(iter(f.get_best(method='sumsquare_error'))))
    return dist_type


def _data_denormalisation(array, df, n_future, testY, break_point=True, is_targeted: bool = False,
                          confidence_lvl: float = 0.5
                          ):
    """
    1. Delogging the values
    2. Getting initial values to create closes
    """
    if is_targeted:
        confidence_rate_up = 1 - confidence_lvl
        confidence_rate_down = confidence_lvl

        # Create a DataFrame from the array without copying
        outcome_table = pd.DataFrame(array)

        # Apply vectorized conditions
        array_copy = np.where(outcome_table > confidence_rate_up, 1,
                                 np.where(outcome_table < confidence_rate_down, -1, 0))


    else:
        array_copy = array.copy()
        df = df[['Close']]
        nat = pd.DataFrame(array_copy)
        for z in range(len(nat.index)):
            for f in range(len(nat.columns)):
                nat.iloc[z, f] = np.exp(nat.iloc[z, f])
        array_copy = nat.values
        if break_point:
            in_val = df.iloc[-len(testY) - n_future:-n_future]
        else:
            in_val = df.iloc[-len(testY):]
        for x in range(n_future):
            in_val[x] = np.ones(len(in_val))
        for x in range(len(in_val)):
            for y in range(n_future):
                inp = in_val.iloc[x, y]
                out = pd.DataFrame(array_copy).iloc[x, y]
                in_val.iloc[x, y + 1] = out * inp
        array_copy = in_val.drop(columns=['Close']).values
    return array_copy


def array_maker(input_list):
    # Find the maximum length of the subarrays
    max_length = max([len(subarray) for subarray in input_list])
    # Pad the smaller subarrays with NaN values
    padded_list = []
    for subarray in input_list:
        if len(subarray) == 0:
            padded_list.append([np.nan] * max_length)
        else:
            padded_list.append(np.concatenate([subarray, [np.nan] * (max_length - len(subarray))]))
    # Convert the padded list to a NumPy array
    array = np.array(padded_list, dtype=float)
    return array


def _split_data(df: pd.DataFrame, future: int, past: int, break_point=True, is_targeted: bool = False):
    n_future = future
    n_past = past
    if break_point:
        dataset_train = df.iloc[:int(df.shape[0] * 0.8), :]
        dataset_test = df.iloc[int(df.shape[0] * 0.8):, :]
    else:
        dataset_train = df.iloc[:int(df.shape[0] * 0.35), :]
        dataset_test = df.iloc[int(df.shape[0] * 0.35):, :]

    def fractioning(df: pd.DataFrame, break_point: bool, is_targeted: bool = False):
        train_x, train_y = [], []
        df_training = df.reset_index(drop=True)
        if break_point:
            for x in range(n_past, len(df_training) - n_future + 1):
                if is_targeted:
                    input_val = df_training.loc[x - n_past:x, :].drop(columns=['Target']).values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, :][
                                     'Target'] - df_training.iloc[x, :]['Target']
                    output_val[output_val > 0] = 1
                    output_val[output_val <= 0] = 0
                else:
                    input_val = df_training.loc[x - n_past:x, :].values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, 0].values
                if len(output_val) < n_future:
                    break
                train_x.append(input_val)
                train_y.append(output_val)
            return train_x, train_y
        else:
            for x in range(n_past, len(df_training)):
                if is_targeted:
                    input_val = df_training.loc[x - n_past:x, :].drop(columns=['Target']).values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, :][
                                     'Target'] - df_training.iloc[x, :]['Target']
                    output_val[output_val > 0] = 1
                    output_val[output_val <= 0] = 0
                else:
                    input_val = df_training.loc[x - n_past:x, :].values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, 0].values
                train_x.append(input_val)
                train_y.append(output_val)
            return train_x, train_y

    train_x, train_y = fractioning(dataset_train, break_point, is_targeted)
    test_x, test_y = fractioning(dataset_test, break_point, is_targeted)

    trainX = np.array(train_x, dtype=float)
    trainY = array_maker(train_y)
    testX = np.array(test_x, dtype=float)
    testY = array_maker(test_y)

    return trainX, trainY, testX, testY


def partial_data_split(df: pd.DataFrame, future: int, past: int, is_targeted: bool = False):
    test_x, test_y = [], []
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    for x in range(past, len(df)):
        if is_targeted:
            input_val = df.loc[x - past:x, :].drop(columns=['Target']).values
            output_val = df.iloc[x + 1:x + future + 1, :][
                             'Target'] - df.iloc[x, :]['Target']
            output_val[output_val > 0] = 1
            output_val[output_val <= 0] = 0
        else:
            input_val = df.loc[x - past:x, :].values
            output_val = df.iloc[x + 1:x + future + 1, 0].values

        test_x.append(input_val)
        test_y.append(output_val)

    testX = np.array(test_x, dtype=float)
    testY = array_maker(test_y)

    return testX, testY
