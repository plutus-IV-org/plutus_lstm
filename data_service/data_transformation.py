from typing import Tuple, List
from fitter import Fitter, get_common_distributions
import pandas as pd
import numpy as np
from numpy import ndarray

from Const import CROSS_VALIDATION_CHUNKS


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


def log_z_score_rolling(data, window=85, is_targeted: bool = False):
    # TEST normalisation 22.10.2024 - No significant results...
    if is_targeted:
        column_to_add = data['Close'].rename('Target')

    add_timestamp = False
    if 'timestamp_int' in data.columns:
        df_to_add = data['timestamp_int']
        data = data.drop(columns=['timestamp_int'])
        add_timestamp = True

    data.replace(0, 1, inplace=True)

    z_scores = ((data - data.rolling(window).mean()) / data.rolling(window).std()).dropna() + 4
    data = (np.log(z_scores)).dropna()
    if is_targeted:
        data = pd.concat([data, column_to_add], axis=1)
    if add_timestamp:
        data = pd.concat([data, df_to_add], axis=1)
    return data


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
    """
    Splits the input DataFrame into training and testing sets based on the provided 'future' and 'past' time steps.
    It also generates the input sequences (trainX, testX) and their corresponding targets (trainY, testY).

    Parameters:
    - df: pd.DataFrame, the input time series data.
    - future: int, the number of future time steps to predict.
    - past: int, the number of past time steps used as input features.
    - break_point: bool, indicates whether to split data by 80/20 (True) or 35/65 (False).
    - is_targeted: bool, if True, targets are binary classification (1/0) based on positive/negative change in 'Target' column.

    Returns:
    - trainX: np.array, input sequences for the training set.
    - trainY: np.array, target sequences for the training set.
    - testX: np.array, input sequences for the testing set.
    - testY: np.array, target sequences for the testing set.
    """
    # Define future and past time steps for prediction
    n_future = future
    n_past = past

    # Split the DataFrame into training and testing sets (80/20 if break_point is True, 35/65 if False)
    if break_point:
        dataset_train = df.iloc[:int(df.shape[0] * 0.8), :]
        dataset_test = df.iloc[int(df.shape[0] * 0.8):, :]
    else:
        dataset_train = df.iloc[:int(df.shape[0] * 0.35), :]
        dataset_test = df.iloc[int(df.shape[0] * 0.35):, :]

    def fractioning(df: pd.DataFrame, break_point: bool, is_targeted: bool = False):
        """
        Creates input sequences (X) and their corresponding targets (Y) from the provided DataFrame.

        Parameters:
        - df: pd.DataFrame, the input data.
        - break_point: bool, same as defined above, used to decide between two modes of splitting.
        - is_targeted: bool, indicates whether to create binary classification targets based on 'Target' column.

        Returns:
        - train_x: list, input sequences.
        - train_y: list, target sequences.
        """
        train_x, train_y = [], []  # Initialize lists to store input sequences and targets
        df_training = df.reset_index(drop=True)  # Reset DataFrame index to avoid issues with slicing

        # Loop over the dataset to create past/future sequences
        if break_point:
            for x in range(n_past, len(df_training) - n_future + 1):
                if is_targeted:
                    # If is_targeted, calculate binary target values based on 'Target' column
                    input_val = df_training.loc[x - n_past:x, :].drop(columns=['Target']).values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, :]['Target'] - df_training.iloc[x, :][
                        'Target']
                    # Convert target values to 1 (positive change) or 0 (negative/no change)
                    output_val[output_val > 0] = 1
                    output_val[output_val <= 0] = 0
                else:
                    # If not targeted, use values from the DataFrame as targets
                    input_val = df_training.loc[x - n_past:x, :].values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, 0].values  # Using the first column as target

                # If target sequence is shorter than n_future, break the loop
                if len(output_val) < n_future:
                    break

                train_x.append(input_val)
                train_y.append(output_val)

            return train_x, train_y
        else:
            # Similar logic when break_point is False, used for another way of slicing
            for x in range(n_past, len(df_training)):
                if is_targeted:
                    input_val = df_training.loc[x - n_past:x, :].drop(columns=['Target']).values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, :]['Target'] - df_training.iloc[x, :][
                        'Target']
                    output_val[output_val > 0] = 1
                    output_val[output_val <= 0] = 0
                else:
                    input_val = df_training.loc[x - n_past:x, :].values
                    output_val = df_training.iloc[x + 1:x + n_future + 1, 0].values

                train_x.append(input_val)
                train_y.append(output_val)

            return train_x, train_y

    # Generate input/target sequences for both training and testing sets
    train_x, train_y = fractioning(dataset_train, break_point, is_targeted)
    test_x, test_y = fractioning(dataset_test, break_point, is_targeted)

    # Convert lists to NumPy arrays for compatibility with LSTM models
    trainX = np.array(train_x, dtype=float)
    trainY = array_maker(train_y)  # Convert to array using the array_maker function
    testX = np.array(test_x, dtype=float)
    testY = array_maker(test_y)

    return trainX, trainY, testX, testY


def fraction(df: pd.DataFrame, n_past: int, n_future: int, is_targeted: bool = False):
    """
    Creates input sequences (X) and their corresponding targets (Y) from the provided DataFrame.

    Parameters:
    - df: pd.DataFrame, the input data.
    - n_past: int
        Lookback period
    - n_future: int
        Look forward period
    - is_targeted: bool, indicates whether to create binary classification targets based on 'Target' column.

    Returns:
    - x: list, input sequences.
    - y: list, target sequences.
    """
    X, Y = [], []  # Initialize lists to store input sequences and targets
    data = df.reset_index(drop=True).copy()  # Reset DataFrame index to avoid issues with slicing

    # Loop over the dataset to create past/future sequences
    for x in range(n_past, len(data) - n_future + 1):
        if is_targeted:
            # If is_targeted, calculate binary target values based on 'Target' column
            input_val = data.loc[x - n_past:x, :].drop(columns=['Target']).values
            output_val = data.iloc[x + 1:x + n_future + 1, :]['Target'] - data.iloc[x, :][
                'Target']
            # Convert target values to 1 (positive change) or 0 (negative/no change)
            output_val[output_val > 0] = 1
            output_val[output_val <= 0] = 0
        else:
            # If not targeted, use values from the DataFrame as targets
            input_val = data.loc[x - n_past:x, :].values
            output_val = data.iloc[x + 1:x + n_future + 1, 0].values  # Using the first column as target

        # If target sequence is shorter than n_future, break the loop
        if len(output_val) < n_future:
            break

        X.append(input_val)
        Y.append(output_val)

    return X, Y


def data_split(df: pd.DataFrame, future: int, past: int, is_targeted: bool = False) -> tuple[
    ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, List[pd.Timestamp]]:
    """
    Splits the input DataFrame into training, testing, and evaluation sets based on the provided 'future' and 'past' time steps.
    It also generates the input sequences (trainX, evalX, testX) and their corresponding targets (trainY, evalY, testY).

    Parameters:
    ----------
    df : pd.DataFrame
        The input time series data, where each row represents one time step, and columns represent features.

    future : int
        The number of future time steps to predict. For example, if future=5, the model will predict 5 time steps ahead.

    past : int
        The number of past time steps used as input features. For example, if past=100, the model will use the previous 100 time steps as input.

    is_targeted : bool, optional (default=False)
        If True, the target sequences will be binary classification (1 for upward movement, 0 for downward movement)
        based on the positive/negative change in the 'Target' column. If False, the target sequences will be the actual values.

    Returns:
    -------
    trainX : np.ndarray
        A 3D array of input sequences for the training set, with shape (num_samples, past, num_features).

    trainY : np.ndarray
        A 2D array of target sequences for the training set, with shape (num_samples, num_future_steps).

    evalX : np.ndarray
        A 3D array of input sequences for the evaluation set, with shape (num_samples, past, num_features).

    evalY : np.ndarray
        A 2D array of target sequences for the evaluation set, with shape (num_samples, num_future_steps).

    testX : np.ndarray
        A 3D array of input sequences for the testing set, with shape (num_samples, past, num_features).

    testY : np.ndarray
        A 2D array of target sequences for the testing set, with shape (num_samples, num_future_steps).
    """

    # Split the DataFrame into training (60%), evaluation (20%), and testing (20%) sets
    dataset_train = df.iloc[:int(df.shape[0] * 0.6), :]
    dataset_eval = df.iloc[int(df.shape[0] * 0.6):int(df.shape[0] * 0.95), :]
    dataset_test = df.iloc[int(df.shape[0] * 0.95):, :]

    # Generate input/target sequences for all sets
    train_x, train_y = fraction(dataset_train, past, future, is_targeted)
    eval_x, eval_y = fraction(dataset_eval, past, future, is_targeted)
    test_x, test_y = fraction(dataset_test, past, future, is_targeted)

    # Convert lists to NumPy arrays for compatibility with LSTM models
    train_x = np.array(train_x, dtype=float)
    train_y = array_maker(train_y)  # Convert to 2D array using the array_maker function
    eval_x = np.array(eval_x, dtype=float)
    eval_y = array_maker(eval_y)
    test_x = np.array(test_x, dtype=float)
    test_y = array_maker(test_y)

    return train_x, train_y, eval_x, eval_y, test_x, test_y, [dataset_train.index, dataset_eval.index,
                                                              dataset_test.index]


def cross_validation_data_split(df: pd.DataFrame, future: int, past: int, is_targeted: bool = False):
    """
    Function to perform incremental training using an expanding window on time-series data.

    df: pd.DataFrame - The dataset containing time-series data.
    future: int - Number of future days to predict.
    past: int - Number of past days to use for training.
    is_targeted: bool - Whether the function is targeting price movement classification.

    Returns:
    - train_x_list: A list of 3D arrays of training input sequences for each cross-validation chunk.
    - train_y_list: A list of 2D arrays of training target sequences for each cross-validation chunk.
    - eval_x_list: A list of 3D arrays of evaluation input sequences for each cross-validation chunk.
    - eval_y_list: A list of 2D arrays of evaluation target sequences for each cross-validation chunk.
    - test_x_list: A list of 3D arrays of testing input sequences for each cross-validation chunk.
    - test_y_list: A list of 2D arrays of testing target sequences for each cross-validation chunk.
    """

    train_x_list, train_y_list = [], []
    eval_x_list, eval_y_list = [], []
    test_x_list, test_y_list = [], []

    # Loop over each percentage chunk from CROSS_VALIDATION_CHUNKS
    for chunk in CROSS_VALIDATION_CHUNKS:
        # Define the training set size based on the chunk
        current_train_size = int(df.shape[0] * chunk)

        # Create a subset of the data for the current chunk
        subset_df = df.iloc[:current_train_size, :]

        # Use the existing data_split function to get train/eval/test sets for the current chunk
        train_x, train_y, eval_x, eval_y, test_x, test_y, list_of_index = data_split(subset_df, future, past,
                                                                                     is_targeted)

        # Append the results to the respective lists
        train_x_list.append(train_x)
        train_y_list.append(train_y)
        eval_x_list.append(eval_x)
        eval_y_list.append(eval_y)
        test_x_list.append(test_x)
        test_y_list.append(test_y)

    return train_x_list, train_y_list, eval_x_list, eval_y_list, test_x_list, test_y_list, list_of_index


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
