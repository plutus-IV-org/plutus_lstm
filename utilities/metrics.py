from tensorflow.keras.metrics import RootMeanSquaredError, mean_absolute_percentage_error
import numpy as np
import pandas as pd


def _rmse(prediction, actual):
    """
    RootMeanSquaredError RMSE measures the square root
    of the mean square error of the actual values and estimated values
    """
    rmse = RootMeanSquaredError()
    rmse.update_state(prediction, actual)
    rmse = rmse.result().numpy()

    return rmse


def _mape(prediction, actual):
    """
    Mean_absolute_percentage_error estimates the size of
    the error computed as the relative average of the error
    """
    return mean_absolute_percentage_error(actual, prediction).numpy().mean()


def _r(prediction, actual):
    """
    R determines the linear correlation between actual
    and predicted values
    """
    actual = np.reshape(actual, (actual.shape[0],))
    prediction = np.reshape(prediction, (prediction.shape[0],))
    r = np.corrcoef(actual, prediction)[0][1]

    return r


def _gradient_accuracy_test(prediction, actual, best_model):
    actual = np.reshape(actual, (actual.shape[0],))
    yhat = np.reshape(prediction, (prediction.shape[0],))

    p_arr = np.array(yhat)
    a_arr = np.array(actual)

    w = int(best_model['future_days'])
    l = int(len(a_arr) / w)
    p_arr = p_arr.reshape(l, w)
    a_arr = a_arr.reshape(l, w)
    diff = a_arr - p_arr
    std = np.std(diff, axis=0)

    # Directional accuracy
    qf = pd.DataFrame(a_arr - p_arr)
    qf[qf > 0] = 0
    qf[qf < 0] = 1
    lpm0 = qf.sum() / len(qf)

    # LPM1
    qf1 = pd.DataFrame(a_arr - p_arr)
    qf1[qf1 > 0] = 0
    lpm1 = qf1.mean()

    # LPM2
    qf2 = pd.DataFrame(a_arr - p_arr)
    qf2[qf2 > 0] = 0
    lpm2 = qf2.std()
    fd_index = list(np.array(list(range(w))) + 1)
    fd_index = ["lag" + ' ' + str(x) for x in fd_index]

    return std, lpm0, lpm1, lpm2, fd_index


def _directional_accuracy(actual, predicted, best_model, is_targeted: bool = False):
    """
    Calculates the directional accuracy of predicted values compared to actual values.
    Returns a DataFrame with accuracy metrics.
    """
    # Extracting relevant information from best_model
    window_size = int(best_model['future_days'])

    # Preparing the actual DataFrame
    actual_array = np.reshape(actual, (actual.shape[0],))
    actual_reshaped = actual_array.reshape(-1, window_size)
    actual_df = pd.DataFrame(actual_reshaped)

    # Preparing the predicted DataFrame
    predicted_array = np.reshape(predicted, (predicted.shape[0],))
    predicted_reshaped = predicted_array.reshape(-1, window_size)
    predicted_df = pd.DataFrame(predicted_reshaped)

    if is_targeted:
        diff = actual_df + predicted_df
        combined_df = diff.copy()
        def compute_dma(combined_df):
            combined_df[diff==0] = 0
            combined_df[abs(diff)==2] = 1
            combined_df[abs(diff)==1] = np.nan
            return combined_df.mean()
        total_accuracy = compute_dma(combined_df)
        accuracy_6_months = compute_dma(combined_df.tail(126))
        accuracy_1_year = compute_dma(combined_df.tail(252))
        accuracy_6_months = combined_df.tail(126).sum() / len(combined_df.tail(126))
        accuracy_1_year = combined_df.tail(252).sum() / len(combined_df.tail(252))
        q=1

    else:
        # Creating column labels for dataframes
        columns = actual_df.columns.tolist()
        new_columns = [-1] + columns

        # Shifting values for comparison with day 0
        actual_df[-1] = actual_df[0].shift()
        true_zero_vals = actual_df[-1].copy()

        # Dropping NaN values and reordering columns
        actual_df.dropna(inplace=True)
        actual_df = actual_df[new_columns]

        # Computing the directional accuracy for actual values
        for x in actual_df.index:
            for y in columns:
                actual_df.loc[x, y] = actual_df.loc[x, y] - actual_df.loc[x, -1]
        actual_df = actual_df[columns]
        actual_df[actual_df > 0] = 1
        actual_df[actual_df < 0] = -1

        # Computing the directional accuracy for predicted values
        predicted_df[-1] = true_zero_vals
        predicted_df.dropna(inplace=True)
        predicted_df = predicted_df[new_columns]

        # Computing the directional accuracy for predicted values
        for x in predicted_df.index:
            for y in columns:
                predicted_df.loc[x, y] = predicted_df.loc[x, y] - predicted_df.loc[x, -1]
        predicted_df = predicted_df[columns]
        predicted_df[predicted_df > 0] = 1
        predicted_df[predicted_df < 0] = -1

        # Calculating overall directional accuracy
        combined_df = actual_df + predicted_df
        combined_df[combined_df != 0] = 1
        total_accuracy = combined_df.sum() / len(combined_df)

        # Calculating directional accuracy for different time periods
        accuracy_6_months = combined_df.tail(126).sum() / len(combined_df.tail(126))
        accuracy_1_year = combined_df.tail(252).sum() / len(combined_df.tail(252))

        # Calculating accuracy for long positions
        long_accuracy = ((actual_df[predicted_df > 0] + predicted_df[predicted_df > 0]) / 2).mean()

        # Calculating accuracy for short positions
        short_accuracy = ((actual_df[predicted_df < 0] + predicted_df[predicted_df < 0]) / 2).mean() * -1

    # Creating the resulting DataFrame
    result_df = pd.DataFrame(total_accuracy)
    result_df = pd.concat([result_df, accuracy_6_months, accuracy_1_year, long_accuracy, short_accuracy], axis=1)
    result_df.columns = ['Directional accuracy total', '6 months', '1 year', 'Long', 'Short']

    return result_df
