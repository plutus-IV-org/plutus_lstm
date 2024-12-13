from tensorflow.keras.metrics import RootMeanSquaredError, mean_absolute_percentage_error
from utilities.directiona_accuracy_utililities import weights
import tensorflow as tf
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

    actual = tf.cast(actual, dtype=tf.float32)
    prediction = tf.cast(prediction, dtype=tf.float32)
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


def _directional_accuracy(actual, predicted, best_model, is_targeted: bool = False, reshape_required: bool = True):
    """
    Calculates the directional accuracy of predicted values compared to actual values.
    Returns a DataFrame with accuracy metrics.
    """
    # Extracting relevant information from best_model
    window_size = int(best_model['future_days'])
    actual = actual.copy()
    predicted = predicted.copy()
    if reshape_required:
        # Preparing the actual DataFrame
        actual_array = np.reshape(actual, (actual.shape[0],))
        actual_reshaped = actual_array.reshape(-1, window_size)
        actual_df = pd.DataFrame(actual_reshaped)

        # Preparing the predicted DataFrame
        predicted_array = np.reshape(predicted, (predicted.shape[0],))
        predicted_reshaped = predicted_array.reshape(-1, window_size)
        predicted_df = pd.DataFrame(predicted_reshaped)
    else:
        actual_df = actual.copy()
        predicted_df = predicted

    if is_targeted:
        diff = actual_df + predicted_df
        combined_df = diff.copy()

        def compute_dma(combined_df: pd.DataFrame):
            combined_df[diff == 0] = 0
            combined_df[abs(diff) == 2] = 1
            combined_df[abs(diff) == 1] = np.nan
            trades_coverage_array = (~np.isnan(combined_df)).astype(int).mean()
            return combined_df.mean(), trades_coverage_array, combined_df

        total_accuracy, total_cov ,combined_df= compute_dma(combined_df.copy())
        accuracy_1_year, one_year_cov,combined_df = compute_dma(combined_df.tail(252).copy())
        accuracy_6_months, six_month_cov, combined_df = compute_dma(combined_df.tail(126).copy())

        # Old calcs
        # long_accuracy = diff.copy()
        # long_accuracy[diff == 0] = 0
        # long_accuracy[diff == 2] = 1
        # long_accuracy[abs(diff) == 1] = np.nan
        # long_accuracy[diff == -2] = np.nan
        # long_cov = (~np.isnan(long_accuracy)).astype(int).mean()
        # long_accuracy = long_accuracy.mean()

        # short_accuracy = diff.copy()
        # short_accuracy[diff == 0] = 0
        # short_accuracy[diff == -2] = 1
        # short_accuracy[abs(diff) == 1] = np.nan
        # short_accuracy[diff == 2] = np.nan
        # short_cov = (~np.isnan(short_accuracy)).astype(int).mean()
        # short_accuracy = short_accuracy.mean()

        # New calcs
        long_accuracy_df = diff.copy()
        long_accuracy = (long_accuracy_df[predicted_df == 1]/2).mean()
        long_cov = (~np.isnan(long_accuracy_df[predicted_df == 1])).astype(int).mean()

        short_accuracy_df = diff.copy()
        short_accuracy = (short_accuracy_df[predicted_df == -1] / 2).mean() * (-1)
        short_cov = (~np.isnan(short_accuracy_df[predicted_df == -1])).astype(int).mean()

    else:
        try:
            columns = actual_df.columns.tolist()
            new_columns = [-1] + columns

            actual_df[-1] = actual_df[0].shift()
            true_zero_vals = actual_df[-1].copy()

            actual_df.dropna(inplace=True)
            actual_df = actual_df[new_columns]

            for x in actual_df.index:
                for y in columns:
                    actual_df.loc[x, y] = actual_df.loc[x, y] - actual_df.loc[x, -1]
            actual_df = actual_df[columns]
            actual_df[actual_df > 0] = 1
            actual_df[actual_df < 0] = -1

            predicted_df[-1] = true_zero_vals
            predicted_df.dropna(inplace=True)
            predicted_df = predicted_df[new_columns]

            for x in predicted_df.index:
                for y in columns:
                    predicted_df.loc[x, y] = predicted_df.loc[x, y] - predicted_df.loc[x, -1]
            predicted_df = predicted_df[columns]
            predicted_df[predicted_df > 0] = 1
            predicted_df[predicted_df < 0] = -1
        except Exception:
            pass
        combined_df = actual_df + predicted_df
        combined_df[combined_df != 0] = 1
        total_accuracy = combined_df.sum() / len(combined_df)
        accuracy_6_months = combined_df.tail(126).sum() / len(combined_df.tail(126))
        accuracy_1_year = combined_df.tail(252).sum() / len(combined_df.tail(252))

        total_cov = (~np.isnan(combined_df)).astype(int).mean()
        six_month_cov = (~np.isnan(combined_df.tail(126))).astype(int).mean()
        one_year_cov = (~np.isnan(combined_df.tail(252))).astype(int).mean()

        long_trades = ((actual_df[predicted_df > 0] + predicted_df[predicted_df > 0]) / 2)
        long_accuracy = long_trades.mean()
        long_cov = (~np.isnan(long_trades)).astype(int).mean()

        short_trades = ((actual_df[predicted_df < 0] + predicted_df[predicted_df < 0]) / 2)
        short_accuracy = short_trades.mean() * -1
        short_cov = (~np.isnan(short_trades)).astype(int).mean()

    result_df = pd.DataFrame(total_accuracy)
    result_df = pd.concat([result_df, accuracy_6_months, accuracy_1_year, long_accuracy, short_accuracy], axis=1)
    trades_coverage_df = pd.concat([total_cov, six_month_cov, one_year_cov, long_cov, short_cov], axis=1)
    result_df.columns = ['DA total', 'L_126', 'L_252', 'Long', 'Short']
    trades_coverage_df.columns = ['Cov total', 'L_126', 'L_252', 'Long', 'Short']
    return result_df, trades_coverage_df


def directional_accuracy_score(df: pd.DataFrame, coverage: np.array):
    w = weights()
    df.replace(np.nan, 0.5, inplace=True)
    diff_df = df - 0.5
    # 13/12/2023 Added additional multipliers 0.75 and 1.25
    # To raise the importance of the accuracy over the coverage
    # We simply care more about precision
    multiplied_df = diff_df * (coverage.values * 0.75) * (w * 1.25)
    value = multiplied_df.mean().sum() * 2
    return value
