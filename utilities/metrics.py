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

def _directional_accuracy(actual,predicted, best_model):
    """
    Taking 2 dataframes and comparing each column values against the day 0, if both dataframes have the same sign
    that implies that LSTM predictions go in the same direction as actual values. As the result a dataframe is formed
    with values from 0 to 1, higher value means higher precision of the predictions.
    """
    # Actual - Day 0
    dataframe = np.reshape(actual, (actual.shape[0],))
    df_array = np.array(dataframe)
    w = int(best_model['future_days'])
    l = int(len(df_array)/w)
    df_reshaped_array = df_array.reshape(l, w)
    gg = pd.DataFrame(df_reshaped_array)
    col = gg.columns.tolist()
    new_col = [-1] + col
    gg[-1] = gg[0].shift()
    true_zero_vals = gg[-1].copy()
    gg.dropna(inplace=True)
    gg = gg[new_col]
    for x in gg.index:
        for y in col:
            gg.loc[x, y] = gg.loc[x, y] - gg.loc[x, -1]
    gg = gg[col]
    gg[gg > 0] = 1
    gg[gg < 0] = -1
    # Predicted - Day 0
    dataframe_pred = np.reshape(predicted, (predicted.shape[0],))
    df_array_pred = np.array(dataframe_pred)
    df_reshaped_array_pred = df_array_pred.reshape(l, w)
    gg2 = pd.DataFrame(df_reshaped_array_pred)
    gg2[-1] = true_zero_vals
    gg2.dropna(inplace=True)
    gg2 = gg2[new_col]
    for x in gg2.index:
        for y in col:
            gg2.loc[x, y] = gg2.loc[x, y] - gg2.loc[x, -1]
    gg2 = gg2[col]
    gg2[gg2 > 0] = 1
    gg2[gg2 < 0] = -1
    gg3 = gg + gg2
    gg3[gg3 != 0] = 1
    gg4 = gg3.sum() / len(gg3)
    gg5 = gg3.tail(126).sum() / len(gg3.tail(126))
    gg6 = gg3.tail(252).sum()/ len(gg3.tail(252))
    # Only longs accuracy
    long_accuracy = ((gg[gg2>0] + gg2[gg2>0])/2).mean()
    # Only shorts accuracy
    short_accuracy = ((gg[gg2 < 0] + gg2[gg2 < 0]) / 2).mean() *-1
    sum_frame2 = pd.DataFrame(gg4)
    sum_frame2 = pd.concat([sum_frame2, gg5, gg6, long_accuracy, short_accuracy],axis =1)
    sum_frame2.columns = ['Directional accuracy total', '6 months', '1 year', 'Long', 'Short']
    return sum_frame2