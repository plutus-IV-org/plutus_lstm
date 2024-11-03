from keras.callbacks import Callback
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from Const import *
import tensorflow as tf
import keras
from PATH_CONFIG import _ROOT_PATH
from utilities.service_functions import _slash_conversion, calculate_patience
from datetime import datetime

plt.style.use('ggplot')
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


# mod = Sequential()

class CustomEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') <= logs.get('val_loss'):
            self.model.stop_training = True


# Enable eager execution
# tf.config.run_functions_eagerly(True)
def directional_loss(y_true, y_pred):
    # Calculate the difference between consecutive elements in y_true and y_pred
    y_true_cumsum = tf.cumsum(y_true, axis=1)
    y_pred_cumsum = tf.cumsum(y_pred, axis=1)

    # Calculate the signs of the differences
    true_sign = tf.sign(y_true_cumsum)
    pred_sign = tf.sign(y_pred_cumsum)

    # Calculate the absolute difference between the true and predicted signs
    sign_diff = tf.abs(true_sign - pred_sign)

    # Calculate the mean of the absolute differences
    loss = tf.reduce_mean(sign_diff) / 2.0

    return loss


# @tf.function
def mda(y_true, y_pred):
    arr_pred = tf.convert_to_tensor(y_pred)
    arr_act = tf.convert_to_tensor(y_true)
    cus_sum_pred = tf.math.cumsum(arr_pred, axis=1)
    cus_sum_actual = tf.math.cumsum(arr_act, axis=1)
    multiplication = cus_sum_actual * cus_sum_pred
    multiplication = tf.where(multiplication > 0, 1, 0)
    mean = tf.keras.backend.mean(tf.cast(multiplication, tf.float32))
    return mean


def _perfect_model(testing, asset, df_normalised, table, trainX_list, trainY_list, testX_list, testY_list, epo=500,
                   is_targeted: bool = False):
    """
    Function to train a model incrementally over different folds using cross-validation data.

    Parameters:
    - testing: bool, whether the model is in testing mode.
    - asset: str, name of the asset being trained on.
    - df_normalised: pd.DataFrame, normalized data for training/testing.
    - table: pd.DataFrame or dict, best model parameters.
    - trainX_list, trainY_list: List of training data for each fold.
    - testX_list, testY_list: List of test data for each fold.
    - epo: int, number of epochs.
    - is_targeted: bool, if True, it will use binary classification (sigmoid) for the output layer.

    Returns:
    - history: training history of the final fold.
    - prediction: model predictions on the last fold.
    - mod: trained model.
    """

    try:
        best_model = table.iloc[table['accuracy'].argmax(), :]
    except Exception:
        best_model = table

    if not testing and isinstance(table, pd.DataFrame):
        best_model_df = pd.DataFrame(best_model).T
        best_model_df.columns = [datetime.now()]
        best_model_df['asset'] = asset
        best_model_df.to_csv(_ROOT_PATH() + _slash_conversion() + 'vaults/data_vault/best_models_register.csv',
                             mode='a', index=True, header=False)

    # Initialize the model based on the first fold's input shape
    mod = Sequential()
    # Initialize an empty list to store histories
    all_fold_histories = []

    # Build the model based on best_model parameters
    if str(best_model['second_lstm_layer']) == 'nan' and str(best_model['third_lstm_layer']) == 'nan':
        mod.add(LSTM(int(best_model['first_lstm_layer']), activation='tanh',
                     input_shape=(trainX_list[0].shape[1], trainX_list[0].shape[2])))
        mod.add(Dropout(best_model['dropout']))
    elif str(best_model['second_lstm_layer']) != 'nan' and str(best_model['third_lstm_layer']) == 'nan':
        mod.add(LSTM(int(best_model['first_lstm_layer']), return_sequences=True, activation='tanh',
                     input_shape=(trainX_list[0].shape[1], trainX_list[0].shape[2])))
        mod.add(Dropout(best_model['dropout']))
        mod.add(LSTM(int(best_model['second_lstm_layer']), activation='tanh'))
        mod.add(Dropout(best_model['dropout']))
    else:
        mod.add(LSTM(int(best_model['first_lstm_layer']), return_sequences=True, activation='tanh',
                     input_shape=(trainX_list[0].shape[1], trainX_list[0].shape[2])))
        mod.add(Dropout(best_model['dropout']))
        mod.add(LSTM(int(best_model['second_lstm_layer']), return_sequences=True, activation='tanh'))
        mod.add(Dropout(best_model['dropout']))
        mod.add(LSTM(int(best_model['third_lstm_layer']), activation='tanh'))
        mod.add(Dropout(best_model['dropout']))

    # Add the output layer based on the problem type
    # if is_targeted:
    #     mod.add(Dense(trainY_list[0].shape[1], activation='sigmoid'))
    #     mod.compile(optimizer=keras.optimizers.Adam(lr=best_model['lr']), loss='binary_crossentropy',
    #                 metrics=['accuracy'])
    # else:
    mod.add(Dense(trainY_list[0].shape[1]))
    mod.compile(optimizer=keras.optimizers.Adam(lr=best_model['lr']), loss='mse', metrics=['mse'])

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=calculate_patience(best_model['lr']),
                              verbose=1, mode='min')

    # Incrementally train the model on each fold
    for i, (trainX, trainY, valX, valY) in enumerate(zip(trainX_list, trainY_list, testX_list, testY_list)):
        print(f"Incremental Fold {i + 1}:")
        print(f"Training on {trainX.shape[0]} samples")

        # Train the model incrementally (this does not reset the model)
        history = mod.fit(trainX, trainY, batch_size=int(best_model['batch_size']), epochs=epo, verbose=1,
                          validation_data=(valX, valY), callbacks=[earlystop])
        # Append history for this fold to all_fold_histories
        all_fold_histories.append(history.history)

    # Make predictions with the final trained model on the last fold
    prediction = mod.predict(testX_list[-1]).tolist()

    return all_fold_histories, prediction, mod


def _raw_model_saver(asset, df_type, epo, past, future, interval, dta, source, unique_name, mod,
                     is_targeted: bool = False):
    if is_targeted:
        file_name = asset + '_' + df_type + '_' + str(epo) + '_' + str(past[0]) + '_' + str(
            future[0]) + '_' + interval + '_A_' + str(round(dta, 2)) + '_S_' + source[
                        0] + '_' + unique_name + '_T'
    else:
        file_name = asset + '_' + df_type + '_' + str(epo) + '_' + str(past[0]) + '_' + str(
            future[0]) + '_' + interval + '_A_' + str(round(dta, 2)) + '_S_' + source[
                        0] + '_' + unique_name
    folder_path = 'raw_model_vault'
    _dir = _ROOT_PATH()
    slash = _slash_conversion()
    full_path = _dir + slash + "vaults" + slash + folder_path + slash + 'LSTM_research_models' + slash + file_name + slash + 'LSTM.h5'
    mod.save(full_path)
    return full_path
