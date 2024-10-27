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


def _perfect_model(testing, asset, df_normalised, table, trainX, trainY, testX, testY, epo=500,
                   is_targeted: bool = False):
    try:
        best_model = table.iloc[table['accuracy'].argmax(), :]
    except Exception:
        best_model = table
    if not testing and type(table) != dict:
        best_model_df = pd.DataFrame(best_model)
        best_model_df.columns = [datetime.now()]
        best_model_df = best_model_df.T
        best_model_df['asset'] = asset
        slash = _slash_conversion()
        _dir = _ROOT_PATH()
        slash = _slash_conversion()
        best_model_df.to_csv(_dir + slash + 'vaults' + slash + 'data_vault' + slash + 'best_models_register.csv',
                             mode='a', index=True, header=False)
    else:
        pass

    mod = Sequential()
    epo = epo

    custom_early_stopping = CustomEarlyStopping()

    if str(best_model['second_lstm_layer']) == 'nan' and str(best_model['third_lstm_layer']) == 'nan':
        mod.add(LSTM(int(best_model['first_lstm_layer']), activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        # mod.add(Dropout(best_model['dropout']))
        if is_targeted:
            mod.add(Dense(trainY.shape[1], activation='sigmoid'))
            opt = keras.optimizers.Adam(lr=best_model['lr'])
            mod.compile(opt, loss=MSE, metrics=[MSE])
        else:
            mod.add(Dense(trainY.shape[1]))
            opt = keras.optimizers.Adam(lr=best_model['lr'])
            mod.compile(opt, loss=MSE, metrics=[MSE])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=calculate_patience(best_model['lr']),
                                  verbose=1, mode='min')
        # Incrementally train the model
        for i, (trainX, trainY, valX, valY) in enumerate(zip(trainX, trainY, testX, testY)):
            print(f"Incremental Fold {i + 1}:")
            print(f"Training on {trainX.shape[0]} samples")
            history = mod.fit(trainX, trainY, batch_size=int(best_model['batch_size']), epochs=epo, verbose=1,
                              validation_data=[testX, testY], callbacks=[earlystop])

    if str(best_model['second_lstm_layer']) != 'nan' and str(best_model['third_lstm_layer']) == 'nan':
        mod.add(LSTM(int(best_model['first_lstm_layer']), return_sequences=True, activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        mod.add(LSTM(int(best_model['second_lstm_layer']), activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        if is_targeted:
            mod.add(Dense(trainY.shape[1], activation='sigmoid'))
            opt = keras.optimizers.Adam(lr=best_model['lr'])
            mod.compile(opt, loss=MSE, metrics=[MSE])
        else:
            mod.add(Dense(trainY.shape[1]))
            opt = keras.optimizers.Adam(lr=best_model['lr'])
            mod.compile(opt, loss=MSE, metrics=[MSE])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=calculate_patience(best_model['lr']),
                                  verbose=1, mode='min')
        # Incrementally train the model
        for i, (trainX, trainY, valX, valY) in enumerate(zip(trainX, trainY, testX, testY)):
            print(f"Incremental Fold {i + 1}:")
            print(f"Training on {trainX.shape[0]} samples")
            history = mod.fit(trainX, trainY, batch_size=int(best_model['batch_size']), epochs=epo, verbose=1,
                              validation_data=[testX, testY], callbacks=[earlystop])

    if str(best_model['second_lstm_layer']) != 'nan' and str(best_model['third_lstm_layer']) != 'nan':
        mod.add(LSTM(int(best_model['first_lstm_layer']), return_sequences=True, activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        mod.add(LSTM(int(best_model['second_lstm_layer']), return_sequences=True, activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        mod.add(LSTM(int(best_model['third_lstm_layer']), activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        if is_targeted:
            mod.add(Dense(trainY.shape[1], activation='sigmoid'))
            opt = keras.optimizers.Adam(lr=best_model['lr'])
            mod.compile(opt, loss=MSE, metrics=[MSE])
        else:
            mod.add(Dense(trainY.shape[1]))
            opt = keras.optimizers.Adam(lr=best_model['lr'])
            mod.compile(opt, loss=MSE, metrics=[MSE])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=calculate_patience(best_model['lr']),
                                  verbose=1, mode='min')
        # Incrementally train the model
        for i, (trainX, trainY, valX, valY) in enumerate(zip(trainX, trainY, testX, testY)):
            print(f"Incremental Fold {i + 1}:")
            print(f"Training on {trainX.shape[0]} samples")
            history = mod.fit(trainX, trainY, batch_size=int(best_model['batch_size']), epochs=epo, verbose=1,
                              validation_data=[testX, testY], callbacks=[earlystop])
    prediction = mod.predict(testX).tolist()
    return history, prediction, mod


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
