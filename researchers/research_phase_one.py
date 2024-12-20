from utilities.service_functions import _slash_conversion
from Const import *
import platform
import glob
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import talos
from talos.utils import lr_normalizer
from talos import Reporting
import os

import tensorflow as tf
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


# source, interval,data
def _run_training(trainX, trainY, asset, type, p_d, f_d, testing, is_targeted: bool = False):
    # Enable eager execution
    # tf.config.run_functions_eagerly(True)
    if TESTING:
        from utilities.hyperparameters_test import _hyperparameters_one_layer, \
            _hyperparameters_two_layers, _hyperparameters_three_layers
    else:
        from utilities.hyperparameters import _hyperparameters_one_layer, \
            _hyperparameters_two_layers, _hyperparameters_three_layers

    results_table = pd.DataFrame()

    # In case of cross validation we need to extract full train X and Y
    if isinstance(trainX, list):
        trainX, trainY = trainX[-1], trainY[-1]

    # @tf.function
    def inverse_mda(y_true, y_pred):
        arr_pred = tf.convert_to_tensor(y_pred)
        arr_act = tf.convert_to_tensor(y_true)
        cus_sum_pred = tf.math.cumsum(arr_pred, axis=1)
        cus_sum_actual = tf.math.cumsum(arr_act, axis=1)
        multiplication = cus_sum_actual * cus_sum_pred
        stop_grad_multiplication = tf.stop_gradient(multiplication)
        multiplication = tf.where(stop_grad_multiplication > 0, 1.0, 0.0)
        mean = tf.keras.backend.mean(tf.cast(multiplication, tf.float32))
        epsilon = 1e-7  # A small constant to prevent division by zero
        inv_mean = 1 / (mean + epsilon)
        return inv_mean

    # Define a custom metric function
    # @tf.function
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

    class Model_1:
        def __init__(self, is_targeted: bool = False):
            self.is_targeted = is_targeted

        def model_builder_1(self, trainX, trainY, testX, testY, params):
            trainX = tf.cast(trainX, tf.float32)
            trainY = tf.cast(trainY, tf.float32)
            testX = tf.cast(testX, tf.float32)
            testY = tf.cast(testY, tf.float32)
            model = Sequential()
            model.add(LSTM(params['first_lstm_layer'], activation=params['activation'],
                           input_shape=(trainX.shape[1], trainX.shape[2])))
            if self.is_targeted:  # use self.is_targeted instead of is_targeted
                model.add(Dense(trainY.shape[1], activation='sigmoid'))  # correct placement of bracket
                model.compile(params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                              loss=LOSS_FUNCTION,
                              metrics=METRICS)
            else:
                model.add(Dense(trainY.shape[1]))
                model.compile(params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                              loss=LOSS_FUNCTION,
                              metrics=METRICS)
            history = model.fit(trainX, trainY,
                                batch_size=params['batch_size'],
                                epochs=params['epochs'],
                                verbose=1,
                                validation_data=(testX, testY),  # it should be a tuple not a list
                                callbacks=[talos.utils.early_stopper(epochs=params["epochs"],
                                                                     monitor=LOSS_FUNCTION,
                                                                     patience=15,
                                                                     min_delta=0.01)])
            return history, model

    p = _hyperparameters_one_layer()

    results_path = "1_" + str(p_d) + '_' + str(f_d) + '_' + asset + "_" + type

    model_1 = Model_1(is_targeted=is_targeted)
    scan_object = talos.Scan(x=trainX,
                             y=trainY,
                             params=p,
                             model=model_1.model_builder_1,
                             experiment_name=results_path,
                             clear_session=True,
                             )

    absolute_path = glob.glob('**/*.csv', recursive=True)[0]
    results = Reporting(absolute_path)

    table = results.data

    table["past_days"] = p_d * len(table)
    table["future_days"] = f_d * len(table)

    results_table = results_table.append(table)
    cwd = os.getcwd()

    slash = _slash_conversion()

    cwd_fold = cwd + slash + results_path
    for filename in os.listdir(cwd_fold):
        f = os.path.join(cwd_fold, filename)
        if os.path.isfile(f):
            file_path = cwd_fold + slash + filename
            os.remove(file_path)
    os.rmdir(cwd_fold)

    class Model_2:
        def __init__(self, is_targeted: bool = False):
            self.is_targeted = is_targeted

        def model_builder_2(self, trainX, trainY, testX, testY, params):
            model = Sequential()
            model.add(LSTM(params['first_lstm_layer'], return_sequences=True, activation=params['activation'],
                           input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Dropout(params['dropout']))
            model.add(LSTM(params['second_lstm_layer'], activation=params['activation'],
                           input_shape=(trainX.shape[1], trainX.shape[2])))
            if self.is_targeted:  # use self.is_targeted instead of is_targeted
                model.add(Dense(trainY.shape[1], activation='sigmoid'))  # correct placement of bracket
                model.compile(params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                              loss=LOSS_FUNCTION,
                              metrics=METRICS)
            else:
                model.add(Dense(trainY.shape[1]))
                model.compile(params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                              loss=LOSS_FUNCTION,
                              metrics=METRICS)
            history = model.fit(trainX, trainY,
                                batch_size=params['batch_size'],
                                epochs=params['epochs'],
                                verbose=1,
                                validation_data=[testX, testY],
                                callbacks=[talos.utils.early_stopper(epochs=params["epochs"],
                                                                     monitor=LOSS_FUNCTION,
                                                                     patience=15,
                                                                     min_delta=0.01)])

            return history, model

    p = _hyperparameters_two_layers()

    results_path = "2_" + str(p_d) + '_' + str(f_d) + '_' + asset + "_" + type

    model_2 = Model_2(is_targeted=is_targeted)
    scan_object = talos.Scan(x=trainX,
                             y=trainY,
                             params=p,
                             model=model_2.model_builder_2,
                             experiment_name=results_path,
                             clear_session=True,
                             )

    absolute_path = glob.glob('**/*.csv', recursive=True)[0]
    results = Reporting(absolute_path)

    table = results.data

    table["past_days"] = p_d * len(table)
    table["future_days"] = f_d * len(table)

    results_table = results_table.append(table)
    cwd = os.getcwd()

    if platform.system() == "Linux":
        slash = '/'
    else:
        slash = '\\'

    cwd_fold = cwd + slash + results_path
    for filename in os.listdir(cwd_fold):
        f = os.path.join(cwd_fold, filename)
        if os.path.isfile(f):
            file_path = cwd_fold + slash + filename
            os.remove(file_path)
    os.rmdir(cwd_fold)

    class Model_3:
        def __init__(self, is_targeted: bool = False):
            self.is_targeted = is_targeted

        def model_builder_3(self, trainX, trainY, testX, testY, params):
            model = Sequential()
            model.add(LSTM(params['first_lstm_layer'], return_sequences=True, activation=params['activation'],
                           input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Dropout(params['dropout']))
            model.add(LSTM(params['second_lstm_layer'], return_sequences=True, activation=params['activation'],
                           input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Dropout(params['dropout']))
            model.add(LSTM(params['third_lstm_layer'], activation=params['activation'],
                           input_shape=(trainX.shape[1], trainX.shape[2])))
            if self.is_targeted:  # use self.is_targeted instead of is_targeted
                model.add(Dense(trainY.shape[1], activation='sigmoid'))  # correct placement of bracket
                model.compile(params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                              loss=LOSS_FUNCTION,
                              metrics=METRICS)
            else:
                model.add(Dense(trainY.shape[1]))
                model.compile(params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                              loss=LOSS_FUNCTION,
                              metrics=METRICS)

            history = model.fit(trainX, trainY,
                                batch_size=params['batch_size'],
                                epochs=params['epochs'],
                                verbose=1,
                                validation_data=[testX, testY],
                                callbacks=[talos.utils.early_stopper(epochs=params["epochs"],
                                                                     monitor=LOSS_FUNCTION,
                                                                     patience=15,
                                                                     min_delta=0.01)])

            return history, model

    p = _hyperparameters_three_layers()

    results_path = "3_" + str(p_d) + '_' + str(f_d) + '_' + asset + "_" + type

    model_3 = Model_3(is_targeted=is_targeted)
    scan_object = talos.Scan(x=trainX,
                             y=trainY,
                             params=p,
                             model=model_3.model_builder_3,
                             experiment_name=results_path,
                             clear_session=True,
                             )

    absolute_path = glob.glob('**/*.csv', recursive=True)[0]
    results = Reporting(absolute_path)

    table = results.data

    table["past_days"] = p_d * len(table)
    table["future_days"] = f_d * len(table)

    results_table = results_table.append(table)

    cwd = os.getcwd()

    if platform.system() == "Linux":
        slash = '/'
    else:
        slash = '\\'

    cwd_fold = cwd + slash + results_path
    for filename in os.listdir(cwd_fold):
        f = os.path.join(cwd_fold, filename)
        if os.path.isfile(f):
            file_path = cwd_fold + slash + filename
            os.remove(file_path)
    os.rmdir(cwd_fold)

    metric = LOSS_FUNCTION
    val_metric = 'val_' + LOSS_FUNCTION
    # Reorder columns of final table
    results_table = results_table[
        ['round_epochs', 'past_days', 'future_days', 'loss', metric, 'val_loss', val_metric, 'lr',
         'epochs', 'batch_size', 'dropout', 'first_lstm_layer', 'second_lstm_layer', 'third_lstm_layer',
         'optimizer', 'loss.1', 'activation', 'weight_regulizer']]

    return results_table
