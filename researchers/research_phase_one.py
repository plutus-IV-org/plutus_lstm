from messenger_commands.messenger_commands import _send_telegram_msg, _send_telegram_photo, _dataframe_to_png, \
    _send_discord_message

from utilities.service_functions import _slash_conversion
import platform
import glob
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import talos
from talos.utils import lr_normalizer
from talos import Reporting
import os

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

# source, interval,data
def _run_training(trainX, trainY,asset,type,p_d,f_d, testing):
    if testing == True:
        from utilities.hyperparameters_test import _hyperparameters_one_layer, \
            _hyperparameters_two_layers, _hyperparameters_three_layers
    else:
        from utilities.hyperparameters import _hyperparameters_one_layer, \
            _hyperparameters_two_layers, _hyperparameters_three_layers

    results_table = pd.DataFrame()

    def model_builder_1(trainX, trainY, testX, testY, params):
        model = Sequential()
        model.add(LSTM(params['first_lstm_layer'], activation=params['activation'],
                       input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                      loss=params['loss'],
                      metrics=['accuracy'])
        history = model.fit(trainX, trainY,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            verbose=1,
                            validation_data=[testX, testY],
                            callbacks=[talos.utils.early_stopper(epochs=params["epochs"],
                                                                 monitor="val_accuracy",
                                                                 patience=15,
                                                                 min_delta=0.01)])
        return history, model

    p = _hyperparameters_one_layer()

    results_path =  "1_" + str(p_d) + '_' + str(f_d) + '_' + asset + "_" + type

    scan_object = talos.Scan(x=trainX,
                             y=trainY,
                             params=p,
                             model=model_builder_1,
                             experiment_name=results_path,
                             clear_session=True,
                             )

    absolute_path = glob.glob('**/*.csv',recursive=True)[0]
    results = Reporting(absolute_path)

    table = results.data

    table["past_days"] = p_d * len(table)
    table["future_days"] = f_d  * len(table)

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


    def model_builder_2(trainX, trainY, testX, testY, params):

        model = Sequential()
        model.add(LSTM(params['first_lstm_layer'], return_sequences=True, activation=params['activation'],
                       input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dropout(params['dropout']))
        model.add(LSTM(params['second_lstm_layer'], activation=params['activation'],
                       input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                      loss=params['loss'],
                      metrics=['accuracy'])
        history = model.fit(trainX, trainY,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            verbose=1,
                            validation_data=[testX, testY],
                            callbacks=[talos.utils.early_stopper(epochs=params["epochs"],
                                                                 monitor="val_accuracy",
                                                                 patience=15,
                                                                 min_delta=0.01)])

        return history, model

    p = _hyperparameters_two_layers()

    results_path = "2_" + str(p_d) + '_' + str(f_d) + '_' + asset + "_" + type

    scan_object = talos.Scan(x=trainX,
                             y=trainY,
                             params=p,
                             model=model_builder_2,
                             experiment_name=results_path,
                             clear_session=True,
                             )

    absolute_path = glob.glob('**/*.csv',recursive=True)[0]
    results = Reporting(absolute_path)

    table = results.data

    table["past_days"] = p_d  * len(table)
    table["future_days"] = f_d  * len(table)

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


    def model_builder_3(trainX, trainY, testX, testY, params):

        model = Sequential()
        model.add(LSTM(params['first_lstm_layer'], return_sequences=True, activation=params['activation'],
                       input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dropout(params['dropout']))
        model.add(LSTM(params['second_lstm_layer'], return_sequences=True, activation=params['activation'],
                       input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dropout(params['dropout']))
        model.add(LSTM(params['third_lstm_layer'], activation=params['activation'],
                       input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                      loss=params['loss'],
                      metrics=['accuracy'])

        history = model.fit(trainX, trainY,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            verbose=1,
                            validation_data=[testX, testY],
                            callbacks=[talos.utils.early_stopper(epochs=params["epochs"],
                                                                 monitor="val_accuracy",
                                                                 patience=15,
                                                                 min_delta=0.01)])

        return history, model

    p = _hyperparameters_three_layers()

    results_path = "3_" + str(p_d) + '_' + str(f_d) + '_' + asset + "_" + type

    scan_object = talos.Scan(x=trainX,
                             y=trainY,
                             params=p,
                             model=model_builder_3,
                             experiment_name=results_path,
                             clear_session=True,
                             )

    absolute_path = glob.glob('**/*.csv',recursive=True)[0]
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

    # Reorder columns of final table
    results_table = results_table[
        ['round_epochs', 'past_days', 'future_days', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr',
         'epochs', 'batch_size', 'dropout', 'first_lstm_layer', 'second_lstm_layer', 'third_lstm_layer',
         'optimizer', 'loss.1', 'activation', 'weight_regulizer']]

    return results_table
