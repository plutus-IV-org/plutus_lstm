
from utilities.service_functions import _slash_conversion
"""
Put it in initializer 05.03.23

"""
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import keras
from PATH_CONFIG import _ROOT_PATH
from utilities.service_functions import _slash_conversion

plt.style.use('ggplot')
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None
#mod = Sequential()

def _perfect_model(testing, asset, df_normalised, table , trainX, trainY, testX, testY, epo=500):
    best_model = table.iloc[table['accuracy'].argmax(), :]
    if testing == False:
        from datetime import datetime
        best_model_df = pd.DataFrame(best_model)
        best_model_df.columns = [datetime.now()]
        best_model_df = best_model_df.T
        best_model_df['asset'] = asset
        slash = _slash_conversion()
        _dir = _ROOT_PATH()
        slash = _slash_conversion()
        best_model_df.to_csv(_dir + slash + 'vaults' + slash +'data_vault'+'best_models_register.csv', mode='a', index=True, header=False)
    else:
        pass

    mod = Sequential()
    epo = epo
    if str(best_model['second_lstm_layer']) == 'nan' and str(best_model['third_lstm_layer']) == 'nan':
        mod.add(LSTM(int(best_model['first_lstm_layer']), activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        # mod.add(Dropout(best_model['dropout']))
        mod.add(Dense(trainY.shape[1]))
        opt = keras.optimizers.Adam(lr=best_model['lr'])
        mod.compile(opt, loss='mse', metrics=['accuracy'])
        history = mod.fit(trainX, trainY, batch_size=int(best_model['batch_size']), epochs=epo, verbose=1,
                          validation_data=[testX, testY])
    if str(best_model['second_lstm_layer']) != 'nan' and str(best_model['third_lstm_layer']) == 'nan':
        mod.add(LSTM(int(best_model['first_lstm_layer']), return_sequences=True, activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        mod.add(LSTM(int(best_model['second_lstm_layer']), activation='tanh',
                     input_shape=(trainX.shape[1], trainX.shape[2])))
        mod.add(Dropout(best_model['dropout']))
        mod.add(Dense(trainY.shape[1]))
        opt = keras.optimizers.Adam(lr=best_model['lr'])
        mod.compile(opt, loss='mse', metrics=['accuracy'])
        history = mod.fit(trainX, trainY, batch_size=int(best_model['batch_size']), epochs=epo, verbose=1,
                          validation_data=[testX, testY])

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
        mod.add(Dense(trainY.shape[1]))
        opt = keras.optimizers.Adam(lr=best_model['lr'])
        mod.compile(opt, loss='mse', metrics=['accuracy'])
        history = mod.fit(trainX, trainY, batch_size=int(best_model['batch_size']), epochs=epo, verbose=1,
                          validation_data=[testX, testY])
    prediction =mod.predict(testX).tolist()
    return history, prediction,mod

def _raw_model_saver(asset,df_type,epo,past,future,interval,dta,source,unique_name,mod):
    file_name = asset + '_' + df_type + '_' + str(epo) + '_' + str(past[0]) + '_' + str(
        future[0]) + '_' + interval + '_A_' + str(round(dta, 2)) + '_S_' + source[
                    0] + '_' + unique_name
    folder_path = 'raw_model_vault'
    _dir= _ROOT_PATH()
    slash = _slash_conversion()
    full_path = _dir  + slash + "vaults" +slash + folder_path + slash + 'LSTM_research_models' + slash + file_name + slash + 'LSTM.h5'
    mod.save(full_path)
    return full_path

