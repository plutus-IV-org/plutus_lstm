import datetime as dt
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model
from data_service.data_preparation import DataPreparation
from data_service.data_transformation import _data_normalisation, _data_denormalisation, _split_data
pd.options.mode.chained_assignment = None
from PATH_CONFIG import _ROOT_PATH
from utilities.service_functions import _slash_conversion
import tensorflow as tf
from UI.custom_type import ListboxSelection
import pickle


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

def data_preparation(cluster:str):
    t1=dt.datetime.now()
    sl = _slash_conversion()
    path = _ROOT_PATH() +  sl + 'vaults' + sl + 'cluster' + sl + cluster
    models_names = []
    scope_lst =[]
    for m in os.listdir(path):
        scope_lst.append(m.split('_')[0:2])
        models_names.append(m)
    #Common table required as the 1st output form dict
    ct = pd.DataFrame()
    #Creating main dict
    pred_saver = {}
    for a,b in zip(models_names,scope_lst):
        dp = DataPreparation(b[0],b[1],a.split('_')[9], a.split('_')[5],a.split('_')[3])
        data = dp._download_prices()
        if b[1]== "Custom":
            lstm_research_dict_path = path + sl + a + sl + 'lstm_research_dict.pickle'
            with open(lstm_research_dict_path, "rb") as f:
                loaded_dict = pickle.load(f)
            custom_indicators = loaded_dict['data_table'].columns.tolist()
            data = data[custom_indicators]
            data = data.dropna()
            if len(data) < 1500:
                raise Exception('Too short selected data')
        #data = _download_prices(b[0], 'Close')
        sep = a.split('_')
        f_d = int(sep[4])
        p_d = int(sep[3])
        #common_table = _download_prices(b[0], 'Open').rename(columns={'Open': sep[0]})
        common_table = data[["Close"]].rename(columns={'Close': sep[0]})
        ct = pd.concat([ct, common_table],axis =1)
        df = data.dropna()
        df_prices = df.copy().iloc[-1000:]

        # Normalisation
        normalised_data = _data_normalisation(df_prices)
        trainX, trainY, testX, testY= _split_data(normalised_data, f_d, p_d,break_point=False)
        #Loading model
        absolute_path = path + '\\' + a + '\\LSTM.h5'
        try:
            model = load_model(absolute_path)
            predictions = model.predict(testX, verbose=0)
        except:
            model = load_model(absolute_path, custom_objects={'directional_loss': directional_loss})
            predictions = model.predict(testX, verbose=0)

        denormalised = _data_denormalisation(predictions.copy(), df_prices, f_d,testY, break_point=False)
        time_index = normalised_data.tail(len(denormalised)).index
        df_pred = pd.DataFrame(denormalised, index = time_index)
        pred_saver[a] =common_table.iloc[-126:],df_pred.iloc[-126:]
        interval = a.split('_')[5]

    #Common table
    #df = ct.loc[:,~ct.columns.duplicated()].copy()
    #for x in pred_saver.keys():
    #    pred_saver[x] = common_table.iloc[-126:],pred_saver[x]
    t2 = dt.datetime.now()
    delta = t2 - t1
    print(delta)
    return pred_saver, interval







