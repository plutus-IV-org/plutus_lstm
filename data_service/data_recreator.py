import datetime as dt
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model
from data_service.data_preparation import DataPreparation
from data_service.data_transformation import _data_normalisation, _data_denormalisation, _split_data
pd.options.mode.chained_assignment = None
from PATH_CONFIG import _ROOT_PATH
from utilities.service_functions import _slash_conversion


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
        trainX, trainY, testX, testY= _split_data(normalised_data, f_d, p_d)
        #Loading model
        absolute_path = path + '\\' + a + '\\LSTM.h5'
        model = load_model(absolute_path)
        predictions = model.predict(testX, verbose=0)
        df_predictions = pd.DataFrame(predictions)
        df_predictions.index = df.iloc[-predictions.shape[0]:].index
        denormalised = _data_denormalisation(df_predictions.values, df_prices, f_d,testY)
        df_pred = pd.DataFrame(denormalised).iloc[-126:]
        df_pred.index = data.iloc[-126:].index
        pred_saver[a] =df_pred
        interval = a.split('_')[5]

    #Common table
    df = ct.loc[:,~ct.columns.duplicated()].copy()
    for x in pred_saver.keys():
        pred_saver[x] = df.iloc[-126:],pred_saver[x]
    t2 = dt.datetime.now()
    delta = t2 - t1
    print(delta)
    return pred_saver, interval







