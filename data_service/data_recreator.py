import datetime as dt
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from data_service.data_preparation import DataPreparation
from data_service.data_transformation import _data_normalisation, _data_denormalisation, partial_data_split, _split_data
from utilities.use_mean_unitilities import apply_means

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


def data_preparation(cluster: str, time_range: int = 126):
    extended_range = time_range + 5
    sl = _slash_conversion()
    interval_lst = []
    path = _ROOT_PATH() + sl + 'vaults' + sl + 'cluster' + sl + cluster
    models_names = []
    scope_lst = []
    for m in os.listdir(path):
        scope_lst.append(m.split('_')[0:2] + [m.split('_')[5]] + [m.split('_')[3]])
        models_names.append(m)

    # Store data for unique assets in a dictionary
    unique_assets = {}
    t1 = dt.datetime.now()
    for a, b in zip(models_names, scope_lst):

        asset_name = b[0] + '_' + b[1] + '_' + b[2] + '_' + b[3]
        if asset_name not in unique_assets:
            dp = DataPreparation(asset_name, b[1], a.split('_')[9], a.split('_')[5], a.split('_')[3], True)
            unique_assets[asset_name] = dp._download_prices()

    t2 = dt.datetime.now()
    print(f'Data has been downloaded for...{t2 - t1}')
    t1 = dt.datetime.now()
    # Common table required as the 1st output form dict
    ct = pd.DataFrame()
    model_stop_watch = dt.timedelta(0)
    # Creating main dict
    pred_saver = {}
    prepared_data_input_storage = [(pd.DataFrame(), 10, 1, np.zeros(1), np.zeros(1))]
    for a, b in zip(models_names, scope_lst):
        asset_name = b[0] + '_' + b[1] + '_' + b[2] + '_' + b[3]
        data = unique_assets[asset_name]
        if b[1] == "Custom":
            lstm_research_dict_path = path + sl + a + sl + 'lstm_research_dict.pickle'
            with open(lstm_research_dict_path, "rb") as f:
                loaded_dict = pickle.load(f)

            if "Weekly_Close" in loaded_dict['data_table'].columns.tolist():
                custom_indicators = []
                for col in loaded_dict['data_table'].columns.tolist():
                    if '_' not in col:
                        custom_indicators.append(col)
                data = data[custom_indicators]
                data = apply_means(data)
                data = data.dropna()
            else:
                custom_indicators = loaded_dict['data_table'].columns.tolist()
                data = data[custom_indicators]
                data = data.dropna()

            # if len(data) < 1500:
            #     raise Exception('Too short selected data')
        # data = _download_prices(b[0], 'Close')
        sep = a.split('_')
        f_d = int(sep[4])
        p_d = int(sep[3])
        # common_table = _download_prices(b[0], 'Open').rename(columns={'Open': sep[0]})
        common_table = data[["Close"]].rename(columns={'Close': sep[0]})
        ct = pd.concat([ct, common_table], axis=1)
        df = data.dropna()
        max_past = max([int(model.split('_')[3]) for model in models_names])
        df_prices = df.copy().iloc[-(max_past + extended_range):]

        if a.split('_')[-1] == 'T':
            targeted = True
        else:
            targeted = False
        # Normalisation
        normalised_data = _data_normalisation(df_prices, is_targeted=targeted)
        testX, testY = partial_data_split(normalised_data, f_d, p_d, is_targeted=targeted)

        # Loading model
        absolute_path = path + '\\' + a + '\\LSTM.h5'
        time_model_init = dt.datetime.now()
        try:
            model = load_model(absolute_path)
            predictions = model.predict(testX[-time_range:], verbose=0)
        except:
            model = load_model(absolute_path, custom_objects={'directional_loss': directional_loss})
            predictions = model.predict(testX[-time_range:], verbose=0)
        model_stop_watch += dt.datetime.now() - time_model_init
        if 'confidence_tail' in loaded_dict.keys():
            con_tail = float(loaded_dict['confidence_tail'])
        else:
            con_tail = 0.5

        denormalised = _data_denormalisation(predictions.copy(), df_prices, f_d, testY[-time_range:], break_point=False,
                                             is_targeted=targeted, confidence_lvl=con_tail)
        time_index = normalised_data.tail(len(denormalised)).index
        df_pred = pd.DataFrame(denormalised, index=time_index)

        if targeted:
            constructed_df = pd.DataFrame()
            for t in range(f_d):
                constructed_df = pd.concat([constructed_df, df_prices.loc[df_pred.index, 'Close']], axis=1)
            sd = constructed_df.std().values[0]
            # In order to make graph prediction lines visible we need to add a
            # small random value to differentiate targeted predictions.
            random_adding = np.random.randint(-15, 15, 1)[0]
            if sd > 100:
                sd = sd + random_adding
            elif 100 > sd > 10:
                sd = sd + random_adding / 10
            elif 10 > sd > 1:
                sd = sd + random_adding / 100
            else:
                sd = sd + random_adding / 1000
            df_pred = constructed_df.loc[df_pred.index].values + (df_pred * sd)
        pred_saver[a] = common_table.iloc[-time_range:], df_pred.iloc[-time_range:]
        interval_lst.append(a.split('_')[5])

    t2 = dt.datetime.now()
    delta = t2 - t1
    print(f'{len(models_names)} models have been processed')
    print(f'Data preparation took... {delta}, time spent on model computation...{model_stop_watch}')
    return pred_saver, interval_lst
