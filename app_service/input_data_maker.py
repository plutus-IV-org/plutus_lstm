import pandas as pd
import datetime as dt
import numpy as np
from data_service.data_recreator import data_preparation
from utilities.cluster_common_line import compute_averages


def generate_data(cluster):
    if type(cluster) == str:
        asset_prediction_dic = {}
        asset_price_dic = {}
        asset_names = []

        prepared_data, interval = data_preparation(cluster)
        keys = []
        for x in prepared_data.keys():
            keys.append(x)

        t1 = dt.datetime.now()
        keys, prepared_data = compute_averages(keys, prepared_data)

        for x in keys:
            asset_name = x.split('_')[0]
            interval = x.split('_')[5]
            asset_names.append(asset_name + "_" + interval)
            asset_names = list(set(asset_names))

            asset_price = prepared_data[x][0][asset_name]
            asset_price = asset_price.to_frame()
            asset_price = asset_price.set_index(pd.to_datetime(asset_price.index))

            asset_prediction = pd.DataFrame(prepared_data[x][1])
            asset_prediction_dic[x] = asset_prediction
            asset_price_dic[asset_name + "_" + interval] = asset_price
        t2 = dt.datetime.now()
        print(f'Computing averages...{t2 - t1}')
        return asset_prediction_dic, asset_price_dic, asset_names, interval

    else:
        asset_dict = cluster.copy()
        short_name = asset_dict['asset']
        full_name = asset_dict['raw_model_path'].split('\\')[-2]
        interval = asset_dict['interval']

        if isinstance(asset_dict['yhat'], list):
            predicted_price = np.array(asset_dict['yhat']).reshape(
                int(len(asset_dict['yhat']) / asset_dict['future'][0]),
                asset_dict['future'][0])
        else:
            predicted_price = asset_dict['yhat'].reshape(int(len(asset_dict['yhat']) / asset_dict['future'][0]),
                                                         asset_dict['future'][0])
        actual_price = asset_dict['data_table'].tail(len(predicted_price))
        time_index = actual_price.index

        # step required to assign properly the time index
        # slicing time_index
        future_days = int(full_name.split('_')[4])
        sliced_time_index = time_index[:-future_days]
        sliced_predicted_price = predicted_price[future_days:]

        predicted_price_df = pd.DataFrame(sliced_predicted_price, index=sliced_time_index)
        # in case of target orientation
        if full_name.split('_')[-1] == 'T':
            real_price = asset_dict['data_table'].tail(len(predicted_price))[["Close"]]
            final_frame = pd.DataFrame()
            for t in range(future_days):
                final_frame = pd.concat([final_frame, real_price], axis=1)
            final_frame.dropna(inplace=True)
            sd = final_frame.std().values[0]
            predicted_price_df = final_frame.loc[predicted_price_df.index].values + (predicted_price_df * sd)
        # need to add n future days in order to recreate 126xN actual and predicted tables
        asset_prediction_dic = {full_name: predicted_price_df.tail(126)}
        asset_price_dic = {short_name + '_' + interval: actual_price[['Close']].tail(126 + future_days)}
        asset_names = [short_name + '_' + interval]
        return asset_prediction_dic, asset_price_dic, asset_names, interval
