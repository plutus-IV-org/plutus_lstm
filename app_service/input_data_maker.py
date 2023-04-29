import pandas as pd
from data_service.data_recreator import data_preparation


def generate_data(cluster):
    if type(cluster) == str:
        asset_prediction_dic = {}
        asset_price_dic = {}
        asset_names = []

        prepared_data, interval = data_preparation(cluster)
        keys = []
        for x in prepared_data.keys():
            keys.append(x)

        # Create a set of unique names and intervals
        names = set(i.split('_')[0] for i in keys)
        intervals = set(i.split('_')[5] for i in keys)

        def returnSum(myDict):
            total = 0
            for value in myDict.values():
                total += value
            return total

        for n in names:
            for interval in intervals:
                value = 0
                dfa = {}
                for i in keys:
                    if i.startswith(n) and (i.split('_')+['dummy_ending'])[5]==interval:
                        value += 1
                        dfa[value, prepared_data[i][1].shape[1]] = prepared_data[i][1]

                l1 = []
                for h in dfa.keys():
                    l1.append(list(h)[1])
                l2 = set(l1)

                for t in l2:
                    dfa2 = {}
                    val = 0
                    for h in dfa.keys():
                        if list(h)[1] == t:
                            val += 1
                            dfa2[val] = dfa[h]

                    summed = returnSum(dfa2)
                    dic_sum = summed.dropna() / val

                    prepared_data_keys_list =  list(prepared_data.keys())
                    filtered_list = [x for x in prepared_data_keys_list if interval in x]
                    prepared_data[n + '_average_' + str(t) + '_future_steps_' + interval] = prepared_data[filtered_list[0]][0], dic_sum
                    keys.append(n + '_average_' + str(t) + '_future_steps_' + interval)

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

        return asset_prediction_dic, asset_price_dic, asset_names, interval

    else:
        asset_dict= cluster.copy()
        short_name = asset_dict['asset']
        full_name = asset_dict['raw_model_path'].split('\\')[-2]
        interval = asset_dict['interval']

        predicted_price = asset_dict['yhat'].reshape(int(len(asset_dict['yhat'])/asset_dict['future'][0]),asset_dict['future'][0])
        actual_price = asset_dict['data_table'].tail(len(predicted_price))
        time_index = actual_price.index

        # step required to assign properly the time index
        # slicing time_index
        future_days= int(full_name.split('_')[4])
        sliced_time_index = time_index[:-future_days]
        sliced_predicted_price = predicted_price[future_days:]

        predicted_price_df = pd.DataFrame(sliced_predicted_price, index = sliced_time_index )
        # need to add n future days in order to recreate 126xN actual and predicted tables
        asset_prediction_dic = {full_name:predicted_price_df.tail(126)}
        asset_price_dic = {short_name:actual_price[['Close']].tail(126+future_days)}
        asset_names = [short_name]
        return asset_prediction_dic, asset_price_dic, asset_names, interval