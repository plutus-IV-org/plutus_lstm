import pandas as pd
#from data_preparer_new import data_preparation

def generate_data(cluster):
    """
    Types of clusters:
    stocks
    bonds
    commodities
    etf
    forex
    indecies
    crypto
    """
    if type(cluster)==str:
        asset_prediction_dic = {}
        asset_price_dic = {}
        asset_names = []
        '''
        prepared_data, interval = data_preparation(cluster)
        keys = []
        for x in prepared_data.keys():
            keys.append(x)

        #crate avarage line
        names = []
        for i in keys:
            names.append(i.split('_')[0])
        names = set(names).copy()

        def returnSum(myDict):
            list = []
            for i in myDict:
                list.append(myDict[i])
            final = sum(list)
            return final
        for n in names:
            value = 0
            dfa = {}
            for i in keys:
                if i.startswith(n):
                    value +=1
                    dfa[value,prepared_data[i][1].shape[1] ] = prepared_data[i][1]
            l1 = []
            for h in dfa.keys():
                l1.append(list(h)[1])
            l2 = set(l1)
            for t in l2:
                dfa2 = {}
                val =0
                for h in dfa.keys():
                    if list(h)[1]==t:
                        val +=1
                        dfa2[val] = dfa[h]
                summed = returnSum(dfa2)
                dic_sum = summed.dropna()/val
                prepared_data[n + '_average_' + str(t) + '_days'] = prepared_data[next(iter(prepared_data))][0], dic_sum
                keys.append(n + '_average_' + str(t) + '_days')

        for x in keys:
            asset_name = x.split('_')[0]
            asset_names.append(asset_name)
            asset_names = list(set(asset_names))

            asset_price = prepared_data[x][0][asset_name]
            asset_price = asset_price.to_frame()
            asset_price = asset_price.set_index(pd.to_datetime(asset_price.index))

            asset_prediction = pd.DataFrame(prepared_data[x][1])
            #if x ==keys[-1]:
            #    q=1
            #asset_prediction.index = asset_price.index
            #asset_prediction.set_index(pd.to_datetime(asset_prediction.index))
            asset_prediction_dic[x] = asset_prediction
            asset_price_dic[asset_name] = asset_price
            '''
        return asset_prediction_dic, asset_price_dic, asset_names, #interval
    else:
        asset_dict= cluster.copy()
        short_name = asset_dict['asset']
        full_name = asset_dict['raw_model_path'].split('\\')[-2]
        interval = asset_dict['interval']

        predicted_price = asset_dict['yhat'].reshape(int(len(asset_dict['yhat'])/asset_dict['future'][0]),asset_dict['future'][0])
        actual_price = asset_dict['data_table'].tail(len(predicted_price))
        time_index = actual_price.index
        actual_price_df = pd.DataFrame(actual_price, index=time_index)
        predicted_price_df = pd.DataFrame(predicted_price, index = time_index)

        asset_prediction_dic = {full_name:predicted_price_df.tail(126)}
        asset_price_dic = {short_name:actual_price_df[['Close']].tail(126)}
        asset_names = [short_name]
        return asset_prediction_dic, asset_price_dic, asset_names, interval