import os
from researchers.research_phase_one import _run_training
from researchers.research_phase_two import _perfect_model, _raw_model_saver
from messenger_commands.messenger_commands import _send_telegram_msg, _dataframe_to_png, _send_discord_message
from data_service.data_preparation import DataPreparation
from data_service.data_transformation import _data_normalisation, _split_data, _distribution_type, _data_denormalisation
from messenger_commands.messenger_commands import _visualize_loss_results, _visualize_accuracy_results, \
    _visualize_prediction_results, _visualize_prediction_results_daily, _visualize_mda_results
from utilities.metrics import _rmse, _mape, _r, _gradient_accuracy_test, _directional_accuracy
from utilities.unique_name_generator import name_generator
from UI.custom_layers import CustomLayerUI
import pandas as pd
from utilities.service_functions import _slash_conversion
from PATH_CONFIG import _ROOT_PATH
from distutils.dir_util import copy_tree
import pickle
from UI.custom_type import ListboxSelection
import numpy as np

username = os.getlogin()


class InitiateResearch:
    def __init__(self, asset: str, df_type: str, past_period: list, future_period: list,
                 epo: int, testing: bool, source: str, interval: str, custom_layers: bool = False):
        """

        """
        self.asset = asset
        self.type = df_type
        self.past = past_period
        self.future = future_period
        self.epo = epo
        self.testing = testing
        self.source = source
        self.interval = interval
        self.container = {}
        self.root_path = _ROOT_PATH()
        self.loops_to_run = 1
        self.custom_layers = custom_layers

    def _initialize_training(self):

        def init_message():

            if self.testing == True:
                _send_discord_message(
                    'TESTING\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_')
                _send_discord_message('TESTING - ' + username + ' starts model training')

                str_past = str(self.past)
                str_fut = str(self.future)

                _send_discord_message(
                    'TESTING - 1st phase for ' + self.asset + ' ' + self.type + ' ' + ' for inputs ' + str_past + ' and outputs ' + str_fut + ' with time interval ' + str(
                        self.interval) + ' and source provider ' + str(self.source) + ' has been started')
            else:

                _send_discord_message(
                    '\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_')
                _send_discord_message(username + ' starts model training')

                str_past = str(self.past)
                str_fut = str(self.future)
                str_int = str(self.interval)
                _send_discord_message(
                    '1st phase for ' + self.asset + ' ' + self.type + ' ' + ' for inputs ' + str_past + ' and outputs ' + str_fut + ' with time interval ' + str_int + ' and source provider ' + str(
                        self.source) + ' has been started')
            return

        init_message()

        # Getting close table

        df = DataPreparation(self.asset, self.type, self.source, self.interval, self.past)._download_prices()

        if self.type == "Custom":
            allowed_indicators = df.columns.tolist()
            listbox_selector = ListboxSelection(allowed_indicators)
            selected_items, times_to_run = listbox_selector.get_selected_items()
            _send_discord_message(
                f'Custom type has been identified. Selected indictiors are {selected_items}. LSTM research will run {times_to_run} times.')
            self.loops_to_run = int(times_to_run)
            df = df[selected_items]
            df = df.dropna()
            if len(df) < 1500:
                raise Exception('Too short selected data')
        if self.custom_layers == True:
            custom_layer_ui = CustomLayerUI()
            custom_layer_ui.show()

        df = df.dropna()
        self.data_table = df.copy()
        # Normalisation
        normalised_data = _data_normalisation(df)
        self.data_table_normalized = normalised_data.copy()
        distribution_type = _distribution_type(df)

        text = 'Got ' + distribution_type + ' distribution'
        _send_discord_message(text)

        for p_d in self.past:
            for f_d in self.future:
                # Split and fractionating based on dc zscore
                self.trainX, self.trainY, self.testX, self.testY = _split_data(normalised_data, f_d, p_d)
        loop_number = 1
        storage = {}
        while loop_number <= self.loops_to_run:
            loop_number += 1
            if self.custom_layers == False:
                self.research_results = _run_training(self.trainX, self.trainY, self.asset,
                                                      self.type, self.past, self.future, self.testing)

                _send_discord_message('1st phase for ' + self.asset + ' ' + self.type + ' has successfully finished')
            else:
                self.research_results = custom_layer_ui.custom_layers_dict
            _send_discord_message('2nd phase for ' + self.asset + ' ' + self.type + ' has been started')

            epochs_test_collector = {}

            for x in [1]:

                history, predicted_test_x, mod = _perfect_model(self.testing, self.asset, self.data_table_normalized,
                                                                self.research_results,
                                                                self.trainX, self.trainY, self.testX, self.testY,
                                                                epo=int(self.epo / x))
                self.epo = len(history.history['loss'])
                yhat = _data_denormalisation(predicted_test_x, self.data_table[['Close']], int(self.future[0]),
                                             self.testY).reshape(-1, 1)
                actual = _data_denormalisation(self.testY, self.data_table[['Close']], int(self.future[0]),
                                               self.testY).reshape(-1, 1)

                # Metrics
                RMSE = _rmse(yhat, actual)
                MAPE = _mape(yhat, actual)
                R = _r(yhat, actual)

                sf = pd.Series({'Root mean squared error': RMSE, 'Mean absolute percentage error': MAPE,
                                "Linear correlation": R})

                # Gradiant accuracy test
                if self.custom_layers == True:
                    best_model = pd.Series(self.research_results)
                    best_model.loc['future_days'] = self.future[0]
                    best_model.loc['past_days'] = self.past[0]
                else:
                    best_model = self.research_results.iloc[self.research_results['accuracy'].argmax(), :]
                std, lpm0, lpm1, lpm2, fd_index = _gradient_accuracy_test(yhat, actual, best_model)

                sum_frame = pd.DataFrame({'Std': std, 'LPM 0': lpm0, 'LPM 1': lpm1, 'LPM 2': lpm2})
                sum_frame.index = fd_index
                sum_frame1 = pd.concat([best_model, sf])
                sum_frame1 = sum_frame1.to_frame()

                sum_frame2 = _directional_accuracy(actual, yhat, best_model)
                sum_frame2.index = fd_index
                dta = sum_frame2["Directional accuracy total"].mean()
                slash = _slash_conversion()
                unique_name = name_generator()

                sum_frame1.loc['Directional accuracy'] = str(round(dta, 3))
                sum_frame1.loc['Name'] = unique_name

                collected_data = {'history': history, 'predicted_test_x': predicted_test_x, 'mod': mod,
                                  'yhat': yhat,
                                  'actual': actual, 'RMSE': RMSE, 'MAPE': MAPE, 'R': R, 'sf': sf,
                                  'sum_frame': sum_frame,
                                  'sum_frame1': sum_frame1, 'sum_frame2': sum_frame2, 'dta': dta,
                                  'unique_name': unique_name, 'epo_div_x': int(self.epo / x)}
                epochs_test_collector[x] = collected_data.copy()
            # Find the best test result based on the highest directional total accuracy (dta)
            best_test = max(epochs_test_collector, key=lambda x: epochs_test_collector[x]['dta'])

            # Set the self variables of your class to the best test result
            self.history = epochs_test_collector[best_test]['history']
            self.predicted_test_x = epochs_test_collector[best_test]['predicted_test_x']
            self.mod = epochs_test_collector[best_test]['mod']
            self.yhat = epochs_test_collector[best_test]['yhat']
            self.actual = epochs_test_collector[best_test]['actual']
            self.RMSE = epochs_test_collector[best_test]['RMSE']
            self.MAPE = epochs_test_collector[best_test]['MAPE']
            self.R = epochs_test_collector[best_test]['R']
            self.sf = epochs_test_collector[best_test]['sf']
            self.sum_frame = epochs_test_collector[best_test]['sum_frame']
            self.sum_frame1 = epochs_test_collector[best_test]['sum_frame1']
            self.sum_frame2 = epochs_test_collector[best_test]['sum_frame2']
            self.mean_directional_accuracy = epochs_test_collector[best_test]['dta']
            self.unique_name = epochs_test_collector[best_test]['unique_name']
            self.general_model_table = epochs_test_collector[best_test]['sum_frame1']
            self.epo = epochs_test_collector[best_test]['epo_div_x']

            _visualize_loss_results(self.history)
            _visualize_accuracy_results(self.history)
            # _visualize_mda_results(self.history)
            _visualize_prediction_results_daily(pd.DataFrame(self.predicted_test_x), pd.DataFrame(self.testY))
            _visualize_prediction_results(pd.DataFrame(self.predicted_test_x), pd.DataFrame(self.testY))

            self.raw_model_path = _raw_model_saver(self.asset, self.type, self.epo, self.past, self.future,
                                                   self.interval,
                                                   self.mean_directional_accuracy, self.source,
                                                   self.unique_name, self.mod)

            _dataframe_to_png(self.sum_frame1, "table_training_details")

            _dataframe_to_png(self.sum_frame2, "table_dir_vector")

            _send_discord_message('2nd phase for ' + self.asset + ' ' + self.type + ' has successfully finished')

            self.general_model_table.to_csv(self.raw_model_path[:-7] + "general_model_table.csv", index=False,
                                            header=False)

            old_abs_path = self.root_path + _slash_conversion() + 'vaults' + _slash_conversion() + 'picture_vault'
            new_abs_path = self.raw_model_path[:-7]
            copy_tree(old_abs_path, new_abs_path)

            def save(self, filename):
                with open(filename, 'wb') as f:
                    copy_dict = self.__dict__.copy()
                    copy_dict.pop('mod')
                    copy_dict.pop('history')
                    pickle.dump(copy_dict, f)

            # specify the full path of the pickle file
            filename = os.path.join(new_abs_path, 'lstm_research_dict.pickle')

            # save the object as a pickle file in the specified directory
            save(self, filename)

            def save_model_in_model_vault():
                old_abs_path = self.raw_model_path[:-7]
                new_abs_path = self.root_path + _slash_conversion() + 'vaults' + _slash_conversion() + 'model_vault' \
                               + _slash_conversion() + 'LSTM_research_models' + _slash_conversion() + \
                               self.raw_model_path.split('\\')[-2] + _slash_conversion()
                copy_tree(old_abs_path, new_abs_path)
                _send_discord_message(self.unique_name + ' has been saved in models vault')
                print(self.unique_name + ' has been saved in models vault')
                return None

            if self.R > 0.9 and self.MAPE < 5 and self.RMSE < 10 and dta > 0.51 and self.testing == False:
                save_model_in_model_vault()

            storage[self.unique_name] = vars(self).copy()
            self.epo = best_test * self.epo
            if self.type == 'Custom':
                _send_discord_message(f'End of {loop_number - 1} loop. Unique name is {self.unique_name}.')
        # Find the dictionary with the highest mean_directional_accuracy
        max_accuracy_dict_key = max(storage, key=lambda x: storage[x]["mean_directional_accuracy"])

        # Get the dictionary with the highest mean_directional_accuracy
        max_accuracy_dict = storage[max_accuracy_dict_key]

        self.loops_to_run = 1
        return max_accuracy_dict
