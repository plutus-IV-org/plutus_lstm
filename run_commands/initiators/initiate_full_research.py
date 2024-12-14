import os
import pickle
import pandas as pd
import numpy as np

from distutils.dir_util import copy_tree

# Project-specific imports
from PATH_CONFIG import _ROOT_PATH
from Const import LOSS_FUNCTION, CROSS_VALIDATION_CHUNKS, METRICS

from researchers.research_phase_one import _run_training
from researchers.research_phase_two import _perfect_model, _raw_model_saver
from messenger_commands.messenger_commands import (
    _send_telegram_msg,
    _dataframe_to_png,
    _send_discord_message,
    _visualize_loss_results,
    _visualize_accuracy_results,
    _visualize_prediction_results,
    _visualize_prediction_results_daily,
    _visualize_mda_results,
    _visualize_probability_distribution,
    _visualize_cross_validation_accuracy_results,
    _visualize_cross_validation_loss_results,
    _dataframes_to_single_png
)
from data_service.data_preparation import DataPreparation
from data_service.data_transformation import (
    _data_normalisation,
    _split_data,
    _distribution_type,
    _data_denormalisation,
    log_z_score_rolling,
    cross_validation_data_split
)
from utilities.metrics import (
    _rmse,
    _mape,
    _r,
    _gradient_accuracy_test,
    _directional_accuracy,
    directional_accuracy_score
)
from utilities.unique_name_generator import name_generator
from utilities.directiona_accuracy_utililities import confidence_tails
from utilities.use_mean_unitilities import apply_means
from UI.custom_layers import CustomLayerUI
from utilities.service_functions import _slash_conversion
from UI.custom_type import ListboxSelection

username = os.getlogin()


class InitiateResearch:
    """
    Class to initiate and run research/training processes on a given asset and data type.
    It handles data preparation, normalization, training (including custom layers and
    directional orientation), and evaluation. Also manages result visualization and
    model/storage saving.
    """

    def __init__(self,
                 asset: str,
                 df_type: str,
                 past_period: list,
                 future_period: list,
                 epo: int,
                 testing: bool,
                 source: str,
                 interval: str,
                 custom_layers: bool = False,
                 directional_orientation: bool = False,
                 use_means: bool = False):

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
        self.directional_orientation = directional_orientation
        self.use_means = use_means

    def _initialize_training(self):
        """
        Main method to initialize and run the entire research process, including:
         - Sending initial messages.
         - Data preparation and normalization.
         - Running training and evaluation (possibly multiple loops).
         - Visualizing results and saving the model.
        """
        self._send_init_messages()
        df = self._prepare_data()

        # Main training loop (for custom selections may run multiple times)
        loop_number = 1
        storage = {}

        while loop_number <= self.loops_to_run:
            loop_number += 1

            # Split data into training, testing, and evaluation sets (cross-validation)
            self._split_data_for_cross_validation(df)

            # Run first phase of training
            self._run_first_phase_of_training()

            # Send a message about starting second phase
            _send_discord_message(f'2nd phase for {self.asset} {self.type} has been started')

            # Evaluate the model with different epoch divisions if needed
            epochs_test_collector = self._evaluate_model_with_varied_epochs(df)

            # Select the best result based on directional accuracy
            self._choose_best_result(epochs_test_collector)

            # Visualize cross-validation results
            _visualize_cross_validation_loss_results(self.history)
            _visualize_cross_validation_accuracy_results(self.history)

            # Visualize predictions and distributions
            self._visualize_results()

            # Save the raw model and results
            self._save_model_and_results()

            # Save a backup model if performance criteria are met
            self._conditional_save_in_model_vault()

            # Store variables for this loop
            storage[self.unique_name] = vars(self).copy()

            # If custom loops needed, notify completion of loop
            if self.type == 'Custom':
                _send_discord_message(f'End of {loop_number - 1} loop. Unique name is {self.unique_name}.')

        # After loops finish, return the best performing result
        return self._get_best_from_storage(storage)

    def _send_init_messages(self):
        """Send initial messages to Discord to indicate start of training."""
        header_msg = 'TESTING' if self.testing else ''
        start_underline = 'TESTING\\_' if self.testing else '\\_'

        _send_discord_message(start_underline * 32)
        _send_discord_message(f'{header_msg} - {username} starts model training')

        str_past = str(self.past)
        str_fut = str(self.future)
        str_int = str(self.interval)
        mode_msg = 'TESTING - ' if self.testing else ''
        _send_discord_message(
            f'{mode_msg}1st phase for {self.asset} {self.type} for inputs {str_past} and outputs {str_fut} '
            f'with time interval {str_int} and source provider {str(self.source)} has been started'
        )

    def _prepare_data(self):
        """
        Download, optionally filter (custom selections), and normalize data.
        Also apply mean transformations if requested.
        """
        df = DataPreparation(self.asset, self.type, self.source, self.interval, self.past)._download_prices()
        self.columns_names = df.columns

        # Handle custom indicator selection
        if self.type == "Custom":
            allowed_indicators = df.columns.tolist()
            listbox_selector = ListboxSelection(allowed_indicators, df)
            selected_items, times_to_run = listbox_selector.get_selected_items()
            _send_discord_message(
                f'Custom type identified. Selected indicators: {selected_items}. '
                f'LSTM research will run {times_to_run} times.'
            )
            self.loops_to_run = int(times_to_run)
            df = df[selected_items].dropna()
            self.columns_names = df.columns
            if len(df) < 1500:
                raise ValueError('Too short selected data')

        # If custom layers are enabled, set them up
        if self.custom_layers:
            custom_layer_ui = CustomLayerUI()
            custom_layer_ui.show()
            # Store the UI instance so it can be accessed later
            self.custom_layer_ui = custom_layer_ui
        # Apply mean transformations if requested
        if self.use_means:
            df = apply_means(df)
        df = df.dropna()

        self.data_table = df.copy()

        # Data normalization
        normalised_data = _data_normalisation(df, self.directional_orientation)
        self.data_table_normalized = normalised_data.copy()

        distribution_type = _distribution_type(df)
        _send_discord_message(f'Got {distribution_type} distribution')

        return normalised_data

    def _split_data_for_cross_validation(self, normalised_data):
        """
        Perform cross-validation based data splitting. Updates class attributes
        with training, testing, evaluation sets.
        """
        for p_d in self.past:
            for f_d in self.future:
                (self.trainX, self.trainY,
                 self.testX, self.testY,
                 self.evalX, self.evalY,
                 self.list_of_index) = cross_validation_data_split(
                    normalised_data, f_d, p_d, is_targeted=self.directional_orientation
                )

    def _run_first_phase_of_training(self):
        """
        Run the initial training phase (phase one) or load custom layers.
        """
        if not self.custom_layers:
            self.research_results = _run_training(
                self.trainX, self.trainY, self.asset, self.type, self.past, self.future,
                self.testing, is_targeted=self.directional_orientation
            )
            _send_discord_message(
                f'1st phase for {self.asset} {self.type} has successfully finished'
            )
        else:
            # Custom layers are selected from UI
            # Assuming `custom_layer_ui` is accessible or passed at initialization time if needed
            # or store it as an attribute when created in _prepare_data.
            self.research_results = self.custom_layer_ui.custom_layers_dict

    def _evaluate_model_with_varied_epochs(self, df):
        """
        Evaluate the model with varied epoch divisions (currently just one iteration, could be extended).
        Returns a dictionary containing metrics and predictions for each epoch test scenario.
        """
        epochs_test_collector = {}

        # Evaluate with one scenario (can be extended if needed)
        for x in [1]:
            (history, training_predictions, validation_predictions, mod) = _perfect_model(
                self.testing, self.asset, self.data_table_normalized,
                self.research_results, self.trainX, self.trainY,
                self.testX, self.testY, epo=int(self.epo / x),
                is_targeted=self.directional_orientation
            )

            # If cross-validation, pick the largest batch
            if isinstance(self.trainX, list):
                self.trainX = self.trainX[-1]
                self.trainY = self.trainY[-1]
                self.testX = self.testX[-1]
                self.testY = self.testY[-1]
                self.evalX = self.evalX[-1]
                self.evalY = self.evalY[-1]

            eval_predictions = mod.predict(self.evalX, verbose=0)

            (yhat, actual, actual_training, actual_eval,
             best_confidence_level) = self._denormalize_and_select_confidence(
                df, training_predictions, validation_predictions, eval_predictions
            )

            # Compute metrics
            RMSE = _rmse(yhat, actual)
            MAPE = _mape(yhat, actual)
            R = _r(yhat, actual)

            # Extract best model or custom layers
            best_model = self._get_best_model_params()

            # Gradient accuracy test
            std, lpm0, lpm1, lpm2, fd_index = _gradient_accuracy_test(yhat, actual, best_model)

            # Summaries for tables and direction accuracy
            sum_frame, sum_frame1, sum_frame2, trades_coverage = self._compile_summary_frames(
                actual, yhat, actual_training, training_predictions,
                actual_eval, eval_predictions, best_model, fd_index, R, RMSE, MAPE,
                best_confidence_level
            )

            dta_total = float(sum_frame1.loc['Directional accuracy score'].iloc[0])
            unique_name = sum_frame1.loc['Name'].iloc[0]

            # Store collected data
            collected_data = {
                'history': history,
                'predicted_test_x': validation_predictions,
                'mod': mod,
                'yhat': yhat,
                'actual': actual,
                'RMSE': RMSE,
                'MAPE': MAPE,
                'R': R,
                'sum_frame': sum_frame,
                'sum_frame1': sum_frame1,
                'sum_frame2': sum_frame2,
                'dta_total': dta_total,
                'unique_name': unique_name,
                'epo_div_x': int(self.epo / x),
                'trades_coverage': trades_coverage,
                'confidence tail': best_confidence_level
            }
            epochs_test_collector[x] = collected_data.copy()

        return epochs_test_collector

    def _denormalize_and_select_confidence(self, df, training_predictions, validation_predictions, eval_predictions):
        """
        Denormalize predictions and select the best confidence level for directional orientation if applicable.
        Returns yhat, actual, actual_training, actual_eval, best_confidence_level
        """
        actual_training = _data_denormalisation(
            self.trainY, self.data_table[['Close']], int(self.future[0]),
            self.trainY, is_targeted=self.directional_orientation
        ).reshape(-1, 1)

        actual = _data_denormalisation(
            self.testY, self.data_table[['Close']], int(self.future[0]),
            self.testY, is_targeted=self.directional_orientation
        ).reshape(-1, 1)

        actual_eval = _data_denormalisation(
            self.evalY, self.data_table[['Close']], int(self.future[0]),
            self.testY, is_targeted=self.directional_orientation
        ).reshape(-1, 1)

        if self.directional_orientation:
            # Try different confidence levels and choose the best one
            confidence_levels = confidence_tails()
            confidence_test_collector = {}
            for tail in confidence_levels:
                training_pred = _data_denormalisation(
                    training_predictions, self.data_table[['Close']], int(self.future[0]),
                    self.trainY, is_targeted=self.directional_orientation, confidence_lvl=tail
                ).reshape(-1, 1)

                eval_pred = _data_denormalisation(
                    eval_predictions, self.data_table[['Close']], int(self.future[0]),
                    self.testY, is_targeted=self.directional_orientation, confidence_lvl=tail
                ).reshape(-1, 1)

                yhat = _data_denormalisation(
                    validation_predictions, self.data_table[['Close']], int(self.future[0]),
                    self.testY, is_targeted=self.directional_orientation, confidence_lvl=tail
                ).reshape(-1, 1)

                summary_table, trades_cov = _directional_accuracy(
                    actual, yhat, {'future_days': self.future[0]},
                    is_targeted=self.directional_orientation
                )
                dta_score = directional_accuracy_score(summary_table, trades_cov)
                confidence_test_collector[str(tail)] = (dta_score, tail, yhat, training_pred, eval_pred)

            # Choose the best confidence level
            best_confidence_level, results = max(confidence_test_collector.items(), key=lambda item: item[1][0])
            yhat = results[2]
            # Override predictions with the best found scenario
            training_predictions_denorm = results[3]
            eval_predictions_denorm = results[4]

            return yhat, actual, actual_training, actual_eval, best_confidence_level
        else:
            yhat = _data_denormalisation(
                validation_predictions, self.data_table[['Close']], int(self.future[0]),
                self.testY, is_targeted=self.directional_orientation
            ).reshape(-1, 1)
            return yhat, actual, actual_training, actual_eval, 0.5

    def _get_best_model_params(self):
        """
        Extract the best model parameters from the research results. If custom layers
        are used, return them directly.
        """
        if self.custom_layers:
            best_model = pd.Series(self.research_results)
            best_model.loc['future_days'] = self.future[0]
            best_model.loc['past_days'] = self.past[0]
        else:
            best_model = self.research_results.iloc[self.research_results['accuracy'].argmax(), :]
        return best_model

    def _compile_summary_frames(self, actual, yhat, actual_training, training_pred,
                                actual_eval, eval_pred, best_model, fd_index, R, RMSE, MAPE,
                                best_confidence_level):
        """
        Compile summary DataFrames that show metrics, directional accuracy, and coverage.
        Also updates sum_frame1 with directional accuracy scores.
        """
        sf = pd.Series({
            'Root mean squared error': RMSE,
            'Mean absolute percentage error': MAPE,
            "Linear correlation": R,
            'Confidence tail': best_confidence_level
        })

        # Directional accuracy tests
        sum_frame3, training_trades_coverage = _directional_accuracy(
            actual_training, training_pred, best_model, is_targeted=self.directional_orientation
        )
        sum_frame3.index = fd_index

        sum_frame4, eval_trades_coverage = _directional_accuracy(
            actual_eval, eval_pred, best_model, is_targeted=self.directional_orientation
        )
        sum_frame4.index = fd_index

        sum_frame2, trades_coverage = _directional_accuracy(
            actual, yhat, best_model, is_targeted=self.directional_orientation
        )
        sum_frame2.index = fd_index

        dta_test = directional_accuracy_score(sum_frame4, eval_trades_coverage)
        dta_validation = directional_accuracy_score(sum_frame2, trades_coverage)
        dta_training = directional_accuracy_score(sum_frame3, training_trades_coverage)
        dta_std = np.std([dta_training, dta_validation, dta_test])
        dta_total = (dta_training + dta_validation + dta_test) / 3 - dta_std

        # Gradient accuracy test
        std, lpm0, lpm1, lpm2, fd_index = _gradient_accuracy_test(yhat, actual, best_model)
        sum_frame = pd.DataFrame({'Std': std, 'LPM 0': lpm0, 'LPM 1': lpm1, 'LPM 2': lpm2})
        sum_frame.index = fd_index

        # Add model parameters and metrics
        if self.custom_layers:
            best_model_params = pd.Series(self.research_results)
        else:
            best_model_params = best_model

        sum_frame1 = pd.concat([best_model_params, sf]).to_frame()

        # Timeframes for training/validation/testing
        train_start = self.list_of_index[0][-len(self.trainY):][0].date().strftime("%d/%m/%y")
        train_end = self.list_of_index[0][-len(self.trainY):][-1].date().strftime("%d/%m/%y")
        validation_start = self.list_of_index[1][-len(self.testY):][0].date().strftime("%d/%m/%y")
        validation_end = self.list_of_index[1][-len(self.testY):][-1].date().strftime("%d/%m/%y")
        test_start = self.list_of_index[2][-len(self.evalY):][0].date().strftime("%d/%m/%y")
        test_end = self.list_of_index[2][-len(self.evalY):][-1].date().strftime("%d/%m/%y")

        unique_name = name_generator()

        # Populate sum_frame1 with DA details
        sum_frame1.loc[f'DA training ({train_start} - {train_end})'] = str(round(dta_training, 3))
        sum_frame1.loc[f'DA validation ({validation_start} - {validation_end})'] = str(round(dta_validation, 3))
        sum_frame1.loc[f'DA test ({test_start} - {test_end})'] = str(round(dta_test, 3))
        sum_frame1.loc['DA std'] = str(round(dta_std, 3))
        sum_frame1.loc['Directional accuracy score'] = str(round(dta_total, 3))
        sum_frame1.loc['Name'] = unique_name
        sum_frame1.loc['Means applies'] = self.use_means
        sum_frame1.loc['Selected regressors'] = str(self.columns_names.to_list())

        return sum_frame, sum_frame1, sum_frame2, trades_coverage

    def _choose_best_result(self, epochs_test_collector):
        """
        From collected epoch tests, choose the best result based on the highest directional accuracy.
        Update class attributes with the chosen best result.
        """
        best_test = max(epochs_test_collector, key=lambda x: epochs_test_collector[x]['dta_total'])
        best_result = epochs_test_collector[best_test]

        self.history = best_result['history']
        self.predicted_test_x = best_result['predicted_test_x']
        self.mod = best_result['mod']
        self.yhat = best_result['yhat']
        self.actual = best_result['actual']
        self.RMSE = best_result['RMSE']
        self.MAPE = best_result['MAPE']
        self.R = best_result['R']
        self.sum_frame = best_result['sum_frame']
        self.sum_frame1 = best_result['sum_frame1']
        self.sum_frame2 = best_result['sum_frame2']
        self.mean_directional_accuracy = best_result['dta_total']
        self.unique_name = best_result['unique_name']
        self.general_model_table = best_result['sum_frame1']
        self.epo = best_result['epo_div_x']
        self.trades_coverage = best_result['trades_coverage']
        self.confidence_tail = best_result['confidence tail']
        self.loss_function = LOSS_FUNCTION
        self.cross_validation_chunks = CROSS_VALIDATION_CHUNKS
        self.metrics = METRICS

    def _visualize_results(self):
        """
        Visualize prediction results depending on whether the orientation is directional or not.
        """
        if not self.directional_orientation:
            _visualize_prediction_results_daily(pd.DataFrame(self.predicted_test_x), pd.DataFrame(self.testY))
            _visualize_prediction_results(pd.DataFrame(self.predicted_test_x), pd.DataFrame(self.testY))
        else:
            _visualize_probability_distribution(pd.DataFrame(self.predicted_test_x))

    def _save_model_and_results(self):
        """
        Save the trained model and related results, including dataframes and
        optional aggregated tables (if directional_orientation is enabled).
        """
        self.raw_model_path = _raw_model_saver(
            self.asset, self.type, self.epo, self.past, self.future,
            self.interval, self.mean_directional_accuracy, self.source,
            self.unique_name, self.mod, is_targeted=self.directional_orientation
        )

        _dataframe_to_png(self.sum_frame1, " ")

        if self.directional_orientation:
            # Save combined dataframes if directional orientation is used
            # sum_frame2 & trades_coverage & training/eval coverage frames assumed from previous steps
            dataframes = [
                self.sum_frame2, self.trades_coverage,  # Validation results and coverage
                # training results/coverage (sum_frame3/training_trades_coverage) should be defined in context
                # eval results/coverage (sum_frame4/eval_trades_coverage) as well
                # For clarity, they should be passed or stored as attributes when compiled
            ]
            table_names = [
                "Validate results",
                "Validate coverage"
                # Add names for training/test if they are needed and available
            ]
            _dataframes_to_single_png(dataframes, table_names, "combined_tables")
        else:
            _dataframe_to_png(self.sum_frame2, "table_dir_vector")

        # Save general model table
        self.general_model_table.to_csv(self.raw_model_path[:-7] + "general_model_table.csv",
                                        index=False, header=False)

        # Copy figures to model folder
        old_abs_path = self.root_path + _slash_conversion() + 'vaults' + _slash_conversion() + 'picture_vault'
        new_abs_path = self.raw_model_path[:-7]
        copy_tree(old_abs_path, new_abs_path)

        # Save object as a pickle
        self._save_pickle(new_abs_path)

        _send_discord_message(f'2nd phase for {self.asset} {self.type} has successfully finished')

    def _save_pickle(self, new_abs_path: str):
        """Save class state as a pickle file (excluding the model itself and history)."""

        def save(obj, filename):
            with open(filename, 'wb') as f:
                copy_dict = obj.__dict__.copy()
                copy_dict.pop('mod', None)
                copy_dict.pop('history', None)
                pickle.dump(copy_dict, f)

        filename = os.path.join(new_abs_path, 'lstm_research_dict.pickle')
        save(self, filename)

    def _conditional_save_in_model_vault(self):
        """
        Conditionally save model in a model vault if performance criteria are met
        and the run is not in testing mode.
        """

        def save_model_in_model_vault():
            old_abs_path = self.raw_model_path[:-7]
            new_abs_path = (
                    self.root_path + _slash_conversion() + 'vaults' + _slash_conversion() +
                    'model_vault' + _slash_conversion() + 'LSTM_research_models' + _slash_conversion() +
                    self.raw_model_path.split('\\')[-2] + _slash_conversion()
            )
            copy_tree(old_abs_path, new_abs_path)
            _send_discord_message(self.unique_name + ' has been saved in models vault')
            print(self.unique_name + ' has been saved in models vault')

        # Criteria for saving to model vault
        if (self.R > 0.9 and self.MAPE < 5 and self.RMSE < 10 and
                self.mean_directional_accuracy > 0.51 and not self.testing):
            save_model_in_model_vault()

    def _get_best_from_storage(self, storage):
        """
        From multiple loops, find and return the dictionary with the highest directional accuracy.
        """
        max_accuracy_key = max(storage, key=lambda x: storage[x]["mean_directional_accuracy"])
        return storage[max_accuracy_key]
