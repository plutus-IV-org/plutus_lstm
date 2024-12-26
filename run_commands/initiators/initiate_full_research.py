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

            # Evaluate the model with different epoch divisions (second phase)
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
        start_underline = '\\_'

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

            (train_pred, val_pred, test_pred, train_actual, val_actual, test_actual,
             best_confidence_level) = self._denormalize_and_select_confidence(
                training_predictions, validation_predictions, eval_predictions
            )

            # Extract best model or custom layers
            best_model = self._get_best_model_params()

            # Gradient accuracy test
            std, lpm0, lpm1, lpm2, fd_index = _gradient_accuracy_test(val_pred, val_actual, best_model)

            # Summaries for tables and directional accuracy
            gradient_results, model_summary, train_results, train_coverage, val_results, val_coverage, test_results, test_coverage = self._compile_summary_frames(
                train_actual, train_pred,
                val_actual, val_pred,
                test_actual, test_pred,
                best_model, fd_index,
                best_confidence_level
            )

            # Extract key metrics from model_summary
            dta_total = float(model_summary.loc['Directional accuracy score'].iloc[0])
            unique_name = model_summary.loc['Name'].iloc[0]

            # Store the collected data in a dictionary
            collected_data = {
                'history': history,
                'yhat': val_pred,
                'raw_yhat': validation_predictions,  # Non normalised predictions required for distribution graph
                'mod': mod,
                'gradient_results': gradient_results,
                'model_summary': model_summary,
                'train_results': train_results,
                'test_results': test_results,
                'train_coverage': train_coverage,
                'test_coverage': test_coverage,
                'val_results': val_results,
                'val_coverage': val_coverage,
                'dta_total': dta_total,
                'unique_name': unique_name,
                'epo_div_x': int(self.epo / x),
                'confidence_tail': best_confidence_level
            }

            epochs_test_collector[x] = collected_data.copy()

        return epochs_test_collector

    def _denormalize_and_select_confidence(self, train_predictions, val_predictions, test_predictions):
        """
        Denormalize predictions for training, validation, and test sets, and select the best confidence level
        for directional orientation if applicable.

        Args:
            train_predictions (ndarray): Normalized predictions for the training set.
            val_predictions (ndarray): Normalized predictions for the validation set.
            test_predictions (ndarray): Normalized predictions for the test/evaluation set.

        Returns:
            train_pred (ndarray): Denormalized training predictions.
            val_pred (ndarray): Denormalized validation predictions.
            test_pred (ndarray): Denormalized test (evaluation) predictions.
            train_actual (ndarray): Denormalized actual training set values.
            val_actual (ndarray): Denormalized actual validation set values.
            test_actual (ndarray): Denormalized actual test (evaluation) set values.
            best_confidence_level (float): Best confidence tail level (if directional_orientation=True), otherwise 0.5.
        """
        # Denormalize actual values
        train_actual = _data_denormalisation(
            self.trainY, self.data_table[['Close']], int(self.future[0]),
            self.trainY, is_targeted=self.directional_orientation
        ).reshape(-1, 1)

        val_actual = _data_denormalisation(
            self.testY, self.data_table[['Close']], int(self.future[0]),
            self.testY, is_targeted=self.directional_orientation
        ).reshape(-1, 1)

        test_actual = _data_denormalisation(
            self.evalY, self.data_table[['Close']], int(self.future[0]),
            self.testY, is_targeted=self.directional_orientation
        ).reshape(-1, 1)

        # If directional orientation is True, try different confidence levels and choose the best one
        if self.directional_orientation:
            confidence_levels = confidence_tails()
            confidence_test_collector = {}

            for tail in confidence_levels:
                # Denormalize for each confidence tail
                train_pred = _data_denormalisation(
                    train_predictions, self.data_table[['Close']], int(self.future[0]),
                    self.trainY, is_targeted=self.directional_orientation, confidence_lvl=tail
                ).reshape(-1, 1)

                test_pred = _data_denormalisation(
                    test_predictions, self.data_table[['Close']], int(self.future[0]),
                    self.testY, is_targeted=self.directional_orientation, confidence_lvl=tail
                ).reshape(-1, 1)

                val_pred = _data_denormalisation(
                    val_predictions, self.data_table[['Close']], int(self.future[0]),
                    self.testY, is_targeted=self.directional_orientation, confidence_lvl=tail
                ).reshape(-1, 1)

                # Evaluate directional accuracy for the validation set
                summary_table, trades_cov = _directional_accuracy(
                    val_actual, val_pred, {'future_days': self.future[0]},
                    is_targeted=self.directional_orientation
                )
                dta_score = directional_accuracy_score(summary_table, trades_cov)

                # Store results keyed by confidence tail
                confidence_test_collector[str(tail)] = (dta_score, tail, train_pred, val_pred, test_pred)

            # Pick the best confidence tail based on directional accuracy score
            best_confidence_level, results = max(confidence_test_collector.items(), key=lambda item: item[1][0])
            best_confidence_level = results[1]
            train_pred = results[2]
            val_pred = results[3]
            test_pred = results[4]

        else:
            # If no directional orientation, just denormalize all sets (confidence tail is meaningless)
            best_confidence_level = 0.5

            train_pred = _data_denormalisation(
                train_predictions, self.data_table[['Close']], int(self.future[0]),
                self.trainY, is_targeted=self.directional_orientation
            ).reshape(-1, 1)

            val_pred = _data_denormalisation(
                val_predictions, self.data_table[['Close']], int(self.future[0]),
                self.testY, is_targeted=self.directional_orientation
            ).reshape(-1, 1)

            test_pred = _data_denormalisation(
                test_predictions, self.data_table[['Close']], int(self.future[0]),
                self.testY, is_targeted=self.directional_orientation
            ).reshape(-1, 1)

        return train_pred, val_pred, test_pred, train_actual, val_actual, test_actual, best_confidence_level

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
            best_model = self.research_results.iloc[self.research_results[self.research_results.columns[4]].argmax(), :]
        return best_model

    def _compile_summary_frames(
            self,
            train_actual, train_pred,
            val_actual, val_pred,
            test_actual, test_pred,
            best_model, fd_index,
            best_confidence_level
    ):
        """
        Compile summary DataFrames for directional accuracy, gradient accuracy, and coverage
        for training, validation, and test sets. Also populates a final summary table with
        key metrics, including directional accuracy and confidence level.

        Args:
            train_actual (ndarray): Denormalized actual values for the training set.
            train_pred (ndarray): Denormalized predictions for the training set.
            val_actual (ndarray): Denormalized actual values for the validation set.
            val_pred (ndarray): Denormalized predictions for the validation set.
            test_actual (ndarray): Denormalized actual values for the test set.
            test_pred (ndarray): Denormalized predictions for the test set.
            best_model (pd.Series or DataFrame row): Best model parameters.
            fd_index (list or Index): Time index for storing gradient accuracy outputs.
            best_confidence_level (float): Best confidence tail level (if directional_orientation=True); else 0.5.

        Returns:
            gradient_results (pd.DataFrame): DataFrame containing gradient accuracy metrics (Std, LPM0, LPM1, LPM2).
            model_summary (pd.DataFrame): DataFrame summarizing the best model, directional accuracies, confidence level.
            val_results (pd.DataFrame): DataFrame containing directional accuracy details for the validation set.
            val_coverage (pd.DataFrame): DataFrame containing coverage info for the validation set.
        """

        # 1. Directional Accuracy Tests
        # Training set
        train_results, train_coverage = _directional_accuracy(
            train_actual, train_pred, best_model, is_targeted=self.directional_orientation
        )
        train_results.index = fd_index
        train_coverage.index = fd_index

        # Validation set
        val_results, val_coverage = _directional_accuracy(
            val_actual, val_pred, best_model, is_targeted=self.directional_orientation
        )
        val_results.index = fd_index
        val_coverage.index = fd_index

        # Test set
        test_results, test_coverage = _directional_accuracy(
            test_actual, test_pred, best_model, is_targeted=self.directional_orientation
        )
        test_results.index = fd_index
        test_coverage.index = fd_index

        # 2. Directional Accuracy Scores
        da_training = directional_accuracy_score(train_results, train_coverage)
        da_validation = directional_accuracy_score(val_results, val_coverage)
        da_test = directional_accuracy_score(test_results, test_coverage)

        da_std = np.std([da_training, da_validation, da_test])
        da_total = (da_training + da_validation + da_test) / 3 - da_std

        # 3. Gradient Accuracy Test on validation set (commonly used set for these metrics)
        std, lpm0, lpm1, lpm2, fd_index = _gradient_accuracy_test(val_pred, val_actual, best_model)
        gradient_results = pd.DataFrame({'Std': std, 'LPM 0': lpm0, 'LPM 1': lpm1, 'LPM 2': lpm2})
        gradient_results.index = fd_index

        # 4. Model Parameters
        if self.custom_layers:
            best_model_params = pd.Series(self.research_results)
        else:
            best_model_params = best_model

        # Convert best_model_params to DataFrame
        model_summary = pd.concat([best_model_params]).to_frame()

        # 5. Timeframes for training/validation/test sets
        train_start = self.list_of_index[0][-len(self.trainY):][0].date().strftime("%d/%m/%y")
        train_end = self.list_of_index[0][-len(self.trainY):][-1].date().strftime("%d/%m/%y")

        val_start = self.list_of_index[1][-len(self.testY):][0].date().strftime("%d/%m/%y")
        val_end = self.list_of_index[1][-len(self.testY):][-1].date().strftime("%d/%m/%y")

        test_start = self.list_of_index[2][-len(self.evalY):][0].date().strftime("%d/%m/%y")
        test_end = self.list_of_index[2][-len(self.evalY):][-1].date().strftime("%d/%m/%y")

        unique_name = name_generator()

        # 6. Populate model_summary with directional accuracy details
        model_summary.loc[f'DA training ({train_start} - {train_end})'] = str(round(da_training, 3))
        model_summary.loc[f'DA validation ({val_start} - {val_end})'] = str(round(da_validation, 3))
        model_summary.loc[f'DA test ({test_start} - {test_end})'] = str(round(da_test, 3))
        model_summary.loc['DA std'] = str(round(da_std, 3))
        model_summary.loc['Directional accuracy score'] = str(round(da_total, 3))
        model_summary.loc['Confidence level'] = str(best_confidence_level)
        model_summary.loc['Name'] = unique_name
        model_summary.loc['Means applies'] = self.use_means
        model_summary.loc['Selected regressors'] = str(self.columns_names.to_list())
        model_summary.columns = ['Values']
        # 7. Return relevant DataFrames
        return gradient_results, model_summary, train_results, train_coverage, val_results, val_coverage, test_results, test_coverage

    def _choose_best_result(self, epochs_test_collector):
        """
        From collected epoch tests, choose the best result based on the highest directional accuracy.
        Update class attributes with the chosen best result.
        """
        # Pick the key with the highest directional accuracy score
        best_test_key = max(epochs_test_collector, key=lambda x: epochs_test_collector[x]['dta_total'])
        best_result = epochs_test_collector[best_test_key]

        # Update class attributes with the chosen best result
        self.history = best_result['history']
        self.yhat = best_result['yhat']  # Non normalised validation predictions
        self.raw_yhat = best_result['raw_yhat']
        self.mod = best_result['mod']
        self.gradient_results = best_result['gradient_results']
        self.general_model_table = best_result['model_summary']
        self.val_results = best_result['val_results']
        self.val_coverage = best_result.get('val_coverage', None)
        self.train_results = best_result['train_results']
        self.train_coverage = best_result.get('train_coverage', None)
        self.test_results = best_result['test_results']
        self.test_coverage = best_result.get('test_coverage', None)
        self.mean_directional_accuracy = best_result['dta_total']
        self.unique_name = best_result['unique_name']
        self.epo = best_result['epo_div_x']
        self.confidence_tail = best_result['confidence_tail']

        # Additional shared class-level metadata
        self.loss_function = LOSS_FUNCTION
        self.cross_validation_chunks = CROSS_VALIDATION_CHUNKS
        self.metrics = METRICS

    def _visualize_results(self):
        """
        Visualize prediction results depending on whether the orientation is directional or not.
        """
        if not self.directional_orientation:
            _visualize_prediction_results_daily(pd.DataFrame(self.yhat), pd.DataFrame(self.testY))
            _visualize_prediction_results(pd.DataFrame(self.yhat), pd.DataFrame(self.testY))
        else:
            _visualize_probability_distribution(pd.DataFrame(self.raw_yhat))

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

        _dataframe_to_png(self.general_model_table, " ")

        if self.directional_orientation:
            # Save combined dataframes if directional orientation is used
            # sum_frame2 & trades_coverage & training/eval coverage frames assumed from previous steps
            dataframes = [
                self.train_results, self.train_coverage,
                self.val_results, self.val_coverage,
                self.test_results, self.test_coverage
            ]
            table_names = [
                "Training results",
                "Training coverage",
                "Validation results",
                "Validation coverage",
                "Test results",
                "Test coverage"

            ]
            _dataframes_to_single_png(dataframes, table_names, "combined_tables")
        else:
            _dataframe_to_png(self.val_results, "table_dir_vector")

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
                copy_dict.pop('custom_layer_ui', None)
                copy_dict.pop('history', None)
                pickle.dump(copy_dict, f)

        filename = os.path.join(new_abs_path, 'lstm_research_dict.pickle')
        save(self, filename)

    def _get_best_from_storage(self, storage):
        """
        From multiple loops, find and return the dictionary with the highest directional accuracy.
        """
        max_accuracy_key = max(storage, key=lambda x: storage[x]["mean_directional_accuracy"])
        return storage[max_accuracy_key]
