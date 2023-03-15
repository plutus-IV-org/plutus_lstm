from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.activations import softmax, tanh, sigmoid


def _hyperparameters_one_layer():
    param_1_layer = {
        'lr': [0.0001],
        'first_lstm_layer' :[32],
        'batch_size': [32],
        'epochs': [10],
        'optimizer': [Adam],
        'loss': ['mse'],
        'activation': [tanh],
        'weight_regulizer': [None]}

    return param_1_layer

def _hyperparameters_two_layers():
    param_2_layers = {
        'lr': [0.0001],
        'first_lstm_layer' :[64],
        'second_lstm_layer' :[64],
        'batch_size': [64],
        'epochs': [10],
        'dropout': [0.1],
        'optimizer': [Adam],
        'loss': ['mse'],
        'activation': [tanh],
        'weight_regulizer': [None]}

    return param_2_layers

def _hyperparameters_three_layers():
    param_3_layers = {
        'lr': [0.0001],
        'first_lstm_layer' :[64],
        'second_lstm_layer' :[64],
        'third_lstm_layer' :[64],
        'batch_size': [32],
        'epochs': [10],
        'dropout': [0.1],
        'optimizer': [Adam],
        'loss': ['mse'],
        'activation': [tanh],
        'weight_regulizer': [None]}

    return param_3_layers

