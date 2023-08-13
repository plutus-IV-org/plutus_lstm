from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.activations import softmax, tanh, sigmoid


def _hyperparameters_one_layer():
    param_1_layer = {
        'lr': [0.001, 0.0001],
        'first_lstm_layer': [64, 128, 256],
        'batch_size': [32, 64],
        'epochs': [40],
        'optimizer': [Adam],
        'loss': ["binary_crossentropy"],
        'activation': [tanh],
        'weight_regulizer': [None]}

    return param_1_layer
def _hyperparameters_two_layers():
    param_2_layers = {
        'lr': [0.001, 0.0001],
        'first_lstm_layer': [64, 128, 256],
        'second_lstm_layer': [64, 128, 256],
        'batch_size': [32, 64],
        'epochs': [40],
        'dropout': [0.1, 0.2],
        'optimizer': [Adam],
        'loss': ["binary_crossentropy"],
        'activation': [tanh],
        'weight_regulizer': [None]}

    return param_2_layers

def _hyperparameters_three_layers():
    param_3_layers = {
        'lr': [0.001, 0.0001],
        'first_lstm_layer': [64, 128],
        'second_lstm_layer': [64, 128, 256],
        'third_lstm_layer': [64, 128],
        'batch_size': [32, 64],
        'epochs': [40],
        'dropout': [0.1, 0.2],
        'optimizer': [Adam],
        'loss': ["binary_crossentropy"],
        'activation': [tanh],
        'weight_regulizer': [None]}

    return param_3_layers