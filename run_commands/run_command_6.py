"""
This module creates LSTM model graph decay.
"""

from utilities.accuracy_decay import show_decay, load_da_register

df = load_da_register()
fig, ax = show_decay('ETH-USD', df)
