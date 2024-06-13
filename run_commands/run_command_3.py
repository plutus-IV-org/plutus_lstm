"""
This module sorts existing models from models_vault to cluster vault as per ticker asset class.
"""
from PATH_CONFIG import _ROOT_PATH
from utilities.service_functions import _slash_conversion
import os
from distutils.dir_util import copy_tree, remove_tree



root = _ROOT_PATH()
sl = _slash_conversion()
model_vault_path = root + sl + 'vaults' + sl + 'model_vault' + sl + 'LSTM_research_models'
cluster_folder_path = root + sl + 'vaults' + sl + 'cluster'
files = os.listdir(model_vault_path)

for name in files:
    ticker = name.split('_')[0]
    if '=X' in ticker:
        end_path = 'forex'
    if '-USD' in ticker:
        end_path = 'crypto'
    if '=F' in ticker:
        end_path = 'commodities'
    if '=F' not in ticker and '-USD' not in ticker and '=X' not in ticker :
        end_path = 'stocks'
    cluster_absolute_path = cluster_folder_path + sl + end_path + sl + name
    old_absolute_path = root + sl + 'vaults' + sl + 'model_vault' + sl + 'LSTM_research_models' + sl + name
    copy_tree(old_absolute_path, cluster_absolute_path)
    # Clean old directory
remove_tree(model_vault_path)

