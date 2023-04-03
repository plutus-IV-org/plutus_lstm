"""
This module searches for existing saved models in model_vault, offers to chose one model to plot.
Bear in mind this process recreates saved data, therefore doesn't enrich new data from data_service.
"""

import os
from PATH_CONFIG import _ROOT_PATH
from utilities.service_functions import _slash_conversion
import threading
from utilities.name_catcher import check_file_name_in_list
import pickle
from app_service.app import generate_app


root = _ROOT_PATH()
sl = _slash_conversion()
folder_path = root + sl + 'vaults' + sl + 'model_vault' + sl + 'LSTM_research_models'
files = os.listdir(folder_path)

print(f"Available models to plot")
for x in files:
    print(x)
while True:
    short_name = input("Please specify short name (or type 'END' to exit): ").lower()
    if short_name == 'end':
        break
    else:
        value = check_file_name_in_list(str(short_name),files)
        if value is None:
            print("Please specify short name again.")
            continue
        else:
            file_full_path = folder_path + sl + value + sl + "lstm_research_dict.pickle"
            # Load the pickled file using the pickle module
            with open(file_full_path, "rb") as f:
                loaded_dict = pickle.load(f)
            threading.Thread(target=generate_app, daemon=True, args=(loaded_dict,)).start()
            continue