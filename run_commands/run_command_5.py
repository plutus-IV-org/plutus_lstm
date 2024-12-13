"""
This module runs LSTM research for specified tickers and their parameters. This process will not include first part
 of the research, and enables users via UI to input desired lstm parameters and thereafter runs second phase of
 the project.
"""

from initiators.initiate_full_research import InitiateResearch
import pandas as pd
from utilities.name_catcher import check_name_in_dict
from app_service.app import generate_app
import threading
import logging
from distutils.dir_util import copy_tree
from Const import *

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

if TESTING:
    df = pd.read_excel(r'inputs/init_test.xlsx')

else:
    df = pd.read_excel(r'inputs/init.xlsx')


self_container = {}
for x in range(len(df.index)):
    """
        -Close
        -Volume
        -Open
        -Technical
        -Indicators
        -Macro
        -Full    
    """

    source = df.loc[x, 'SOURCE']
    asset = df.loc[x, 'ASSET']
    df_type = df.loc[x, 'TYPE']
    pd = int(df.loc[x, 'PAST'])
    fd = int(df.loc[x, 'FUTURE'])
    epo = int(df.loc[x, 'EPOCHS'])
    interval = df.loc[x, 'INTERVAL']

    i = InitiateResearch(asset, df_type, [pd], [fd], epo, TESTING, source, interval, custom_layers=True,
                         directional_orientation=DIRECTIONAL_ORIENTATION, use_means=USE_MEANS)
    research_dict = i._initialize_training()
    self_container[research_dict['unique_name']] = research_dict
    """
    try:
        i = InitiateResearch(asset, df_type, [pd], [fd], epo, TESTING, source, interval)
        i._initialize_training()
        self_container[i.unique_name] = i.__dict__
    except Exception:
        pass
    """

print(f"Available models to plot {list(self_container.keys())}")
while True:
    short_name = input("Please specify short name (or type 'END' to exit): ").lower()
    if short_name == 'end':
        break
    else:
        value = check_name_in_dict(str(short_name), self_container)
        if value is None:
            print("Please specify short name again.")
            continue
        else:
            threading.Thread(target=generate_app, daemon=True, args=(self_container[value],)).start()
            print(self_container.keys())
            continue

print(f"Available models for force savings are {list(self_container.keys())}")
while True:
    short_name = input("Would you like to force save a particular model (or type 'END' to exit):").lower()
    if short_name == 'end':
        break
    else:
        value = check_name_in_dict(str(short_name), self_container)
        if value is None:
            print("Please specify short name again.")
            continue
        else:
            #
            row_model_name = self_container[value]['raw_model_path']
            old_path = self_container[value]['raw_model_path'].split('\\')[:-1]
            str_old_path = '\\'.join(old_path)
            main_vault_path_list = self_container[value]['raw_model_path'].split('\\')[:6]
            main_vault_path_list.append('model_vault')
            abs_path = main_vault_path_list + self_container[value]['raw_model_path'].split('\\')[7:-1]
            str_abs_path = '\\'.join(abs_path)
            copy_tree(str_old_path, str_abs_path)
            print(f"{value} has been saved in  model vault")
            continue
