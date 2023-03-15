from initiators.initiate_full_research import InitiateResearch
import pandas as pd
import glob

testing = True

if testing == True:
    df = pd.read_excel(r'inputs/init_test.xlsx')
    # df = pd.read_json(r'service_space/asset_config_test.json')
else:
    df = pd.read_excel(r'inputs/init.xlsx')
    # df = pd.read_json(r'service_space/asset_config.json')

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

    source = df.loc[x,'SOURCE']
    asset = df.loc[x, 'ASSET']
    df_type = df.loc[x, 'TYPE']
    pd = int(df.loc[x,'PAST'])
    fd = int(df.loc[x,'FUTURE'])
    epo = int(df.loc[x,'EPOCHS'])
    interval = df.loc[x, 'INTERVAL']

    h= InitiateResearch(asset, df_type, [pd], [fd], epo, testing, source, interval)
    q1 = h._initialize_training()


    q=1