"""
Module for download and storing data into db.
"""
import os
import sys
import logging

# Set the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from data_service.vendors_data.binance_downloader import downloader
from db_service.SQLite import technical_data_add, technical_data_load
from Const import *
from db_service.schemas.tables_schemas import *

# Configure logging with an absolute path
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, 'run_command_8_script.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')


# Redirect print statements to logging
class PrintLogger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def write(self, message):
        if message.strip() != "":
            self.logger.info(message)

    def flush(self):
        pass


sys.stdout = PrintLogger()
sys.stderr = PrintLogger()


def insert_new_data(asset_name: str, table_name: str, table_schema: str, interval: str = '1d') -> None:
    data = downloader(asset_name, interval)
    initial_data_in_db = technical_data_load(table_name)
    technical_data_add(table_name, data, table_schema)
    post_process_data_in_db = technical_data_load(table_name)

    # Additional check
    if len(initial_data_in_db) > 0:
        post_cut_data = post_process_data_in_db.head(len(initial_data_in_db))
        assert post_cut_data[initial_data_in_db != post_cut_data].sum()['Open'] == 0
        assert len(post_process_data_in_db) >= len(initial_data_in_db)
        assert post_process_data_in_db.dtypes.equals(initial_data_in_db.dtypes)
        post_process_data_in_db.Time = post_process_data_in_db.Time.astype('datetime64')
        assert (data.iloc[-1] == post_process_data_in_db.iloc[-1]).all()


def run_daily_db_process():
    print('Initiating run daily db process')
    table_lst = DAILY_CRYPTO_DATA_TABLE_LIST
    schema_lst = DAILY_CRYPTO_DATA_TABLE_SCHEMA_LIST
    tickers_lst = CRYPTO_TICKERS
    assert len(table_lst) == len(schema_lst)
    assert len(table_lst) == len(tickers_lst)
    for x in range(len(table_lst)):
        print('________________________________________________________')
        print(f'Running process for {tickers_lst[x]}, table name is {table_lst[x]}')
        insert_new_data(tickers_lst[x], table_lst[x], schema_lst[x], interval='1d')
    print('Successfully processed db requests.')


if __name__ == '__main__':
    logging.info('Logging setup complete')
    logging.info('Script started')
    try:
        run_daily_db_process()
    except Exception as e:
        logging.error(f'Error occurred: {e}')
    logging.info('Script finished')
