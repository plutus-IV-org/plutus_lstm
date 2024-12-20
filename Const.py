from enum import Enum
from keras.losses import LogCosh

CUT_COEFFICIENT = 15
CUT_TAIL = 650
DA_TABLE = 'da_history.db'
DAILY_ETHEREUM_DATA_TABLE = 'daily_eth.db'
DAILY_RIPPLE_DATA_TABLE = 'daily_xrp.db'
DAILY_BITCOIN_DATA_TABLE = 'daily_btc.db'
DAILY_CRYPTO_DATA_TABLE_LIST = [DAILY_ETHEREUM_DATA_TABLE, DAILY_RIPPLE_DATA_TABLE, DAILY_BITCOIN_DATA_TABLE]
CRYPTO_TICKERS = ['ETH-USD', 'XRP-USD', 'BTC-USD']
SHARE_DRIVE_DIR = "G:\My Drive\Plutus_reports"
TIME_INTERVALS = ['1d', '1h']
DAILY_ALERT_SOUND_PATH = r"C:\Users\ilsbo\PycharmProjects\plutus_lstm\Notifications\Sound\day_alert.mp3"
HOURLY_ALERT_SOUND_PATH = r"C:\Users\ilsbo\PycharmProjects\plutus_lstm\Notifications\Sound\hour_alert.mp3"

# Research params
LOSS_FUNCTION = 'mse'
METRICS = ['mse']
CROSS_VALIDATION_CHUNKS = [0.85, 0.90, 0.95, 1]
TESTING = True
DIRECTIONAL_ORIENTATION = True
USE_MEANS = False


class ModelBatches(Enum):
    _1d_10 = {'Corn_Asian_elephant': 0.25,
              'Jade_Marmot': 0.2,
              'Razzmatazz_Wasp': 0.09,
              'Chocolate_Eagle': 0.09,
              'Taupe_Whale': 0.09,
              'French_rose_Woodchuck': 0.09,
              'Bondi_blue_Sponge': 0.09
              }
    _1h_5 = {'Gold_Snails': 0.6,
             'Brass_Sloth': 0.4}
    _1h_6 = {'Olive_Killer_whale': 1}
    _1h_10 = {'Cobalt_Orangutan': 1}


class LongsBatches(Enum):
    """
    Variables names are interval n_future
    """
    _1d_10 = {'Jade_Marmot': 0.25,
              'Razzmatazz_Wasp': 0.25,
              'Taupe_Whale': 0.25,
              'Bondi_blue_Sponge': 0.25
              }
    _1h_5 = {'Gold_Snails': 0.5,
             'Brass_Sloth': 0.5,
             }
    _1h_6 = {'Olive_Killer_whale': 1
             }
    _1h_10 = {'Cobalt_Orangutan': 1
              }
    _15m_5 = {}


class ShortsBatches(Enum):
    """
    Variables names are interval n_future
    """
    _1d_10 = {'Chocolate_Eagle': 0.5,
              'Bondi_blue_Sponge': 0.5}
    _1h_5 = {}
    _1h_6 = {'Olive_Killer_whale': 1
             }
    _1h_10 = {'Cobalt_Orangutan': 1
              }
    _15m_5 = {}


class Favorite(Enum):
    first = "Hollywood_cerise_Turtle"
    second = 'Jade_Marmot'
    third = 'Corn_Asian_elephant'
    fourth = 'Sea_green_Yeti'
    fifth = 'Olivine_Caiman_lizard'
    sixth = 'Tan_Clown_fish'  # 10/01/24 1h price, rsi Tested not over-fitted model
    seventh = 'Teal_Quail'  # 21/01/23 1h 5 layers with low lr and good val loss (<loss)

# class ModelBatches(Enum):
#     _1d_10 = {
#                     }
#     _1h_5 = {}
#     _1h_6 = {}
#     _1h_10 = {}
#
#
# class LongsBatches(Enum):
#     """
#     Variables names are interval n_future
#     """
#     _1d_10 = {
#                 }
#     _1h_5 = {
#                 }
#     _1h_6 = {
#                 }
#     _1h_10 = {
#                  }
#     _15m_5 = {}
#
#
# class ShortsBatches(Enum):
#     """
#     Variables names are interval n_future
#     """
#     _1d_10 = {}
#     _1h_5 = {}
#     _1h_6 = {
#                 }
#     _1h_10 = {
#                  }
#     _15m_5 = {}
