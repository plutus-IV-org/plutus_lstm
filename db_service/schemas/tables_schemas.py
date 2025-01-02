from Const import DA_TABLE, DAILY_ETHEREUM_DATA_TABLE, DAILY_RIPPLE_DATA_TABLE, DAILY_BITCOIN_DATA_TABLE

DIRECTIONAL_ACCURACY_TABLE_SCHEMA = f"""CREATE TABLE IF NOT EXISTS {DA_TABLE.split('.'[0])} (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        model_name TEXT,
                                        da_score REAL,
                                        da_list TEXT,
                                        date TIMESTAMP
                                    );"""

DAILY_ETHEREUM_DATA_TABLE_SCHEMA = f"""CREATE TABLE IF NOT EXISTS {DAILY_ETHEREUM_DATA_TABLE.split('.')[0]} (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        time TIMESTAMP,
                                        open REAL NOT NULL,
                                        high REAL NOT NULL,
                                        low REAL NOT NULL,
                                        close REAL NOT NULL,
                                        volume REAL NOT NULL
                                    );
                                    """

DAILY_RIPPLE_DATA_TABLE_SCHEMA = f"""CREATE TABLE IF NOT EXISTS {DAILY_RIPPLE_DATA_TABLE.split('.')[0]} (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        time TIMESTAMP,
                                        open REAL NOT NULL,
                                        high REAL NOT NULL,
                                        low REAL NOT NULL,
                                        close REAL NOT NULL,
                                        volume REAL NOT NULL
                                    );
                                    """
DAILY_BITCOIN_DATA_TABLE_SCHEMA = f"""CREATE TABLE IF NOT EXISTS {DAILY_BITCOIN_DATA_TABLE.split('.')[0]} (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        time TIMESTAMP,
                                        open REAL NOT NULL,
                                        high REAL NOT NULL,
                                        low REAL NOT NULL,
                                        close REAL NOT NULL,
                                        volume REAL NOT NULL
                                    );
                                    """

ORDERS_SCHEMA = f"""CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset VARCHAR NOT NULL,
                diff FLOAT NOT NULL,
                target INT NOT NULL,
                verbal VARCHAR NOT NULL,
                time TIMESTAMP NOT NULL,
                as_of TIMESTAMP NOT NULL
                );
                """

DAILY_CRYPTO_DATA_TABLE_SCHEMA_LIST = [DAILY_ETHEREUM_DATA_TABLE_SCHEMA, DAILY_RIPPLE_DATA_TABLE_SCHEMA,
                                       DAILY_BITCOIN_DATA_TABLE_SCHEMA]
