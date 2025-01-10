from Const import DA_TABLE, DAILY_ETHEREUM_DATA_TABLE, DAILY_RIPPLE_DATA_TABLE, DAILY_BITCOIN_DATA_TABLE, \
    HOURLY_ORDERS_DATA_TABLE, FUTURES_BALANCE_TABLE, COMMISSIONS_TABLE, TRADE_HISTORY_TABLE

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

HOURLY_ORDERS_SCHEMA = f""" CREATE TABLE IF NOT EXISTS {HOURLY_ORDERS_DATA_TABLE.split('.')[0]} (
                                asset VARCHAR(50) NOT NULL,
                                time TIMESTAMP NOT NULL,
                                diff FLOAT NOT NULL,
                                target INTEGER NOT NULL,
                                verbal VARCHAR(50) NOT NULL,
                                as_of TIMESTAMP NOT NULL,
                                quantity INTEGER NOT NULL,
                                price FLOAT NOT NULL,
                                leverage INTEGER NOT NULL,
                                take_profit FLOAT NOT NULL,
                                stop_loss FLOAT NOT NULL,
                                executed BOOLEAN DEFAULT 0,
                                order_id INTEGER NOT NULL,
                                testing BOOLEAN DEFAULT 0,
                                PRIMARY KEY (asset,time, as_of)
                            );
                        """

FUTURES_BALANCE_SCHEMA = f""" CREATE TABLE IF NOT EXISTS {FUTURES_BALANCE_TABLE.split('.')[0]} (
                                total_balance FLOAT NOT NULL,
                                available_balance FLOAT NOT NULL,
                                time TIMESTAMP NOT NULL,
                                as_of TIMESTAMP NOT NULL,
                                testing BOOLEAN DEFAULT 0,
                                PRIMARY KEY (time, as_of)
                            );
                        """

COMMISSIONS_SCHEMA = f""" CREATE TABLE IF NOT EXISTS {COMMISSIONS_TABLE.split('.')[0]} (
                                time TIMESTAMP NOT NULL,
                                ticker FLOAT NOT NULL,
                                commission FLOAT NOT NULL,
                                PRIMARY KEY (time, ticker)
                            );
                        """

TRADE_HISTORY_SCHEMA = f""" CREATE TABLE IF NOT EXISTS {TRADE_HISTORY_TABLE.split('.')[0]} (
                                ticker VARCHAR(20) NOT NULL,
                                open_direction VARCHAR(10) NOT NULL,
                                close_direction VARCHAR(10) NOT NULL,
                                initial_price FLOAT NOT NULL,
                                final_price FLOAT NOT NULL,
                                quantity FLOAT NOT NULL,
                                price_diff FLOAT NOT NULL,
                                total_difference FLOAT NOT NULL,
                                commissions_total FLOAT NOT NULL,
                                leverage_1 FLOAT NOT NULL,
                                leverage_2 FLOAT NOT NULL,
                                initial_time TIMESTAMP NOT NULL,
                                final_time TIMESTAMP NOT NULL,
                                testing BOOLEAN DEFAULT 0,
                                PRIMARY KEY (initial_time, ticker)
                            );
                        """

DAILY_CRYPTO_DATA_TABLE_SCHEMA_LIST = [DAILY_ETHEREUM_DATA_TABLE_SCHEMA, DAILY_RIPPLE_DATA_TABLE_SCHEMA,
                                       DAILY_BITCOIN_DATA_TABLE_SCHEMA]
