from Const import DA_TABLE

DIRECTIONAL_ACCURACY_TABLE_SCHEMA = f"""CREATE TABLE IF NOT EXISTS {DA_TABLE.split('.'[0])} (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        model_name TEXT,
                                        da_score REAL,
                                        da_list TEXT,
                                        date TIMESTAMP
                                    );"""