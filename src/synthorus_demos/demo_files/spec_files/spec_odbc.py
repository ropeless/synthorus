"""
This is a spec file demonstrating accessing a database using ODBC.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    datasources: {
        'my_table': {
            data_format: odbc,
            sensitivity: 0,
            table: 'my_table',
            schema: 'my_schema',
            connection: {
                'SERVER': None,     # will use config.DB_SERVER
                'DATABASE': None,   # will use config.DB_DATABASE
                'UID': None,        # will use config.DB_UID
                'PORT': 1433,
                'AUTHENTICATION': 'ActiveDirectoryInteractive',
                'DRIVER': '{ODBC Driver 17 for SQL Server}',
            }
        }
    },

    states: infer_distinct,   # for all rvs
}
