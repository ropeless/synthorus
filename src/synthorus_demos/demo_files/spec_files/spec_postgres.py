"""
This is a spec file demonstrating accessing a Postgres database.

"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,
    min_cell_size: 0,

    datasources: {
        'home_rentals': {
            data_format: postgres,
            table: 'home_rentals',
            schema: 'demo',
            connection: {
                'user': 'demo_user',
                'password': 'demo_password',
                'host': 'samples.mindsdb.com',
                'dbname': 'demo',
            }
        }
    },

    states: infer_distinct,   # for all rvs

    crosstabs: [
        'home_rentals',
    ]
}
