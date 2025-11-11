"""
A spec file to show working with ABS table builder data.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    min_cell_size: 5,
    rng_n: 6,
    epsilon: 0.1,
    sensitivity: 1,

    roots: './datasets/table_builder',

    datasources: {
        'age-sex': {
            data_format: table_builder,
            location: 'age-sex[NSW].csv',
            rvs: {
                'age': 'AGEP Age',
                'sex': 'SEXP Sex',
            }
        }
    },

    rvs: {
        'age': {states: infer_distinct},
        'sex': {states: infer_distinct},
    },

    crosstabs: [
        'age-sex',
    ]
}
