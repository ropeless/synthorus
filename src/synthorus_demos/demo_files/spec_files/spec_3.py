"""
This is a spec file designed to stress the spec file interpreter.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    min_cell_size: 0,
    rng_n: 6,
    epsilon: 0.1,
    sensitivity: 1,

    roots: './datasets',

    datasources: {
        'xyz': {
            data_format: pickle,
            weight: -1,
            condition: 'X',
        },
        'abc': {
            data_format: parquet,
            sensitivity: 0,
            condition: ['A', 'C'],
        },
        'acx': {
            data_format: feather,
        },
        'bmi': {
            data_format: function,
            sensitivity: 0,  # just a lookup function
            condition: [],  # allow the datasource to provide a distribution for weight and height
            function:
            """
            int(weight / height / height * 10000 + 0.5) / 10
            if 0 < int(weight / height / height * 10000 + 0.5) / 10 < 100
            else None
            """,
            input: {
                'weight': {start: 1, stop: 1 + 5000},  # in 0.1 kg
                'height': range(1, 1 + 300),           # in cm
            }
        }
    },

    states: infer_distinct,  # default for all rvs
    rvs: {
        'X': {},
        'Y': {},
        'Z': {},
        'A': {},
        'B': {},
        'C': {},
        'bmi': {states: infer_distinct},
        'weight': {states: infer_range},
        'height': {states: infer_range},
    },

    crosstabs: [
        'xyz',
        ['A', 'B', 'C'],
        {rvs: ['A', 'X']},
        {rvs: 'C'},
        'bmi',
    ]
}
