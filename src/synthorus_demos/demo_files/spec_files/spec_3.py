"""
This is a spec file designed to stress the spec file interpreter.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    rng_n: 6,
    epsilon: 0.1,
    sensitivity: 1,

    roots: './datasets',

    datasources: {
        'xyz': {
            location: 'xyz_weight.pkl',
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
            max(0, min(100, int(weight / height / height * 10000 + 0.5)))   # clamp to range 0-100.
            """,
            input: {
                'weight': {start: 2, stop: 500, step: 0.5},  # in kg
                'height': range(1, 280),                     # in cm
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
        'bmi': {states: infer_range},
        'weight': {states: infer_distinct},
        'height': {states: range(1, 300)},
    },

    crosstabs: [
        'xyz',
        ['A', 'B', 'C'],
        {rvs: ['A', 'X']},
        {rvs: 'C'},
        'bmi',
    ]
}
