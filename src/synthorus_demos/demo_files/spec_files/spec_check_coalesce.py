"""
A spec file to demonstrate coalescing cross-tables.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    min_cell_size: 5,
    rng_n: 6,
    epsilon: 0.1,
    sensitivity: 1,

    datasources: {
        'xyz': {
            data_format: csv,
            weight: -1,
            inline: """
                X,Y,Z,weight
                y,y,y,9
                y,y,n,10
                y,n,y,11
                y,n,n,12
                n,y,y,13
                n,y,n,14
                n,n,y,15
                n,n,n,16
                """
        }
    },

    rvs: {
        'X': {states: infer_distinct},
        'Y': {states: infer_distinct},
        'Z': {states: infer_distinct},
    },

    crosstabs: [
        ('X', 'Y'),
        ('X', 'Z'),
        ('Y', 'Z'),
    ]
}
