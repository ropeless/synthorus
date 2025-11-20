"""
This is a spec file demonstrating a basic configuration.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    name: 'model_1',
    comment: 'a simple demo model',
    author: 'B. Drake',

    rng_n: 6,
    min_cell_size: 0,
    epsilon: 0.1,

    datasources: {
        'xyz': {
            data_format: csv,
            weight: -1,
            sensitivity: 1,
            inline: """
                X,Y,Z,weight
                y,y,y,10
                y,y,n,6
                y,n,y,1
                y,n,n,14
                n,y,y,2
                n,y,n,5
                n,n,y,1
                n,n,n,3
                """
        }
    },

    rvs: {
        'X': {states: infer_distinct},
        'Y': {states: infer_distinct},
        'Z': {states: infer_distinct},
    },

    crosstabs: [
        'xyz',
    ]
}
