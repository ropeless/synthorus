"""
This is a spec file designed to show the use of entities.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,          # for all datasources
    min_cell_size: 0,         # for all cross-tables
    states: infer_distinct,  # for all rvs

    datasources: {
        'patient': {
            data_format: csv,
            weight: -1,
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
        },
        'event': {
            data_format: csv,
            weight: -1,
            condition: ['X', 'Y'],
            inline: """
            X,Y,Q,weight
            y,y,1,1
            y,y,2,6
            y,n,3,1
            y,n,4,4
            n,y,5,2
            n,y,6,5
            n,n,7,1
            n,n,7,3
            """
        },
    },

    crosstabs: {
        'patient',
        'event'
    },

    # Simulation parameters
    parameters: {
        'number_of_patients': 1,
        'time_limit': 100,
    },

    # Simulation entities
    entities: {
        'patient': {
            rvs: ['X', 'Y', 'Z'],
        },
        'event': {
            parent: 'patient',
            rvs: ['Q'],
            fields: {
                'const_field': {value: 123},
                'count_field': {value: 3, sum: ['count_field', 7]},
                'cumulate_field': {value: 0, sum: ['cumulate_field', 'Q', 1]},
                'double_Q': {function: '2 * Q', input: 'Q'},
            },
            cardinality: 5
        },
    },
}
