"""
This is spec file that includes a datasource with multiple condition sources.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,
    min_cell_size: 0,
    states: infer_distinct,

    datasources: {
        'ds-1': {
            data_format: csv,
            condition: ['A', 'B'],
            inline: """
                A,B,C
                y,y,1
                y,n,2
                n,y,3
                n,n,4
                """
        },
        'ds-a': {
            data_format: csv,
            inline: """
                A
                y
                n
                """
        },
        'ds-b': {
            data_format: csv,
            inline: """
            B
            y
            n
            """
        },
    },

    crosstabs: {'ds-1'}
}
