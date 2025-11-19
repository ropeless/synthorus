"""
This is a spec file designed to stress 'defined' columns in datasources.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    roots: './datasets',

    sensitivity: 0,
    min_cell_size: 0,

    datasources: {
        'counter_10': {
            data_format: csv,
            inline:
            """counter
            1
            2
            3
            4
            5
            6
            7
            8
            9
            10
            """,
            define: {
                'square': {
                    function: 'counter ** 2',
                    input: 'counter',
                },
                'groups': {
                    grouping: group_cut,
                    input: 'counter',
                    size: 5
                },
            }
        },
    },

    states: infer_max,  # default for all rvs
    rvs: {
        'counter': {},
        'square': {},
        'groups': {states: infer_distinct},
    },

    crosstabs: [
        'counter_10',
    ]
}
