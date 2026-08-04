"""
This is a spec file demonstrating different noise methods.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    # Where to find datasource files
    roots: './datasets',

    # Defaults for all datasources
    data_format: parquet,
    min_cell_size: 1,
    sensitivity: 1,
    epsilon: 1,

    # Default states for all rvs
    states: infer_distinct,

    datasources: {
        'xyz': {},
        'abc': {},
        'acx': {},
    },


    crosstabs: [
        {
            rvs: ['X', 'Y'],
            # default noiser
        },
        {
            rvs: ['Y', 'Z'],
            noise: basic_laplace  # Does not add noise to zero entries of cross-tables
        },
        {
            rvs: ['A', 'B'],
            noise: laplace  # uses default max_add_rows
        },
        {
            rvs: ['B', 'C'],
            noise: naive_laplace  # uses default max_add_rows
        },
        {
            rvs: ['A', 'X'],
            noise: {
                noise: decomposition_laplace,  # Note, needs positive sensitivity
                max_add_rows: 88
            }
        },
        {
            rvs: ['C', 'X'],
            noise: naive_laplace,
            max_add_rows: 99
        },
    ]
}
