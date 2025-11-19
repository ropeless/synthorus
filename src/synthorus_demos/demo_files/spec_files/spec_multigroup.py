"""
This is a spec file demonstrating automatic grouping of multiple columns
to create a latent variable.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,             # all datasources
    states: infer_distinct,     # all rvs

    datasources: {
        'xyz': {
            data_format: csv,
            weight: -1,
            inline: """
            X,Y,Z,weight
            y,1,a,10
            y,1,b,6
            y,2,a,1
            y,2,b,14
            n,1,a,2
            n,1,b,5
            n,2,a,1
            n,2,b,3
            """,
            define: {
                'group': {
                    grouping: group_normalise,
                    input: ['X', 'Y', 'Z'],
                    size: 5
                },
            }
        }
    },

    crosstabs: [
        ['X', 'group'],
        ['Y', 'group'],
        ['Z', 'group'],
    ]
}
