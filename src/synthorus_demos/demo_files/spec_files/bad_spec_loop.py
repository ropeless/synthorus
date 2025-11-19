"""
This is a bad model that includes a datasource loop.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,
    states: infer_distinct,

    datasources: {
        'ds-1': {
            data_format: csv,
            condition: 'A',
            inline: """
                A,B
                y,y
                y,n
                n,y
                n,n
                """
        },
        'ds-2': {
            data_format: csv,
            condition: 'B',
            inline: """
            A,B
            y,y
            y,n
            n,y
            n,n
            """
        },
    },
}
