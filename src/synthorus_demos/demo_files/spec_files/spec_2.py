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
            data_format: csv,
            weight: -1,
            condition: 'X',
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
        'abc': {  # Contains columns A, B and C
            location: 'abc.csv',
            sensitivity: 0,
            condition: ['A', 'C'],
        },
        'acx': {  # Contains columns A, C and X
            location: 'acx.tsv',
            sensitivity: 2,
        },
        'double_q': {
            function: 'q * 2',       # This column will be called double_q
            input: {'q': range(10)}  # This column will be called q
        },
        'square_q': {
            data_format: csv,
            inline: """
                q,square_q
                0,0
                1,1
                2,4
                3,9
                4,16
                5,25
                6,36
                7,49
                8,64
                9,81
                """
        },
        'double_r': {
            data_format: csv,
            inline: """
            r,double_r
            0,0
            1,2
            2,4
            3,6
            4,8
            5,10
            6,12
            7,14
            8,16
            9,18
            """
        },
        'square_r': {
            data_format: csv,
            inline: """
            r,square_r
            0,0
            1,1
            2,4
            3,9
            4,16
            5,25
            6,36
            7,49
            8,64
            9,81
            """
        },
    },

    states: infer_range,  # default
    rvs: {
        'X': {states: infer_distinct},
        'Y': {states: infer_distinct},
        'Z': {states: infer_distinct},
        'A': {states: infer_distinct},
        'B': {states: infer_distinct},
        'C': {states: infer_distinct},
        'q': {states: 20},                  # = range(20) = 0, 1, 2, ..., 18, 19
        'double_q': {states: {stop: 40}},   # = range(0, 40) = 0, 1, 2, ..., 38, 39
        'square_q': {states: infer_range},  # min, min+1, min+2, ..., max-2, max-1, max
        'r': {datasource: 'double_r'},      # use default, i.e., infer_range, using explicit datasource
        'double_r': {},                     # use default, i.e., infer_range
        'square_r': {},                     # use default, i.e., infer_range
    },

    crosstabs: [
        'xyz',    # use the rvs as defined in data ource 'xyx'
        'abc',    # use the rvs as defined in datasource 'abc'
        {rvs: ['A', 'X']},
        {rvs: ['q', 'double_q']},
        {rvs: ['q', 'square_q']},
        # Implicitly defined a cross-table r (because it's a random variable and is not in any other crostab
        # Implicitly defined a cross-table double_r (same reason)
        # Implicitly defined a cross-table square_r (same reason)
    ]
}
