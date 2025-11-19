"""
This is an example simple Synthorus spec file.
"""
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,  # no data is sensitive data
    min_cell_size: 0,  # no data will be redacted

    states: infer_distinct,  # default for all random variables

    datasources: {
        'xyz': {
            data_format: csv,
            inline: """
                X,Y,Z
                y,y,y
                y,y,n
                y,n,y
                y,n,n
                n,y,y
                n,y,n
                n,n,y
                n,n,n
                """
        }
    }
}
