"""
This is an example simple Synthorus spec file.
"""
from synthorus.spec_file.keys import *

spec = {
    states: infer_distinct,  # all random variable states are inferred from the datasource.

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
