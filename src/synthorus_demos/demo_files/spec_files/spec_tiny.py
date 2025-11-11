"""
This is an example minimal Synthorus spec file.
"""
from synthorus.spec_file.keys import *

spec = {
    # default for all random variables
    states: infer_distinct,

    datasources: {
        'xyz': {
            data_format: csv,
            sensitivity: 0,
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
