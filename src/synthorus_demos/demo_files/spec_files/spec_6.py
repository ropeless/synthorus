"""
This is a spec file designed to show epsilon zeroing in reports.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,          # for all datasources
    epsilon: 1.2,            # sound end up as zero due to zero sensitivity
    states: infer_distinct,  # for all rvs

    datasources: {
        'xyz': {
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
    },

    crosstabs: {
        'xyz',
    },
}
