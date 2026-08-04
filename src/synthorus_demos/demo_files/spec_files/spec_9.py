"""
This is a spec file designed to show an entity with multiple foreign keys.
It also shows a spec file with no data sources or random variables.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {

    cardinality: 2,  # for all entities

    # Simulation entities
    entities: {
        'A': {},
        'B': {},
        'C': {foreign_keys: ['A', 'B']},
    },
}
