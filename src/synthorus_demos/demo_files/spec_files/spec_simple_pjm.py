"""
This is a spec file designed to show the use of entities and parameters
for modelling patient journeys.
"""
__author__ = 'Barry Drake'

# Import all reserved keys.
from synthorus.spec_file.keys import *

spec = {
    sensitivity: 0,          # for all datasources
    data_format: csv,        # for all datasources
    weight: -1,              # for all datasources
    states: infer_distinct,  # for all rvs
    min_cell_size: 0,        # for all cross-tables

    datasources: {
        'patient_age__event_type': {
            inline: """
            patient_age, event_type, weight
            young,       GP,         30
            middle_aged, ED,         24
            middle_aged, GP,         24
            old,         ED,         9
            old,         AP,         9
            old,         GP,         9
            old,         DEATH,      3
            """
        },
        'event_duration': {
            inline: """
            event_duration, weight
            1,              2.56
            2,              1.28
            3,              0.64
            4,              0.32
            5,              0.16
            6,              0.08
            7,              0.04
            8,              0.02
            9,              0.01
            """
        },
        'event_duration_since_last': {
            inline: """
            event_duration_since_last, weight
            1,                         0.01
            2,                         0.02
            3,                         0.04
            4,                         0.08
            5,                         0.16
            6,                         0.32
            7,                         0.64
            8,                         1.28
            9,                         2.56
            """
        },
    },


    # rvs: all rvs from all datasources
    # crosstabs: each datasource will be a cross-table

    # Simulation parameters
    parameters: {
        'number_of_patients': 10,
        'time_limit': 100,
    },

    # Simulation entities
    entities: {
        'patient': {
            rvs: ['patient_age'],
            cardinality: 'number_of_patients'
        },
        'event': {
            rvs: ['event_type', 'event_duration', 'event_duration_since_last'],
            fields: {
                'time': {
                    value: 0,
                    sum: ['time', 'event_duration', 'event_duration_since_last']
                },
            },
            parent: 'patient',
            cardinality: [
                {field: 'time', limit: 'time_limit'},
                {field: 'event_type', state: 'DEATH'},
            ]
        },
    },
}
