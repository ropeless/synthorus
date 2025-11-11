"""
This module declares constants representing names for files
and directories in the file structure of a synthetic data model.

Here is a diagram of the file structure, from the root of
the structure (for some model spec).

{root} / MODEL_SPEC_NAME
       / MODEL_INDEX_NAME
       / SIMULATOR_SPEC_NAME
       / CLEAN_CROSS_TABLES / {cross_table}.pk
       / NOISY_CROSS_TABLES / {cross_table}.pk
       / ENTITY_MODELS / {entity}.py
       / REPORTS / PRIVACY_REPORT_FILE_NAME
       / REPORTS / MODEL_SPEC_REPORT_FILE_NAME
       / REPORTS / UTILITY_REPORT_FILE_NAME
       / REPORTS / UTILITY_RESULTS_FILE_NAME
       / REPORTS / CROSSTAB_REPORT_FILE_NAME

"""

# Subdirectories
CLEAN_CROSS_TABLES = 'clean_cross_tables'
NOISY_CROSS_TABLES = 'noisy_cross_tables'
ENTITY_MODELS = 'pgms'
REPORTS = 'reports'

# Model analysis outputs
MODEL_SPEC_NAME = 'model_spec.json'
MODEL_INDEX_NAME = 'model_index.json'
SIMULATOR_SPEC_NAME = 'simulator_spec.json'

# Report files
PRIVACY_REPORT_FILE_NAME = 'report_on_privacy.txt'
MODEL_SPEC_REPORT_FILE_NAME = 'report_on_model_spec.txt'
UTILITY_REPORT_FILE_NAME = 'report_on_utility.txt'
UTILITY_RESULTS_FILE_NAME = 'utility_results.csv'
CROSSTAB_REPORT_FILE_NAME = 'crosstabs.csv'
