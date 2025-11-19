from pathlib import Path

import pandas as pd

from synthorus.model.model_spec import ModelSpec
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus.workflows.file_names import REPORTS, UTILITY_REPORT_FILE_NAME, UTILITY_RESULTS_FILE_NAME
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus.workflows.report_utility import make_utility_report
from synthorus_demos.demo_files import SPEC_FILES, ROOT_DIR
from synthorus_demos.utils.file_helper import print_file_tree, cat
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem
DEMO_SPEC_FILE_NAME: str = 'spec_2.py'


def main() -> None:
    print(DEMO_NAME, DEMO_SPEC_FILE_NAME)

    # Create a managed directory for the output model definition files.
    with output_directory(DEMO_NAME, overwrite=True) as model_definition_dir:
        model_spec: ModelSpec = load_spec_file(SPEC_FILES / DEMO_SPEC_FILE_NAME, cwd=ROOT_DIR)

        print('--------------------------------------------------------------------------------')
        print(model_spec.model_dump_json(indent=2))
        print('--------------------------------------------------------------------------------')

        make_model_definition_files(
            model_spec,
            model_definition_dir,
            make_privacy_report=False,
            make_crosstab_report=False,
            make_model_spec_report=False,
            cwd=ROOT_DIR,
        )

        # Show what files got created
        print()
        print('--------------------------------------------------------------------------------')
        print_file_tree(model_definition_dir)
        print('--------------------------------------------------------------------------------')
        print()

        print('Making utility report...')
        make_utility_report(model_definition_dir)

        # Show what files got created
        print('--------------------------------------------------------------------------------')
        print_file_tree(model_definition_dir)
        print('--------------------------------------------------------------------------------')
        print()
        cat(model_definition_dir / REPORTS / UTILITY_REPORT_FILE_NAME)
        print()
        df: pd.DataFrame = pd.read_csv(model_definition_dir / REPORTS / UTILITY_RESULTS_FILE_NAME)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 999)
        print(df)


if __name__ == '__main__':
    main()
