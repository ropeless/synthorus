from pathlib import Path

from synthorus.model.model_spec import ModelSpec
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus.workflows.file_names import REPORTS, PRIVACY_REPORT_FILE_NAME
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus.workflows.report_privacy import make_privacy_report
from synthorus_demos.demo_files import SPEC_FILES
from synthorus_demos.utils.file_helper import print_file_tree, cat
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem
DEMO_SPEC_FILE_NAME: str = 'spec_simple_pjm.py'


def main() -> None:
    print(DEMO_NAME, DEMO_SPEC_FILE_NAME)

    # Create a managed directory for the output model definition files.
    with output_directory(DEMO_NAME, overwrite=True) as model_definition_dir:
        model_spec: ModelSpec = load_spec_file(SPEC_FILES / DEMO_SPEC_FILE_NAME)

        print('-------------------------------------------')
        print(model_spec.model_dump_json(indent=2))
        print('-------------------------------------------')

        make_model_definition_files(
            model_spec,
            model_definition_dir,
            make_privacy_report=False,
            make_crosstab_report=False,
            make_model_spec_report=False,
        )

        # Show what files got created
        print()
        print('-------------------------------------------')
        print_file_tree(model_definition_dir)
        print('-------------------------------------------')
        print()

        print('Making privacy report...')
        make_privacy_report(model_definition_dir)

        # Show what files got created
        print('-------------------------------------------')
        print_file_tree(model_definition_dir)
        print('-------------------------------------------')
        print()
        cat(model_definition_dir / REPORTS / PRIVACY_REPORT_FILE_NAME)


if __name__ == '__main__':
    main()
