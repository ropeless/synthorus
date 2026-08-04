from pathlib import Path

from synthorus.model.model_spec import ModelSpec
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus.workflows.file_names import REPORTS, PRIVACY_REPORT_FILE_NAME
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus.workflows.report_privacy import make_privacy_report
from synthorus_demos.demo_files import SPEC_FILES, ROOT_DIR
from synthorus_demos.utils.file_helper import cat
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem
DEMO_SPEC_FILE_NAME: str = 'spec_2.py'


def main() -> None:
    print(DEMO_NAME, DEMO_SPEC_FILE_NAME)

    # Create a managed directory for the output model definition files.
    with output_directory(DEMO_NAME, overwrite=True) as model_definition_dir:
        model_spec: ModelSpec = load_spec_file(SPEC_FILES / DEMO_SPEC_FILE_NAME, cwd=ROOT_DIR)

        make_model_definition_files(
            model_spec,
            model_definition_dir,
            make_privacy_report=False,
            make_crosstab_report=False,
            make_model_spec_report=False,
            cwd=ROOT_DIR,
        )

        print('Making privacy report...')
        make_privacy_report(model_definition_dir, cwd=ROOT_DIR)

        # Show the report
        print('-------------------------------------------')
        cat(model_definition_dir / REPORTS / PRIVACY_REPORT_FILE_NAME)


if __name__ == '__main__':
    main()
