from pathlib import Path
from typing import Set

from synthorus.model.model_spec import ModelSpec
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus_demos.demo_files import SPEC_FILES, ROOT_DIR
from synthorus_demos.utils.output_directory import output_directory

LINE = '-' * 80

EXCLUSIONS: Set[str] = {
    'spec_odbc.py',  # requires a database
    '__init__.py',  # not a spec file
}


def main():
    """
    Confirm all specs files load without error.

    Load and dump all spec files in SPEC_FILES demo directory,
    skipping nominated exclusions.
    """
    script_name: str = Path(__file__).stem

    for spec_file in SPEC_FILES.iterdir():
        if spec_file.is_file() and spec_file.name not in EXCLUSIONS and not spec_file.name.lower().startswith('bad_'):
            print(LINE)
            print(f'{script_name}: {spec_file.name}')
            model_spec: ModelSpec = load_spec_file(spec_file, cwd=ROOT_DIR)

            with output_directory(f'demo__{spec_file.name}') as model_definition_dir:
                make_model_definition_files(model_spec, model_definition_dir, overwrite=True, cwd=ROOT_DIR)

            print(model_spec.model_dump_json(indent=2))

            print()
    print(LINE)
    print('Done.')


if __name__ == '__main__':
    main()
