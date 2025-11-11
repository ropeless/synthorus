from pathlib import Path
from typing import Set

from pydantic import ValidationError

from synthorus.error import SynthorusError
from synthorus.model.model_spec import ModelSpec
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus_demos.demo_files import SPEC_FILES, ROOT_DIR

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
        if spec_file.is_file() and spec_file.name not in EXCLUSIONS:
            print(LINE)
            print(f'{script_name}: {spec_file.name}')

            if spec_file.name.lower().startswith('bad_'):
                got_err = False
                try:
                    load_spec_file(spec_file, cwd=ROOT_DIR)
                except (SynthorusError, ValidationError) as err:
                    print('Error raised (as expected):')
                    print(f'{err.__class__.__name__}: {err}')
                    got_err = True
                if not got_err:
                    raise RuntimeError('expected SpecError to be raised')
            else:
                model_spec: ModelSpec = load_spec_file(spec_file, cwd=ROOT_DIR)
                print(model_spec.model_dump_json(indent=2))
            print()
    print(LINE)
    print('Done.')


if __name__ == '__main__':
    main()
