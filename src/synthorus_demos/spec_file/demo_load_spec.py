from synthorus.model.model_spec import ModelSpec
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus_demos.demo_files import SPEC_FILES, ROOT_DIR

DEMO_SPEC_FILE_NAME: str = 'spec_7.py'


def main() -> None:
    """
    Load and dump one example spec file.
    """
    spec_path = SPEC_FILES / DEMO_SPEC_FILE_NAME

    model_spec: ModelSpec = load_spec_file(spec_path, cwd=ROOT_DIR)

    print()
    print('JSON:')
    print(model_spec.model_dump_json(indent=2))
    print()

    print('Done.')


if __name__ == '__main__':
    main()
