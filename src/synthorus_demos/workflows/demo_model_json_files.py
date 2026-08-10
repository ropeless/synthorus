from pathlib import Path

from synthorus.model.model_spec import ModelSpec
from synthorus.workflows.file_names import MODEL_SPEC_FILE_NAME, MODEL_INDEX_FILE_NAME
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus_demos.model.example_model_spec import make_model_spec_one_entity
from synthorus_demos.utils.file_helper import cat
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem
LINE: str = '-' * 50


def main() -> None:
    print(DEMO_NAME, 'make_model_spec_one_entity')
    model_spec: ModelSpec = make_model_spec_one_entity()

    # Create a managed directory for the output model definition files.
    with output_directory(DEMO_NAME, overwrite=True) as model_definition_dir:
        make_model_definition_files(model_spec, model_definition_dir)
        show_file(model_definition_dir / MODEL_SPEC_FILE_NAME)
        show_file(model_definition_dir / MODEL_INDEX_FILE_NAME)


def show_file(json_file: Path) -> None:
    print(LINE)
    print(json_file.name)
    print()
    cat(json_file)
    print(LINE)
    print()


if __name__ == '__main__':
    main()
