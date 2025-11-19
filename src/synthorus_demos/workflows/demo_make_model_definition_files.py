from pathlib import Path

from synthorus.model.model_spec import ModelSpec
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus_demos.model.example_model_spec import make_model_spec_one_entity
from synthorus_demos.utils.file_helper import print_file_tree
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem


def main() -> None:
    print(DEMO_NAME, 'make_model_spec_one_entity')

    model_spec: ModelSpec = make_model_spec_one_entity()

    # Create a managed directory for the output model definition files.
    with output_directory(DEMO_NAME, overwrite=True) as model_definition_dir:
        make_model_definition_files(model_spec, model_definition_dir)

        # Show what files got created
        print()
        print('-------------------------------------------')
        print_file_tree(model_definition_dir)
        print('-------------------------------------------')


if __name__ == '__main__':
    main()
