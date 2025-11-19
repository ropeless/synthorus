from pathlib import Path
from typing import Set

from synthorus.model.model_spec import ModelSpec
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus.workflows.make_simulator_from_files import make_simulator_from_files
from synthorus_demos.demo_files import SPEC_FILES, ROOT_DIR, Traversable
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem

LINE = '-' * 80

EXCLUSIONS: Set[str] = {
    'spec_odbc.py',  # requires a database
    '__init__.py',  # not a spec file
}


def main():
    """
    Show all specs files make a working simulator.

    Load and dump all spec files in SPEC_FILES demo directory,
    skipping nominated exclusions.
    """

    for spec_file in SPEC_FILES.iterdir():
        if (
                spec_file.is_file()
                and spec_file.name not in EXCLUSIONS
                and not spec_file.name.lower().startswith('bad_')
        ):
            demo_one_spec_file(spec_file)
    print('Done.')


def demo_one_spec_file(spec_file: Path | Traversable) -> None:
    print(LINE)
    print(f'{DEMO_NAME}: {spec_file.name}')
    model_spec: ModelSpec = load_spec_file(spec_file, cwd=ROOT_DIR)
    print(model_spec.model_dump_json(indent=2))

    with output_directory(f'demo__{spec_file.name}', overwrite=True) as model_definition_dir:
        make_model_definition_files(model_spec, model_definition_dir, cwd=ROOT_DIR)
        simulator: Simulator = make_simulator_from_files(model_definition_dir)

    simulator.run(DebugRecorder(), iterations=10)

    print()
    print(LINE)


if __name__ == '__main__':
    main()
