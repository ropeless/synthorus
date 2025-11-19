from pathlib import Path

from synthorus.model.model_spec import ModelSpec
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator
from synthorus.spec_file.interpret_spec_file import load_spec_file
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus.workflows.make_simulator_from_files import make_simulator_from_files
from synthorus_demos.demo_files import SPEC_FILES, ROOT_DIR
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem
DEMO_SPEC_FILE_NAME: str = 'spec_5.py'


def main() -> None:
    print(DEMO_NAME, DEMO_SPEC_FILE_NAME)

    model_spec: ModelSpec = load_spec_file(SPEC_FILES / DEMO_SPEC_FILE_NAME, cwd=ROOT_DIR)

    # Create a managed directory for the output model definition files.
    with output_directory(DEMO_NAME, overwrite=True) as model_definition_dir:

        make_model_definition_files(model_spec, model_definition_dir)
        simulator: Simulator = make_simulator_from_files(model_definition_dir)

    simulator.run(DebugRecorder(), iterations=10)


if __name__ == '__main__':
    main()
