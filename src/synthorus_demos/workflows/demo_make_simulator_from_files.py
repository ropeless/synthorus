from pathlib import Path

from synthorus.model.model_spec import ModelSpec
from synthorus.simulator.sim_recorder import DebugRecorder
from synthorus.simulator.simulator import Simulator
from synthorus.workflows.make_model_definition_files import make_model_definition_files
from synthorus.workflows.make_simulator_from_files import make_simulator_from_files
from synthorus_demos.model.example_model_spec import make_model_spec_two_entities
from synthorus_demos.utils.output_directory import output_directory

DEMO_NAME: str = Path(__file__).stem


def main() -> None:
    print(DEMO_NAME, 'make_model_spec_two_entities')
    model_spec: ModelSpec = make_model_spec_two_entities()
    model_spec.pgm_crosstabs = 'clean'

    # Create a managed directory for the output model definition files.
    with output_directory(DEMO_NAME, overwrite=True) as model_definition_dir:
        make_model_definition_files(model_spec, model_definition_dir)
        simulator: Simulator = make_simulator_from_files(model_definition_dir)

    # ===================================
    #  Run simulation
    # ===================================
    print()
    simulator.run(DebugRecorder(), iterations=8)


if __name__ == '__main__':
    main()
