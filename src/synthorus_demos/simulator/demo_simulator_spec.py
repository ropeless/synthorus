from synthorus.simulator.simulator_spec import SimulatorSpec
from synthorus_demos.simulator import example_simulator_spec


def main() -> None:
    simulator_spec: SimulatorSpec = example_simulator_spec.make_simulator_spec()
    print(simulator_spec.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
