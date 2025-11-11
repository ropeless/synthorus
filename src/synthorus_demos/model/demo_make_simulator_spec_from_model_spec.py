from synthorus.model.model_spec import ModelSpec
from synthorus.simulator.make_simulator_spec_from_model_spec import make_simulator_spec_from_model_spec
from synthorus_demos.model.example_model_spec import make_model_spec_one_entity


def main() -> None:
    model_spec: ModelSpec = make_model_spec_one_entity()
    simulator_spec = make_simulator_spec_from_model_spec(model_spec)

    print()
    print('JSON:')
    print(simulator_spec.model_dump_json(indent=2))
    print()


if __name__ == '__main__':
    main()
