from synthorus.model.model_spec import ModelSpec
from synthorus_demos.model import example_model_spec


def main() -> None:
    model_spec: ModelSpec = example_model_spec.make_model_spec_one_entity()

    print()
    print('JSON:')
    print(model_spec.model_dump_json(indent=2))
    print()


if __name__ == '__main__':
    main()
