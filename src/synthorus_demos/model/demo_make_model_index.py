from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.make_model_index import make_model_index
from synthorus.model.model_index import ModelIndex
from synthorus.model.model_spec import ModelSpec
from synthorus_demos.model.example_model_spec import make_model_spec_one_entity


def main() -> None:
    model_spec: ModelSpec = make_model_spec_one_entity()
    dataset_cache = DatasetCache(model_spec, cwd=None)
    model_index: ModelIndex = make_model_index(model_spec, dataset_cache)

    print()
    print('JSON:')
    print(model_index.model_dump_json(indent=2))
    print()


if __name__ == '__main__':
    main()
