from synthorus.dataset import Dataset
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus_demos.dataset.example_datasource import make_datasource_spec_acx


def main() -> None:
    example_datasource_spec: DatasourceSpec = make_datasource_spec_acx()
    print(example_datasource_spec.model_dump_json(indent=2))
    print()

    example_dataset: Dataset = example_datasource_spec.dataset()

    print(example_dataset.rvs)
    print(example_dataset.number_of_records())
    print('A', example_dataset.value_set('A'))
    print('C', example_dataset.value_set('C'))
    print('X', example_dataset.value_set('X'))
    print()
    print(example_dataset.crosstab(['A', 'C', 'X']))
    print()
    print(example_dataset.crosstab(['A', 'C']))
    print()
    print(example_dataset.crosstab(['C']))
    print()
    print(example_dataset.crosstab([]))
    print()


if __name__ == '__main__':
    main()
