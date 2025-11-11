from synthorus.dataset import Dataset
from synthorus.model.dataset_spec import DatasetSpecFunction


def main() -> None:
    spec = DatasetSpecFunction(
        rvs={
            'x': 3,  # states = [0, 1, 2]
            'y': [2, 3, 5],  # states as given
        },
        output_rv='z',
        function='x * y',
    )

    dataset: Dataset = spec.dataset()

    print()
    print(dataset)
    print()
    print(dataset.crosstab(dataset.rvs))


if __name__ == '__main__':
    main()
