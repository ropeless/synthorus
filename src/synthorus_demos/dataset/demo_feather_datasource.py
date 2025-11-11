from synthorus.dataset import Dataset
from synthorus.model.dataset_spec import DatasetSpecFeather
from synthorus_demos.demo_files import DATASET_ROOTS


def main() -> None:
    spec = DatasetSpecFeather(location='abc.feather')

    dataset: Dataset = spec.dataset(roots=DATASET_ROOTS)

    print()
    print(dataset)
    print()
    print(dataset.crosstab(dataset.rvs))


if __name__ == '__main__':
    main()
