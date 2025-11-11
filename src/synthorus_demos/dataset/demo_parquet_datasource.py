from synthorus.dataset import Dataset
from synthorus.model.dataset_spec_impl import DatasetSpecParquet
from synthorus_demos.demo_files import DATASET_ROOTS


def main() -> None:
    spec = DatasetSpecParquet(location='abc.parquet')

    dataset: Dataset = spec.dataset(roots=DATASET_ROOTS)

    print()
    print(dataset)
    print()
    print(dataset.crosstab(dataset.rvs))


if __name__ == '__main__':
    main()
