from synthorus.dataset import Dataset
from synthorus.model.dataset_spec_impl import DatasetSpecCsv, TextInputSpecLocation
from synthorus_demos.demo_files import DATASET_ROOTS


def main() -> None:
    spec = DatasetSpecCsv(input=TextInputSpecLocation(location='acx.csv'))

    dataset: Dataset = spec.dataset(roots=DATASET_ROOTS)

    print()
    print(dataset)
    print()
    print(dataset.crosstab(dataset.rvs))


if __name__ == '__main__':
    main()
