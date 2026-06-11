from synthorus.dataset import Dataset, PandasDataset
from synthorus.model.dataset_spec_impl import DatasetSpecPickle
from synthorus_demos.demo_files import DATASET_ROOTS


def main() -> None:
    spec = DatasetSpecPickle(location='abc.pkl')

    dataset: Dataset = spec.dataset(roots=DATASET_ROOTS)

    if isinstance(dataset, PandasDataset):
        print()
        print(dataset.dataframe)

    print()
    print(dataset.crosstab(dataset.rvs))


if __name__ == '__main__':
    main()
