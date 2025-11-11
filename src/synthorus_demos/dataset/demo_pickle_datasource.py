from synthorus.dataset import Dataset
from synthorus.model.dataset_spec_impl import DatasetSpecPickle
from synthorus_demos.demo_files import DATASET_ROOTS


def main() -> None:
    spec = DatasetSpecPickle(location='abc.pkl')

    dataset: Dataset = spec.dataset(roots=DATASET_ROOTS)

    print()
    print(dataset)
    print()
    print(dataset.crosstab(dataset.rvs))


if __name__ == '__main__':
    main()
