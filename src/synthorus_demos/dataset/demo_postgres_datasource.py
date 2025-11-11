from synthorus.dataset import Dataset
from synthorus.model.dataset_spec_impl import DatasetSpecDBMS


def main() -> None:
    dataset_spec = DatasetSpecDBMS(
        type='postgres',
        connection={
            'user': 'demo_user',
            'password': 'demo_password',
            'host': 'samples.mindsdb.com',
            'dbname': 'demo',
        },
        schema_name='demo',
        table_name='home_rentals',
    )

    # get the dataset
    dataset: Dataset = dataset_spec.dataset()

    # show the random variables
    print(dataset.rvs)
    print()

    # show an example cross-table
    print(dataset.crosstab(['neighborhood']))


if __name__ == '__main__':
    main()
