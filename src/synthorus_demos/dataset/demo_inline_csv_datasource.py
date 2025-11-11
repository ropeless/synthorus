from synthorus.dataset import Dataset
from synthorus.model.dataset_spec_impl import DatasetSpecCsv, TextInputSpecInline


def main() -> None:
    spec = DatasetSpecCsv(
        input=TextInputSpecInline(
            inline="""
                A,C,X
                y,n,n
                n,n,y
                y,n,y
                y,y,y
                y,y,n
                n,y,y
                y,y,y
                n,n,y
                n,n,n
                y,n,n
            """
        ),
    )

    dataset: Dataset = spec.dataset()

    print()
    print(dataset)
    print()
    print(dataset.crosstab(dataset.rvs))


if __name__ == '__main__':
    main()
