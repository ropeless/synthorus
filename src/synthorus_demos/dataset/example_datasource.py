from synthorus.model.dataset_spec_impl import DatasetSpecCsv, TextInputSpecInline
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.utils.string_extras import strip_lines


def make_datasource_spec_acx(*, sensitivity: int = 1) -> DatasourceSpec:
    return DatasourceSpec(
        sensitivity=sensitivity,
        rvs=['A', 'C', 'X'],
        dataset_spec=make_dataset_spec_acx(),
    )


def make_dataset_spec_acx() -> DatasetSpecCsv:
    return DatasetSpecCsv(
        input=TextInputSpecInline(
            inline=strip_lines("""
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
            """)
        )
    )
