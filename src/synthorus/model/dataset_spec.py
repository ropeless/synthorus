from typing import TypeAlias, Annotated, Union

from pydantic import Field

from synthorus.model.dataset_spec_impl import DatasetSpecCsv, DatasetSpecTableBuilder, \
    DatasetSpecPickle, DatasetSpecParquet, DatasetSpecFeather, DatasetSpecFunction, DatasetSpecDBMS

DatasetSpec: TypeAlias = Annotated[
    Union[
        DatasetSpecCsv,
        DatasetSpecTableBuilder,
        DatasetSpecPickle,
        DatasetSpecParquet,
        DatasetSpecFeather,
        DatasetSpecFunction,
        DatasetSpecDBMS,
    ],
    Field(discriminator='type')
]
"""
A DatasetSpec is a serializable description of a dataset.

All DatasetSpec classes inherit pydantic BaseModel and all
have these members:
    type: Literal[...]
    name: str
    def dataset(self, roots: Sequence[Path | Traversable] = ()) -> Dataset
"""
