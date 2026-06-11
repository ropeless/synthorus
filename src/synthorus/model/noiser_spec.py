from typing import TypeAlias, Annotated, Union, Literal

from pydantic import Field, BaseModel

from synthorus.noise.noiser import Noiser, LaplaceNoise, BasicLaplaceNoise, NaiveLaplaceNoise, DecompositionLaplaceNoise


class NoiserSpecLaplace(BaseModel):
    """
    A NoiserSpec for LaplaceNoise.
    This is the standard noiser.
    """

    type: Literal['laplace'] = 'laplace'

    max_add_rows: int

    def noiser(self) -> Noiser:
        return LaplaceNoise(max_add_rows=self.max_add_rows)


class NoiserSpecBasicLaplace(BaseModel):
    """
    A NoiserSpec for BasicLaplaceNoise.
    THIS DOES NOT SATISFY DIFFERENTIAL PRIVACY REQUIREMENTS.
    This noiser will never add new rows.
    """

    type: Literal['basic_laplace'] = 'basic_laplace'

    @staticmethod
    def noiser() -> Noiser:
        return BasicLaplaceNoise()


class NoiserSpecNaiveLaplace(BaseModel):
    """
    A NoiserSpec for NaiveLaplaceNoise.
    This is provided for internal benchmarking only.
    """

    type: Literal['naive_laplace'] = 'naive_laplace'

    max_add_rows: int

    def noiser(self) -> Noiser:
        return NaiveLaplaceNoise(max_add_rows=self.max_add_rows)


class NoiserSpecDecompositionLaplace(BaseModel):
    """
    A NoiserSpec for NaiveLaplaceNoise.
    This is provided for internal benchmarking only.
    """

    type: Literal['decomposition_laplace'] = 'decomposition_laplace'

    max_add_rows: int

    def noiser(self) -> Noiser:
        return DecompositionLaplaceNoise(max_add_rows=self.max_add_rows)


NoiserSpec: TypeAlias = Annotated[
    Union[
        NoiserSpecLaplace,
        NoiserSpecBasicLaplace,
        NoiserSpecNaiveLaplace,
        NoiserSpecDecompositionLaplace,
    ],
    Field(discriminator='type')
]
"""
A NoiserSpec is a serializable description of a dataset.

All NoiserSpec classes inherit pydantic BaseModel and all
have these members:
    type: Literal[...]
    def noiser(self) -> Noiser
"""
