from dataclasses import dataclass
from typing import Sequence, Callable, Union

import pandas as pd
from ck.pgm import State
from ck.utils.iter_extras import combos as _combos

from synthorus.dataset._dataset_impl.pandas_dataset import PandasDataset
from synthorus.utils.parse_formula import parse_formula


@dataclass
class MathRV:
    """
    A MathRV represents a random variable in the
    formula of a `MathDataset` object.
    """
    name: str
    states: Sequence[State]


class MathDataset(PandasDataset):
    """
    A math dataset is defined by a mathematical formula
    of random variables with finite state.
    """

    def __init__(
            self,
            input_rvs: Sequence[MathRV],
            output_name: str,
            func: Union[str, Callable],
    ):
        """
        The order of the dataset rvs is output name then input rv names.
        This is in keeping with factors where the first random variables
        is the child, and subsequent random variables are the parents.

        Args:
            input_rvs: input random variables
            output_name: name of output random variable
            func: mathematical function to define output.
        """
        in_names = tuple(rv.name for rv in input_rvs)
        input_rv_state = tuple(rv.states for rv in input_rvs)

        if isinstance(func, str):
            self._formula = ' '.join(func.split())
            func = parse_formula(func, in_names)
        elif isinstance(func, Callable):
            self._formula = repr(func)
        else:
            raise TypeError('`func` is not callable or expression string')

        # Run the function over all possible input value combinations
        data = []
        for input_values in _combos(input_rv_state):
            output_value = func(*input_values)
            data.append(input_values + (output_value,))
        # Convert to series
        data = {
            name: series
            for name, series in zip(
                in_names + (output_name,),
                zip(*data)
            )
        }

        # Convert to dataframe
        dataframe = pd.DataFrame(data)

        super().__init__(
            dataframe=dataframe,
            weights=None
        )
