import math
from numbers import Integral, Real
from typing import Any, Iterable, List

from ck.pgm import State


def clean_state(value: Any) -> State:
    """
    Convert a rv value to a clean State value.

    Pandas will sometimes change the types of ints to floats and that
    can mess up the ability to convert a random variable state value to
    a state index. This method works to create consistent state object representation.
    """
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if math.isnan(value):
            return None
        int_state = int(float(value))
        if int_state == value:
            return int_state
        else:
            return float(value)
    return str(value)


def clean_states(states: Iterable) -> List[State]:
    """
    Numpy and pandas can mess up data types for states so here we clean and check.
    """
    return list(clean_state(state) for state in states)
