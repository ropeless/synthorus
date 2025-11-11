from typing import List, Set


def validate_inputs(inputs: List[str]) -> List[str]:
    """
    Given a list of input variable names, validate them to ensure
    that each one is unique and is a Python identifier.

    Args:
        inputs: a list of input variable names.

    Returns:
        a list of valid input variable names.

    Raises:
        ValueError: if any of the input variable names
        is duplicated or is not a Python identifier.
    """
    seen: Set[str] = set()
    for name in inputs:
        if not name.isidentifier():
            raise ValueError(f'invalid input identifier: {name!r}')
        if name in seen:
            raise ValueError(f'duplicated input field: {name!r}')
        seen.add(name)
    return inputs
