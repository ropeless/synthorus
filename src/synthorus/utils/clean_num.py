import math


def clean_num(value) -> str:
    """
    Render a numeric value as a nice string.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    i_value = int(value)
    if i_value == value:
        # Number is an integer
        return f'{i_value:,}'
    else:
        as_fixed = f'{value:,.6f}'.rstrip('0')
        if as_fixed[-1] == '.':
            # Number too small
            return f'{value:.5}'
        else:
            return as_fixed
