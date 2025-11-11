from typing import Sequence, Callable

from synthorus.error import SynthorusError

# An expression can only contain these characters.
# For example, characters !@#$^&|\:;? and non-printable characters
# are not permitted.
_VALID_CHARS = set(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    '_.,+-/%*(){}[]<>='
    ' "\''
)

# An expression is not permitted to contain these character sequences.
_INVALID_TOKENS = [
    'exit', 'eval', 'exec', 'import', 'lambda',
]


def parse_formula(expression: str, in_names: Sequence[str]) -> Callable:
    """
    Return a Python callable that executes the given expression,
    taking parameters `in_names`.

    E.g.,

        f = parse_formula('(x + y) ** 2', ['x', 'y'])
        z = f(2, 3)

    The value of z = (2 + 3) ** 2 = 25.
    """

    # Simplify formula white space.
    expression = ' '.join(expression.split())

    # Check for illegal expressions.
    if not set(expression).issubset(_VALID_CHARS):
        raise SynthorusError(f'formula contains invalid characters: {expression!r}')
    for token in _INVALID_TOKENS:
        if token in expression:
            raise SynthorusError(f'formula contains invalid character sequence: {token!r}')

    # Compile a lambda expression.
    in_args = ', '.join(in_names)
    code = f'lambda {in_args}: {expression}'
    try:
        return eval(code)
    except Exception:
        raise SynthorusError(f'could not parse function: {code}')
