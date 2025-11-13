from typing import Sequence, Callable

from synthorus.error import SynthorusError

# An expression is not permitted to contain these strings.
INVALID_STRINGS = (
    'exit', 'eval', 'exec', 'import', 'lambda', 'print', 'input', 'open',
)


def parse_formula(expression: str, arguments: Sequence[str]) -> Callable:
    """
    Return a Python callable that executes the given expression,
    taking parameters `arguments`.

    E.g.,

        f = parse_formula('(x + y) ** 2', ['x', 'y'])
        z = f(2, 3)

    The value of z = (2 + 3) ** 2 = 25.

    The expression is restricted to what can be in the body of a Python lambda function,
    and the `arguments` must be Python identifiers. Additionally, the expression is
    not allowed to contain the strings in `INVALID_STRINGS`.

    Args:
        expression: The expression forming the body on a lambda function.
        arguments: The identifiers forming arguments for the lambda function.

    Returns:
        a Python callable that executes the given expression, with the given arguments.

    Raises:
        SynthorusError: if the lambda function cannot be properly formed.
    """

    # Simplify formula white space.
    expression = ' '.join(expression.split())

    # Check for invalid strings.
    for token in INVALID_STRINGS:
        if token in expression:
            raise SynthorusError(f'expression contains invalid character sequence: {token!r}')

    # Check input identifiers.
    for arg in arguments:
        if not arg.isidentifier():
            raise SynthorusError(f'invalid argument identifier: {arg!r}')

    # Form a lambda expression.
    in_args = ', '.join(arguments)
    code = f'lambda {in_args}: {expression}'
    try:
        return eval(code)
    except Exception:
        raise SynthorusError(f'could not parse function: {code!r}')
