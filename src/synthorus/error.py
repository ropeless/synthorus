class SynthorusError(RuntimeError):
    """
    An exception raised by Synthorus.
    """


class SpecFileError(SynthorusError):
    """
    An exception raised while loading a spec file.
    """

    def __init__(self, error: str, section=None, details=None):
        if section is not None:
            error += f' in section {section}'
        if details is not None:
            error += f': {details}'
        super().__init__(error)
