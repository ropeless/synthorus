class Const:
    """
    'Const' for creating unique sentinel objects.
    """
    __slots__ = ('_name',)

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)
