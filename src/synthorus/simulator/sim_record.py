from typing import TypeAlias, Mapping

from ck.pgm import State

SimRecord: TypeAlias = Mapping[str, State]
"""
A sim record represent mapping from field name to field value.
"""
