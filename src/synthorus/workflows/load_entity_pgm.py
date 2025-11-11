from pathlib import Path

from ck.pgm import PGM

from synthorus.utils import py_loader


def load_entity_pgm(pgms_path: Path, entity_name: str) -> PGM:
    """
    Load a specific entity model previously saved to `pgms_path`.
    """
    model_path = pgms_path / f'{entity_name}.py'
    pgm: PGM = py_loader.load_object(model_path, object_type=PGM)
    return pgm
