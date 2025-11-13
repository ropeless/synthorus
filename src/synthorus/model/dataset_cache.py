from importlib.abc import Traversable
from pathlib import Path
from typing import Mapping, Optional, Dict, Sequence, Iterable, List, Iterator

from synthorus.dataset import Dataset
from synthorus.error import SynthorusError
from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.model_spec import ModelSpec


class DatasetCache(Mapping[str, Dataset]):
    """
    Keeps track of all datasets loaded for a model spec.
    """

    def __init__(self, model_spec: ModelSpec, cwd: Optional[Path | Traversable]):
        """
        Args:
            model_spec: the model datasources and roots.
            cwd: Optional working directory to use for resolving relative roots.
        """
        self._model_spec = model_spec
        self._roots: Sequence[Path | Traversable] = tuple(interpret_roots(model_spec.roots, cwd))
        self._datasets: Dict[str, Dataset] = {}

    @property
    def roots(self) -> Sequence[Path]:
        return self._roots

    def loaded_keys(self) -> Iterable[str]:
        return self._datasets.keys()

    def keys(self) -> Iterable[str]:
        return self._model_spec.datasources.keys()

    def __getitem__(self, key: str, /) -> Dataset:
        got: Optional[Dataset] = self._datasets.get(key)
        if got is not None:
            return got
        datasource_spec: DatasourceSpec = self._model_spec.datasources[key]
        dataset: Dataset = datasource_spec.dataset(self._roots)
        self._datasets[key] = dataset
        return dataset

    def __len__(self):
        return len(self._model_spec.datasources)

    def __iter__(self) -> Iterator[str]:
        return iter(self._model_spec.datasources)


def interpret_roots(roots: List[str], cwd: Optional[Path | Traversable]) -> List[Path]:
    """
    Return a list of Path objects defining the root directories
    to search for datasource files.

    Any relative roots in `roots` will be resolved relative to `cwd`.
    If the model_spec has no 'roots' entry, then `cwd` is returned (if provided).

    Args:
        roots: List of root strings to interpret as paths.
        cwd: The working directory to use for resolving relative roots.

    Returns:
        A list of Path objects. If `cwd` is an absolute path, then
        all returned paths will be absolute paths.
    """
    if len(roots) == 0:
        if cwd is None:
            return []
        else:
            return [cwd]

    roots_paths = [Path(root) for root in roots]

    # Ensure relative roots are made relative to cwd (if provided)
    if cwd is not None:
        roots_paths = [
            root if root.is_absolute() else cwd / root
            for root in roots_paths
        ]

    return roots_paths
