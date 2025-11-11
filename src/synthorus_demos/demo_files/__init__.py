from importlib.abc import Traversable
from typing import List
import importlib.resources as resources

# Where to find data files
ROOT_DIR: Traversable = resources.files('synthorus_demos.demo_files')
SPEC_FILES: Traversable = ROOT_DIR / 'spec_files'
DATASET_FILES: Traversable = ROOT_DIR / 'datasets'
DATASET_ROOTS: List[Traversable] = [DATASET_FILES, DATASET_FILES / 'table_builder']
