import os
from pathlib import Path

from synthorus.dataset import Dataset
from synthorus.error import SynthorusError
from synthorus.model.dataset_cache import DatasetCache
from synthorus.model.model_spec import ModelSpec
from tests.helpers.make_model_spec import make_model_spec, make_dataset_csv_inline, make_dataset_csv_file, \
    make_dataset_feather_file
from tests.helpers.tmp_dir import tmp_dir
from tests.helpers.unittest_fixture import Fixture, test_main


class DatasetCacheTest(Fixture):

    def test_empty(self) -> None:
        model_spec: ModelSpec = make_model_spec({})

        dataset_cache = DatasetCache(model_spec, cwd=None)

        self.assertEqual(len(dataset_cache), 0)
        self.assertEmpty(set(dataset_cache.keys()))
        self.assertEmpty(set(iter(dataset_cache)))
        self.assertEmpty(set(dataset_cache.loaded_keys()))
        self.assertEqual(list(dataset_cache.roots), [])

    def test_inline(self) -> None:
        name = 'acx'

        model_spec: ModelSpec = make_model_spec({
            name: make_dataset_csv_inline(),
        })

        dataset_cache = DatasetCache(model_spec, cwd=None)

        self.assertEqual(len(dataset_cache), 1)
        self.assertEqual(set(dataset_cache.keys()), {name})
        self.assertEqual(set(iter(dataset_cache)), {name})
        self.assertEmpty(set(dataset_cache.loaded_keys()))
        self.assertEqual(list(dataset_cache.roots), [])

        dataset: Dataset = dataset_cache[name]
        self.assertEqual(dataset.number_of_records(), 10)
        self.assertEqual(list(dataset.rvs), ['A', 'C', 'X'])
        self.assertEqual(set(dataset.value_set('A')), {'y', 'n'})
        self.assertEqual(set(dataset.value_set('C')), {'y', 'n'})
        self.assertEqual(set(dataset.value_set('X')), {'y', 'n'})

        self.assertEqual(len(dataset_cache), 1)
        self.assertEqual(set(dataset_cache.keys()), {name})
        self.assertEqual(set(iter(dataset_cache)), {name})
        self.assertEqual(set(dataset_cache.loaded_keys()), {name})
        self.assertEqual(list(dataset_cache.roots), [])

        # Getting the same dataset should return the same object
        dataset_again: Dataset = dataset_cache[name]
        self.assertIs(dataset_again, dataset)

        self.assertEqual(len(dataset_cache), 1)
        self.assertEqual(set(dataset_cache.keys()), {name})
        self.assertEqual(set(iter(dataset_cache)), {name})
        self.assertEqual(set(dataset_cache.loaded_keys()), {name})
        self.assertEqual(list(dataset_cache.roots), [])

    def test_absolute_filename(self) -> None:
        name = 'ace'

        with tmp_dir(chdir=False) as tmp:
            csv_file: Path = tmp / 'data.csv'  # absolute path location

            model_spec: ModelSpec = make_model_spec({
                name: make_dataset_csv_file(csv_file),
            })

            dataset_cache = DatasetCache(model_spec, cwd=None)

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEmpty(set(dataset_cache.loaded_keys()))
            self.assertEqual(list(dataset_cache.roots), [])

            dataset: Dataset = dataset_cache[name]
            self.assertEqual(dataset.number_of_records(), 10)
            self.assertEqual(list(dataset.rvs), ['A', 'C', 'E'])
            self.assertEqual(set(dataset.value_set('A')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('C')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('E')), {'y', 'n'})

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEqual(set(dataset_cache.loaded_keys()), {name})
            self.assertEqual(list(dataset_cache.roots), [])

    def test_relative_filename(self) -> None:
        name = 'ace'

        with tmp_dir(chdir=True) as tmp:
            csv_file = Path('data.csv')  # relative path location

            model_spec: ModelSpec = make_model_spec(
                {
                    name: make_dataset_csv_file(csv_file),
                },
                roots=[],
            )

            # Confirm the dataset is not findable without a cwd
            dataset_cache = DatasetCache(model_spec, cwd=None)
            with self.assertRaises(SynthorusError):
                _ = dataset_cache[name]

            # Confirm everything works when cwd is provided (implied root of '.')
            dataset_cache = DatasetCache(model_spec, cwd=tmp)

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEmpty(set(dataset_cache.loaded_keys()))
            self.assertEqual(list(dataset_cache.roots), [tmp])

            dataset: Dataset = dataset_cache[name]
            self.assertEqual(dataset.number_of_records(), 10)
            self.assertEqual(list(dataset.rvs), ['A', 'C', 'E'])
            self.assertEqual(set(dataset.value_set('A')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('C')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('E')), {'y', 'n'})

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEqual(set(dataset_cache.loaded_keys()), {name})
            self.assertEqual(list(dataset_cache.roots), [tmp])

    def test_relative_filename_with_sub_dir_location(self) -> None:
        name = 'ace'

        with tmp_dir(chdir=True) as tmp:
            csv_file = Path('a_dir/data.csv')  # relative path location in sub dir

            (tmp / 'a_dir').mkdir(parents=False, exist_ok=False)

            model_spec: ModelSpec = make_model_spec(
                {
                    name: make_dataset_csv_file(csv_file),
                },
                roots=[],
            )

            # Confirm the dataset is not findable without a 'a_dir' in roots
            dataset_cache = DatasetCache(model_spec, cwd=None)
            with self.assertRaises(SynthorusError):
                _ = dataset_cache[name]

            # Should work when cwd provided
            dataset_cache = DatasetCache(model_spec, cwd=tmp)

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEmpty(set(dataset_cache.loaded_keys()))
            self.assertEqual(list(dataset_cache.roots), [tmp])

            dataset: Dataset = dataset_cache[name]
            self.assertEqual(dataset.number_of_records(), 10)
            self.assertEqual(list(dataset.rvs), ['A', 'C', 'E'])
            self.assertEqual(set(dataset.value_set('A')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('C')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('E')), {'y', 'n'})

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEqual(set(dataset_cache.loaded_keys()), {name})
            self.assertEqual(list(dataset_cache.roots), [tmp])

    def test_relative_filename_with_sub_dir(self) -> None:
        name = 'ace'

        with tmp_dir(chdir=True) as tmp:
            csv_file = Path('data.csv')

            (tmp / 'a_dir').mkdir(parents=False, exist_ok=False)
            os.chdir(tmp / 'a_dir')

            model_spec: ModelSpec = make_model_spec(
                {
                    name: make_dataset_csv_file(csv_file),
                },
                roots=[],  # we change this later
            )

            # Confirm the dataset is not findable without a 'a_dir' in roots
            dataset_cache = DatasetCache(model_spec, cwd=tmp)
            self.assertEqual(list(dataset_cache.roots), [tmp])
            with self.assertRaises(SynthorusError):
                _ = dataset_cache[name]

            # Add 'a_dir' to roots and try again
            model_spec.roots.append('a_dir')
            dataset_cache = DatasetCache(model_spec, cwd=tmp)

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEmpty(set(dataset_cache.loaded_keys()))
            self.assertEqual(list(dataset_cache.roots), [tmp / 'a_dir'])

            dataset: Dataset = dataset_cache[name]
            self.assertEqual(dataset.number_of_records(), 10)
            self.assertEqual(list(dataset.rvs), ['A', 'C', 'E'])
            self.assertEqual(set(dataset.value_set('A')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('C')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('E')), {'y', 'n'})

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEqual(set(dataset_cache.loaded_keys()), {name})
            self.assertEqual(list(dataset_cache.roots), [tmp / 'a_dir'])

    def test_absolute_root(self) -> None:
        name = 'ace'

        with tmp_dir(chdir=True) as tmp:
            csv_file = Path('data.csv')

            model_spec: ModelSpec = make_model_spec(
                {
                    name: make_dataset_csv_file(csv_file),
                },
                roots=[],  # we change this later
            )

            # Confirm the dataset is not findable without a 'a_dir' in roots
            dataset_cache = DatasetCache(model_spec, cwd=None)
            self.assertEqual(list(dataset_cache.roots), [])
            with self.assertRaises(SynthorusError):
                _ = dataset_cache[name]

            # Add absolute directory to roots and try again
            model_spec.roots.append(str(tmp))
            dataset_cache = DatasetCache(model_spec, cwd=None)

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEmpty(set(dataset_cache.loaded_keys()))
            self.assertEqual(list(dataset_cache.roots), [tmp])

            dataset: Dataset = dataset_cache[name]
            self.assertEqual(dataset.number_of_records(), 10)
            self.assertEqual(list(dataset.rvs), ['A', 'C', 'E'])
            self.assertEqual(set(dataset.value_set('A')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('C')), {'y', 'n'})
            self.assertEqual(set(dataset.value_set('E')), {'y', 'n'})

            self.assertEqual(len(dataset_cache), 1)
            self.assertEqual(set(dataset_cache.keys()), {name})
            self.assertEqual(set(iter(dataset_cache)), {name})
            self.assertEqual(set(dataset_cache.loaded_keys()), {name})
            self.assertEqual(list(dataset_cache.roots), [tmp])

    def test_multiple_files(self) -> None:
        with tmp_dir(chdir=False) as tmp:
            model_spec: ModelSpec = make_model_spec({
                'csv_inline': make_dataset_csv_inline(),
                'csv_file': make_dataset_csv_file(tmp / 'data.csv'),
                'feather': make_dataset_feather_file(tmp / 'data.feather'),
            })

            dataset_cache = DatasetCache(model_spec, cwd=tmp)

            self.assertEqual(len(dataset_cache), 3)
            self.assertEqual(set(dataset_cache.keys()), {'csv_inline', 'csv_file', 'feather'})
            self.assertEqual(set(iter(dataset_cache)), {'csv_inline', 'csv_file', 'feather'})
            self.assertEmpty(set(dataset_cache.loaded_keys()))
            self.assertEqual(list(dataset_cache.roots), [tmp])

            dataset: Dataset = dataset_cache['csv_inline']
            self.assertEqual(list(dataset.rvs), ['A', 'C', 'X'])

            self.assertEqual(set(dataset_cache.loaded_keys()), {'csv_inline'})

            dataset: Dataset = dataset_cache['csv_file']
            self.assertEqual(list(dataset.rvs), ['A', 'C', 'E'])

            self.assertEqual(set(dataset_cache.loaded_keys()), {'csv_inline', 'csv_file'})

            dataset: Dataset = dataset_cache['feather']
            self.assertEqual(list(dataset.rvs), ['A', 'B', 'C'])

            self.assertEqual(set(dataset_cache.loaded_keys()), {'csv_inline', 'csv_file', 'feather'})


if __name__ == '__main__':
    test_main()
