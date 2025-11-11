import warnings
from io import StringIO

import numpy as np

from synthorus.utils.multiprocessing_extras import num_processes, DefaultTrialLogger, run_trial_processes
from tests.helpers.unittest_fixture import Fixture, test_main


class Trial:
    def __init__(self, result, warning=None, error=None):
        self.result = result
        self.warning = warning
        self.error = error

    def __call__(self):
        if self.warning is not None:
            warnings.warn(self.warning)
        if self.error is not None:
            raise self.error
        return self.result


class TestMultiprocessingExtras(Fixture):

    def test_num_processes(self):
        assume_cpu_count = 8
        self.assertEqual(1, num_processes('single'))
        self.assertEqual(assume_cpu_count, num_processes('max', assume_cpu_count))
        self.assertEqual(assume_cpu_count, num_processes('all', assume_cpu_count))
        self.assertEqual(assume_cpu_count / 2, num_processes('half', assume_cpu_count))

        self.assertEqual(assume_cpu_count / 2, num_processes(0.5, assume_cpu_count))
        self.assertEqual(assume_cpu_count / 4, num_processes(0.25, assume_cpu_count))
        self.assertEqual(assume_cpu_count / 4, num_processes(-0.75, assume_cpu_count))

        self.assertEqual(1, num_processes(1))
        self.assertEqual(1, num_processes(0))
        self.assertEqual(assume_cpu_count - 1, num_processes(-1, assume_cpu_count))

        np_array = np.array([0, 1, -1], dtype=np.intc)
        self.assertEqual(1, num_processes(np_array.item(0)))
        self.assertEqual(1, num_processes(np_array.item(1)))
        self.assertEqual(assume_cpu_count - 1, num_processes(np_array.item(2), assume_cpu_count))

        np_array = np.array([0.5, 0.25, -0.75], dtype=np.single)
        self.assertEqual(assume_cpu_count / 2, num_processes(np_array.item(0), assume_cpu_count))
        self.assertEqual(assume_cpu_count / 4, num_processes(np_array.item(1), assume_cpu_count))
        self.assertEqual(assume_cpu_count / 4, num_processes(np_array.item(2), assume_cpu_count))

        with self.assertRaises(ValueError):
            num_processes('some rubbish')

    def test_run_trial_processes_single(self):
        trials = [
            (lambda: 1),
            (lambda: 2),
            (lambda: 3),
        ]
        collector = sum
        logger = DefaultTrialLogger(log=None)

        result = run_trial_processes(trials, collector, logger, 1)
        self.assertEqual(6, result)

    def test_run_trial_processes_multi(self):
        trials = [
            Trial(1),
            Trial(2),
            Trial(3),
        ]
        collector = sum
        logger = DefaultTrialLogger(log=None)

        result = run_trial_processes(trials, collector, logger, 2)
        self.assertEqual(6, result)

    def test_run_trial_processes_with_known_length(self):
        trials = [
            (lambda: 1),
            (lambda: 2),
            (lambda: 3),
        ]
        collector = sum
        logger = DefaultTrialLogger(log=None, num_trials=len(trials))

        result = run_trial_processes(trials, collector, logger, 1)
        self.assertEqual(6, result)

    def test_run_trial_processes_with_warnings(self):
        trials = [
            Trial(1),
            Trial(2, warning='trial 2 warning'),
            Trial(3),
        ]
        collector = sum
        logger = DefaultTrialLogger(log=None)

        result = run_trial_processes(trials, collector, logger, 1)
        self.assertEqual(6, result)

    def test_run_trial_processes_with_warnings_filtered(self):
        trials = [
            Trial(1),
            Trial(2, warning='trial 2 warning'),
            Trial(3),
        ]
        collector = sum
        logger = DefaultTrialLogger(log=None)

        result = run_trial_processes(trials, collector, logger, 1, discard_results_with_warning=True)
        self.assertEqual(4, result)

    def test_run_trial_processes_with_error(self):
        trials = [
            Trial(1),
            Trial(2, error=RuntimeError('trial 2 error')),
            Trial(3),
        ]
        collector = sum

        log_result = StringIO()

        def _print(*argv, **kwargs) -> None:
            print(*argv, file=log_result, **kwargs)

        logger = DefaultTrialLogger(log=_print)

        result = run_trial_processes(trials, collector, logger, 1)
        self.assertEqual(4, result)

        log = log_result.getvalue()
        self.assertIn('trial 2 error', log)


if __name__ == '__main__':
    test_main()
