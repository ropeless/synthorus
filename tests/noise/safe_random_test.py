import math
from collections import Counter
from itertools import chain

import numpy as np

from synthorus.noise.safe_random import SafeRandom
from tests.helpers.unittest_fixture import test_main, Fixture


class SafeRandomTest(Fixture):

    def _test_choice_index(self, n, num_samples, num_options, tol):
        # Create a SafeRandom(n) then sample choice_index(num_options)
        # asserting that the resulting distribution is approximately
        # uniform.

        distribution = Counter()
        rnd = SafeRandom(n=n)

        self.assertEqual(rnd.n, n)

        for _ in range(num_samples):
            option = rnd.uniform(num_options)
            distribution[option] += 1

        self.assertArraySetEqual(list(range(num_options)), distribution.keys())
        expect = num_samples / num_options
        low = expect * (1 - tol)
        high = expect * (1 + tol)

        for option, count in distribution.items():
            self.assertTrue(low < count < high, msg=f'{option} out of range: {low} < {count} < {high}')

    def _test_samples(self, samples_1, samples_2, bins, tol):
        # Bin samples_1 and samples_2 according to the given bin
        # thresholds, then assert the bins are approximately equal.

        self.assertEqual(len(samples_1), len(samples_2))

        h1, _ = np.histogram(samples_1, bins=bins)
        h2, _ = np.histogram(samples_2, bins=bins)

        tol_count = int(max(chain(h1, h2)) * tol + 0.5)
        for c1, c2 in zip(h1, h2):
            if abs(c1 - c2) > tol_count:
                self.fail(f'a histogram bucket is excessively different: {c1} vs {c2}, more than {tol_count}')

    def test_choice_2(self):
        self._test_choice_index(
            n=4,
            num_samples=10000,
            num_options=2,
            tol=0.05
        )

    def test_choice_3(self):
        self._test_choice_index(
            n=4,
            num_samples=10000,
            num_options=3,
            tol=0.05
        )

    def test_choice_4(self):
        self._test_choice_index(
            n=4,
            num_samples=10000,
            num_options=4,
            tol=0.1
        )

    def test_random_uniform(self):
        num_samples = 10000
        rnd = SafeRandom(n=4)
        tol = 0.1
        num_bins = 10
        bins = [x / num_bins for x in range(num_bins)]

        samples = [rnd.random() for _ in range(num_samples)]
        histogram, _ = np.histogram(samples, bins=bins)

        low_count = num_samples / num_bins * (1 - tol)
        high_count = num_samples / num_bins * (1 + tol)
        for count in histogram:
            if not (low_count <= count <= high_count):
                self.fail(
                    f'a histogram bucket is excessively different,'
                    f' expected: {low_count} <= {count} <= {high_count}'
                )

    def test_gauss(self):
        num_samples = 10000
        rnd = SafeRandom(n=4)
        mu = 0
        sigma = 1
        tol = 0.2
        num_pos_bins = 10
        pos_bins = [-math.log(x / num_pos_bins) for x in range(num_pos_bins - 1, 0, -1)]
        neg_bins = [-b for b in reversed(pos_bins)]
        bins = neg_bins + [0] + pos_bins

        self._test_samples(
            np.random.normal(mu, sigma, num_samples),
            [rnd.gauss(mu, sigma) for _ in range(num_samples)],
            bins=bins,
            tol=tol
        )

    def test_laplace(self):
        num_samples = 10000
        rnd = SafeRandom(n=4)
        loc = 0
        scale = 1
        tol = 0.2
        num_pos_bins = 10
        pos_bins = [-math.log(x / num_pos_bins) for x in range(num_pos_bins - 1, 0, -1)]
        neg_bins = [-b for b in reversed(pos_bins)]
        bins = neg_bins + [0] + pos_bins

        self._test_samples(
            np.random.laplace(loc, scale, num_samples),
            [rnd.laplace(loc, scale) for _ in range(num_samples)],
            bins=bins,
            tol=tol
        )

    def test_laplace_scale_half(self):
        # This test will fail if numpy and SafeRandom interpret
        # the Laplace scale factor differently.
        num_samples = 10000
        rnd = SafeRandom(n=4)
        loc = 0
        scale = 0.5
        tol = 0.2
        num_pos_bins = 10
        pos_bins = [-math.log(x / num_pos_bins) for x in range(num_pos_bins - 1, 0, -1)]
        neg_bins = [-b for b in reversed(pos_bins)]
        bins = neg_bins + [0] + pos_bins

        self._test_samples(
            np.random.laplace(loc, scale, num_samples),
            [rnd.laplace(loc, scale) for _ in range(num_samples)],
            bins=bins,
            tol=tol
        )

    def test_laplace_scale_two(self):
        # This test will fail if numpy and SafeRandom interpret
        # the Laplace scale factor differently.
        num_samples = 10000
        rnd = SafeRandom(n=4)
        loc = 0
        scale = 2.0
        tol = 0.2
        num_pos_bins = 10
        pos_bins = [-math.log(x / num_pos_bins) for x in range(num_pos_bins - 1, 0, -1)]
        neg_bins = [-b for b in reversed(pos_bins)]
        bins = neg_bins + [0] + pos_bins

        self._test_samples(
            np.random.laplace(loc, scale, num_samples),
            [rnd.laplace(loc, scale) for _ in range(num_samples)],
            bins=bins,
            tol=tol
        )

    def test_normal_approx_to_binomial(self):
        num_samples = 100000
        rnd = SafeRandom(n=4)
        tol = 0.2
        n = 100
        p = 0.5
        bins = [0, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 100]

        self._test_samples(
            np.random.binomial(n, p, num_samples),
            [rnd._normal_approx_to_binomial(n, p) for _ in range(num_samples)],
            bins=bins,
            tol=tol
        )

    def test_algorithm_bin(self):
        num_samples = 10000
        rnd = SafeRandom(n=4)
        tol = 0.2
        n = 100
        p = 0.5
        bins = [0, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 100]

        self._test_samples(
            np.random.binomial(n, p, num_samples),
            [rnd._algorithm_bin(n, p) for _ in range(num_samples)],
            bins=bins,
            tol=tol
        )


if __name__ == '__main__':
    test_main()
