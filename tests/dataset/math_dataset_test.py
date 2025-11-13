from synthorus.dataset import MathRV, MathDataset
from tests.helpers.unittest_fixture import Fixture, test_main


class MathDatasetTest(Fixture):

    def test_function_string(self):
        max_kg = 100
        max_cm = 200
        weight = MathRV('weight', list(range(1, 1 + max_kg)))
        height = MathRV('height', list(range(1, 1 + max_cm)))
        func = 'int(weight / height / height * 100000 + 0.5) / 10'

        dataset = MathDataset(
            input_rvs=[weight, height],
            output_name='bmi',
            func=func
        )

        self.assertArrayEqual(dataset.rvs, ['weight', 'height', 'bmi'])
        self.assertArrayEqual(dataset.dataframe.shape, (max_kg * max_cm, 3))

        for row in dataset.dataframe.itertuples(index=False):
            bmi = int(row.weight / row.height / row.height * 100000 + 0.5) / 10
            self.assertEqual(bmi, row.bmi)

    def test_function_callable(self):
        max_kg = 100
        max_cm = 200
        weight = MathRV('weight', list(range(1, 1 + max_kg)))
        height = MathRV('height', list(range(1, 1 + max_cm)))

        def func(_weight: float, _height: float) -> float:
            return int(_weight / _height / _height * 100000 + 0.5) / 10

        dataset = MathDataset(
            input_rvs=[weight, height],
            output_name='bmi',
            func=func
        )

        self.assertArrayEqual(dataset.rvs, ['weight', 'height', 'bmi'])
        self.assertArrayEqual(dataset.dataframe.shape, (max_kg * max_cm, 3))

        for row in dataset.dataframe.itertuples(index=False):
            bmi = int(row.weight / row.height / row.height * 100000 + 0.5) / 10
            self.assertEqual(bmi, row.bmi)

    def test_function_error(self):
        max_kg = 100
        max_cm = 200
        weight = MathRV('weight', list(range(1, 1 + max_kg)))
        height = MathRV('height', list(range(1, 1 + max_cm)))

        with self.assertRaises(TypeError):
            _ = MathDataset(
                input_rvs=[weight, height],
                output_name='bmi',
                func=1234
            )


if __name__ == '__main__':
    test_main()
