import unittest
import numpy as np
from pso import PSO


class TestPSO(unittest.TestCase):
    def setUp(self):
        self.pso = PSO(fonction=lambda x, y: x**2 + y**2, particle=2, iteration=50, dim=2, min=-10, max=10)

    def test_eval_fonction(self):
        self.pso.position = np.array([[3, 4], [5, 12]])
        expected_fitness = np.array([25, 169])
        np.testing.assert_array_equal(self.pso.eval_fonction(), expected_fitness)

    def test_update_personal_best(self):
        self.pso.position = np.array([[3, 4], [5, 12]])
        self.pso.fitness = np.array([25, 169])
        self.pso.personal_best_fitness = np.array([30, 200])
        self.pso.update_personal_best()
        np.testing.assert_array_equal(self.pso.personal_best_position, self.pso.position)
        np.testing.assert_array_equal(self.pso.personal_best_fitness, self.pso.fitness)


if __name__ == "__main__":
    unittest.main()
