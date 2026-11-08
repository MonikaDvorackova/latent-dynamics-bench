import unittest

from main import fit, run


class LatentDynamicsTests(unittest.TestCase):
    def test_fit_recovers_noise_free_linear_transition(self):
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        actions = [1.0, -1.0, 0.5, -0.5, 0.25]
        next_xs = [0.92 * x + 0.35 * u for x, u in zip(xs, actions)]
        state, action, bias = fit(xs, actions, next_xs)
        self.assertAlmostEqual(state, 0.92, delta=1e-4)
        self.assertAlmostEqual(action, 0.35, delta=1e-4)
        self.assertAlmostEqual(bias, 0.0, delta=1e-4)

    def test_multiseed_report_is_deterministic_and_compares_baseline(self):
        first = run([0, 1, 2], training_seed=7, horizon=25)
        second = run([0, 1, 2], training_seed=7, horizon=25)
        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_seeds"], [0, 1, 2])
        self.assertEqual(len(first["episodes"]), 3)
        self.assertIn("mean_rollout_rmse", first["summary"])
        self.assertIn("mean_persistence_baseline_rmse", first["summary"])
        self.assertTrue(all(item["rollout_rmse"] >= 0 for item in first["episodes"]))

    def test_invalid_training_arrays_fail_closed(self):
        with self.assertRaises(ValueError):
            fit([], [], [])


if __name__ == "__main__":
    unittest.main()
