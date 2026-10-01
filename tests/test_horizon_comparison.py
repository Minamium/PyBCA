"""The paired comparison must distinguish observation loss from refitting."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "horizon_comparison", Path(__file__).parents[1] / "scripts/compare_bca_ip_horizons.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def row(hits, final, n=4):
    return {"minimum_duration": 10000, "ever_supported_trial_ids": hits,
            "terminal_trial_ids": final, "ever_supported_optimum_trials": len(hits),
            "terminal_optimal_trials": len(final), "terminal_nonoptimal_trials": n-len(final),
            "terminal_unresolved_trials": 0}


class HorizonComparisonTests(unittest.TestCase):
    def test_refitting_does_not_erase_previously_observed_evidence(self):
        result = comparison.compare_row(row([0, 1], [0]), row([1, 2], [2]), 4)
        ids = result["trial_ids"]
        self.assertEqual(ids["prior_support_not_reproduced_by_longer_fit"], [0])
        self.assertEqual(ids["cumulative_supported"], [0, 1, 2])
        self.assertEqual(ids["cumulative_never_supported"], [3])
        self.assertEqual(ids["cumulative_supported_not_terminal"], [0, 1])

    def test_new_terminal_can_be_a_recovery_or_a_new_observation(self):
        result = comparison.compare_row(row([0, 1], [0]), row([0, 1, 2], [0, 1, 2]), 4)
        ids = result["trial_ids"]
        self.assertEqual(ids["previously_supported_only_now_terminal"], [1])
        self.assertEqual(ids["previously_never_supported_now_terminal"], [2])
        self.assertEqual(ids["new_terminal"], [1, 2])

    def test_terminal_must_have_supporting_evidence(self):
        with self.assertRaises(ValueError):
            comparison.compare_row(row([0], [0]), row([0], [1]), 4)


if __name__ == "__main__":
    unittest.main()
