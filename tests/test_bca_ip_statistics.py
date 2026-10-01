"""Known first-hit samples check the meaning of failure and censoring."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("bca_statistics", Path(__file__).parents[1]/"scripts/summarize_bca_ip_trials.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def row(trial, hit, observed=100):
    return {"trial_id": trial, "first_hit_step": hit, "observed_steps": observed, "decoder_sha256": "a"*64}


class FirstHitTests(unittest.TestCase):
    def test_timeout_is_not_a_success_and_changes_restricted_mean(self):
        result = stats.summarize([row(1, 10), row(2, 30), row(3, None), row(4, None)], 100, 4)
        self.assertEqual(result["failure_rate_at_horizon"], .5)
        self.assertEqual(result["mean_hit_steps_among_observed_successes"], 20)
        self.assertEqual(result["restricted_mean_steps_at_horizon"], 60)
        self.assertIsNone(result["unrestricted_mean_hit_steps"])
        self.assertEqual(result["observed_reached_fraction_curve"][-1]["observed_reached_fraction"], .5)

    def test_interruption_is_unknown_not_an_algorithm_failure(self):
        result = stats.summarize([row(1, 10, 10), row(2, None, 50), row(3, None)], 100, 3)
        self.assertEqual(result["confirmed_failures_at_horizon"], 1)
        self.assertEqual(result["incomplete_without_observed_hit"], 1)
        self.assertIsNone(result["failure_rate_at_horizon"])
        self.assertIsNone(result["restricted_mean_steps_at_horizon"])
        self.assertEqual(result["failure_rate_bounds_with_incomplete"], [1/3, 2/3])

    def test_initial_state_and_last_allowed_step_count_as_success(self):
        result = stats.summarize([row(1, 0), row(2, 100)], 100, 2)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["unrestricted_mean_hit_steps"], 50)
        self.assertEqual(result["failure_rate_at_horizon"], 0)

    def test_all_failures_have_no_conditional_hit_mean(self):
        result = stats.summarize([row(1, None), row(2, 101, 101)], 100, 2)
        self.assertIsNone(result["mean_hit_steps_among_observed_successes"])
        self.assertEqual(result["restricted_mean_steps_at_horizon"], 100)
        self.assertEqual(result["failure_rate_at_horizon"], 1)

    def test_reject_incomplete_roster_duplicate_trials_and_mixed_definitions(self):
        with self.assertRaises(ValueError):
            stats.summarize([row(1, 1)], 100, 2)
        with self.assertRaises(ValueError):
            stats.summarize([row(1, 1), row(1, 2)], 100, 2)
        with self.assertRaises(ValueError):
            stats.summarize([row(1, 1), row(2, 2) | {"decoder_sha256": "b"*64}], 100, 2)


if __name__ == "__main__":
    unittest.main()
