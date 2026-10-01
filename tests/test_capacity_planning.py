"""Budget selection must not recommend an unmeasured or unsafe batch."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("capacity_planning", Path(__file__).parents[1]/"scripts/plan_bca_ip_capacity.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


class CapacityTests(unittest.TestCase):
    def test_bracket_and_bisect_including_zero_and_limit(self):
        for ceiling in (0, 1, 17, 32, 63, 64, 65, 111, 128):
            seen = []
            def measure(n):
                seen.append(n)
                return n <= ceiling
            chosen = capacity.largest_measured_batch(measure, initial=32, maximum=128)
            self.assertEqual(chosen, ceiling)
            if chosen:
                self.assertIn(chosen, seen)
            if chosen < 128:
                self.assertIn(chosen+1, seen)
            self.assertEqual(len(seen), len(set(seen)))

    def test_budget_counts_worst_repeat_saving_setup_and_reserve(self):
        record = {"seconds_per_step_samples": [.010, .020, .015], "flush_checkpoint_sec": 2.,
                  "setup_sec": 8., "peak_allocated_bytes": 80, "peak_reserved_bytes": 100}
        args = dict(target_steps=1000, wall_seconds=53., reserve_seconds=10.,
                    slowdown_factor=1.5, memory_limit_bytes=100, checkpoint_interval=1000)
        result = capacity.assess_measurement(record, **args)
        self.assertAlmostEqual(result["projected_total_seconds"], 53.)
        self.assertTrue(result["fits"])
        self.assertFalse(capacity.assess_measurement(record, **(args | {"wall_seconds": 52.9}))["fits"])
        self.assertFalse(capacity.assess_measurement(record, **(args | {"memory_limit_bytes": 99}))["fits"])


if __name__ == "__main__":
    unittest.main()
