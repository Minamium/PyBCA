"""Prevent overlap of independent trials and incorrect multiworker throughput."""
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_bca_ip_gpu as benchmark


class GPUCapacityTests(unittest.TestCase):
    def test_shards_cover_every_trial_once_within_index_limit(self):
        for total in [1, 512, 3207, 3208, 3584, 6414]:
            parts = benchmark.partition_trials(total, 3207)
            ids = [i for p in parts for i in range(p["start"], p["start"] + p["count"])]
            self.assertEqual(ids, list(range(total)))
            self.assertLessEqual(max(p["count"] for p in parts), 3207)
            self.assertLessEqual(len(parts), 2)
        with self.assertRaises(ValueError):
            benchmark.partition_trials(6415, 3207)

    def test_cohort_timing_counts_all_trials_once_and_all_checkpoints(self):
        result = benchmark.project_runtime([0.2, 0.4, 0.3], 2, 10, trials=512,
                    target_steps=10000, checkpoint_interval=1000, slowdown=1.1)
        self.assertAlmostEqual(result["trial_steps_per_sec"], 512 / 0.3)
        self.assertEqual(result["checkpoint_count_in_projection"], 12)
        self.assertAlmostEqual(result["nominal_hours_for_target"], (10 + 3000 + 24) / 3600)
        self.assertAlmostEqual(result["conservative_hours_for_target"], (10 + 1.1 * (4000 + 24)) / 3600)


if __name__ == "__main__":
    unittest.main()
