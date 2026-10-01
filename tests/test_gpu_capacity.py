"""Prevent overlap of independent trials and incorrect multiworker throughput."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_bca_ip_gpu as benchmark
import profile_bca_ip_capacity as continuation


class GPUCapacityTests(unittest.TestCase):
    def test_continuation_rejects_changed_inputs_or_unfinished_search(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "core.py"
            source.write_text("validated core")
            probe = {"trials": 3584, "memory_fits": True}
            plan = {"schema": "pybca-single-gpu-capacity-v1",
                    "gpu_total_memory_bytes": 80,
                    "largest_short_probe_fitting_memory": 3584,
                    "profile_512": {"status": "ok"},
                    "sha256": {"core.py": hashlib.sha256(source.read_bytes()).hexdigest()},
                    "measurements": [probe]}
            with patch.object(continuation, "ROOT", root):
                self.assertEqual(continuation.validate_source(plan, 80), (3584, probe))
                with self.assertRaisesRegex(ValueError, "GPU memory"):
                    continuation.validate_source(plan, 40)
                with self.assertRaisesRegex(ValueError, "completed baseline"):
                    continuation.validate_source({**plan, "largest_short_probe_fitting_memory": None}, 80)
                source.write_text("different core")
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    continuation.validate_source(plan, 80)

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
