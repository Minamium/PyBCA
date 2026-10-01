"""A short stable regime may be reached and later lost; count trials once."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("short_fsm", SCRIPTS/"sweep_bca_ip_stability.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
finally:
    sys.path.pop(0)

INSTANCE = {"a": [1,1,2,2,4,4], "b": 8, "c": [2,2,1,5,5,5]}


class ShortStabilityTests(unittest.TestCase):
    def test_reached_then_lost_remains_separate_from_terminal_retention(self):
        wires = [np.arange(2500, 100_001, 5000) if b else np.array([], dtype=int)
                 for b in [1,1,0,1,0,1]]
        wires += [np.array([], dtype=int) for _ in range(6)]
        evidence = sweep.collect_regime_evidence(wires, [[0,10,30],[0,30]], 10_000, INSTANCE)
        terminal, _ = sweep.evaluate(wires, [[0,10,30],[0,30]], 10_000, 1, 300_000, INSTANCE)
        result, _ = sweep.classify_duration(evidence, terminal, 100_000)
        self.assertEqual(result["ever_supported_optimum_trials"], 1)
        self.assertEqual(result["terminal_optimal_trials"], 0)
        self.assertEqual(result["earlier_supported_but_not_terminal_ids"], [0])

    def test_shortening_duration_adds_evidence_without_reclassifying_its_bits(self):
        evidence = [{"trial_id":0,"regime_duration":40_000,"segment_end":300_000,"unit":"B"}]
        terminal = [{"trial_id":0,"status":"unresolved"}]
        long, _ = sweep.classify_duration(evidence, terminal, 100_000)
        short, _ = sweep.classify_duration(evidence, terminal, 40_000)
        self.assertEqual(long["ever_supported_optimum_trials"], 0)
        self.assertEqual(short["ever_supported_optimum_trials"], 1)

    def test_earliest_evidence_compares_both_units_and_counts_trial_once(self):
        evidence = [{"trial_id":0,"regime_duration":100_000,"segment_end":200_000,"unit":"A"},
                    {"trial_id":0,"regime_duration":100_000,"segment_end":100_000,"unit":"B"}]
        terminal = [{"trial_id":0,"status":"optimal"}]
        result, first = sweep.classify_duration(evidence, terminal, 100_000)
        self.assertEqual(result["ever_supported_optimum_trials"], 1)
        self.assertEqual(first[0]["unit"], "B")

    def test_fine_split_can_detect_a_change_shorter_than_the_old_floor(self):
        counts = np.zeros((1,60,6),dtype=int)
        counts[0,58:,1] = 10
        self.assertEqual(sweep.segment_rates(counts, min_bins=1), [[0,58,60]])


if __name__ == "__main__":
    unittest.main()
