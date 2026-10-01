"""Exercise signal changes, transient pulses, boundary times and trial counting."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "fsm_readout", Path(__file__).parents[1]/"scripts/read_bca_ip_fsm_outputs.py")
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)
INSTANCE = {"a": [1, 1, 2, 2, 4, 4], "b": 8, "c": [2, 2, 1, 5, 5, 5]}


def steady(vector, start=0, end=300_000):
    return [np.arange(start+2500, end+1, 5000) if bit else np.array([], dtype=int)
            for bit in vector]


class TerminalFSMReadoutTests(unittest.TestCase):
    def test_exhaustive_instance_has_both_degenerate_optima(self):
        value, vectors = readout.solve_binary_instance(INSTANCE)
        self.assertEqual(value, 14)
        self.assertEqual(vectors, [[1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 1, 0]])

    def test_final_regime_does_not_or_together_old_and_new_solutions(self):
        old = steady([1, 1, 0, 1, 1, 0], end=150_000)
        new = steady([1, 1, 0, 1, 0, 1], start=150_000)
        wires = [np.r_[a, b] for a, b in zip(old, new)]
        counts = np.stack([np.histogram(t-1, bins=np.arange(0, 310_000, 10_000))[0]
                           for t in wires], axis=-1)[None]
        edges = readout.segment_rates(counts)[0]
        self.assertEqual(edges, [0, 15, 30])
        result = readout.decode_terminal(wires, edges[-2]*10_000, 300_000)
        self.assertTrue(result["stable_readout"])
        self.assertEqual(result["vector"], [1, 1, 0, 1, 0, 1])

    def test_two_isolated_residual_pulses_do_not_turn_on_a_wire(self):
        wires = steady([1, 1, 0, 1, 0, 1])
        wires[2] = np.array([110_000, 250_000])
        result = readout.decode_terminal(wires, 0, 300_000)
        self.assertEqual(result["vector"], [1, 1, 0, 1, 0, 1])

    def test_burst_is_not_persistent_flow_and_is_not_forced_to_a_bit(self):
        wires = steady([1, 1, 0, 1, 0, 0])
        wires[5] = np.array([101_000, 102_000, 103_000, 104_000, 105_000])
        result = readout.decode_terminal(wires, 0, 300_000)
        self.assertFalse(result["stable_readout"])
        self.assertIsNone(result["vector"][5])

    def test_recent_optimum_is_separate_from_a_confirmed_stable_readout(self):
        result = readout.decode_terminal(steady([1, 1, 0, 1, 0, 1], start=250_000),
                                         250_000, 300_000)
        self.assertEqual(result["vector"], [1, 1, 0, 1, 0, 1])
        self.assertFalse(result["sufficient_duration"])
        self.assertFalse(readout.score_unit(result, INSTANCE, 14)["optimal"])

    def test_infeasible_vector_is_preserved_instead_of_repaired_to_an_optimum(self):
        result = readout.decode_terminal(steady([1, 1, 0, 1, 1, 1]), 0, 300_000)
        scored = readout.score_unit(result, INSTANCE, 14)
        self.assertEqual(scored["vector"], [1, 1, 0, 1, 1, 1])
        self.assertFalse(scored["feasible"])
        self.assertFalse(scored["optimal"])

    def test_exact_completed_update_boundaries(self):
        wires = [np.array([100_000, 150_000, 200_000, 250_000, 300_000])]*6
        result = readout.decode_terminal(wires, 100_000, 300_000)
        self.assertEqual(result["quarter_counts"], [[1, 1, 1, 1]]*6)
        self.assertEqual(result["counts"], [4]*6)

    def test_both_units_are_one_trial_and_either_unit_can_succeed(self):
        units = [steady([1, 1, 0, 1, 0, 1]), steady([1, 1, 0, 1, 1, 0]),
                 steady([1, 1, 0, 1, 0, 0]), steady([1, 1, 0, 1, 1, 0]),
                 steady([1, 1, 0, 1, 0, 0]), steady([1, 1, 0, 1, 0, 0])]
        _, result = readout.evaluate([wire for unit in units for wire in unit],
                                     [[0, 30]]*6, 10_000, 3, 300_000, INSTANCE)
        self.assertEqual(result["optimal_trials"], 2)
        self.assertEqual(result["both_units_optimal_trials"], 1)
        self.assertEqual(result["stable_nonoptimal_trials"], 1)

    def test_rate_change_and_no_signal_segments(self):
        counts = np.zeros((2, 60, 6), dtype=int)
        counts[0, :30, 0] = 20
        counts[0, 30:, 0] = 4
        edges = readout.segment_rates(counts)
        self.assertEqual(edges, [[0, 30, 60], [0, 60]])


if __name__ == "__main__":
    unittest.main()
