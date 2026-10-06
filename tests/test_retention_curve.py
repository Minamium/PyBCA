"""Guard prefix causality, shared decoder semantics, and trial-level counting."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

scripts = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('retention', scripts/'plot_bca_ip_retention.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from read_bca_ip_fsm_outputs import segment_rates
finally:
    sys.path.pop(0)


class PrefixRetentionTests(unittest.TestCase):
    def test_prefix_boundaries_equal_independent_original_fits(self):
        counts = np.random.default_rng(731).poisson(1, (3, 36, 6))
        counts[0, 12:, 0] += 9
        last = module.prefix_last_changes(counts)
        for end in (3, 9, 12, 18, 24, 36):
            expected = [b[-2] for b in segment_rates(counts[:, :end])]
            np.testing.assert_array_equal(last[:, end], expected)

    def test_changing_future_events_cannot_change_earlier_readouts(self):
        counts = np.zeros((2, 40, 6), dtype=int)
        counts[0, :, 0] = 5
        original = module.prefix_last_changes(counts)
        counts[:, 20:] = 500
        changed = module.prefix_last_changes(counts)
        np.testing.assert_array_equal(original[:, :21], changed[:, :21])

    def test_either_unit_succeeds_and_future_signal_is_excluded(self):
        optimal = [1,1,0,1,0,1]
        wires = [np.arange(20500,100001,1000) if b else np.array([],dtype=int)
                 for _ in range(2) for b in optimal]
        last = np.zeros((2,11), dtype=int)
        instance = dict(a=[1,1,2,2,4,4],b=8,c=[2,2,1,5,5,5])
        at20k = module.read_prefix(wires,last,20000,10000,1,instance,14)
        at100k = module.read_prefix(wires,last,100000,10000,1,instance,14)
        np.testing.assert_array_equal(at20k, [[1],[0]])
        np.testing.assert_array_equal(at100k, [[2],[2]])

    def test_wilson_interval_matches_published_endpoint(self):
        low,high = module.wilson_interval(446,512)
        self.assertAlmostEqual(float(low),0.8392810633432318)
        self.assertAlmostEqual(float(high),0.8973793843624412)


if __name__ == '__main__':
    unittest.main()
