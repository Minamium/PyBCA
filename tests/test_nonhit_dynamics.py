"""Readout plateaus must not cross unknown or different solution regimes."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("nonhit", SCRIPTS/"analyze_bca_ip_nonhits.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


class NonhitDynamicsTests(unittest.TestCase):
    def test_empty_outcome_group_is_reportable_without_nan(self):
        summary, records = analysis.group_characteristics([], [], 3_000_000, [])
        self.assertEqual(summary["trials"],0)
        self.assertEqual(records,[])
        self.assertIsNone(summary["reset_median"])

    def test_same_solution_across_rate_changes_counts_as_one_plateau(self):
        rows = [{"stable_readout":True,"vector":[1,1,0,1,0,0],"segment_start":s}
                for s in (10,30,50)]
        self.assertEqual(analysis.same_vector_tail_duration(rows,100),90)

    def test_unknown_or_different_vector_breaks_plateau(self):
        rows = [{"stable_readout":True,"vector":[1,0],"segment_start":0},
                {"stable_readout":False,"vector":[1,None],"segment_start":20},
                {"stable_readout":True,"vector":[1,0],"segment_start":40}]
        self.assertEqual(analysis.same_vector_tail_duration(rows,100),60)
        rows[1] = {"stable_readout":True,"vector":[0,1],"segment_start":20}
        self.assertEqual(analysis.same_vector_tail_duration(rows,100),60)

    def test_suboptimal_packing_can_require_removal_before_reaching_an_optimum(self):
        targets = [[1,1,0,1,0,1],[1,1,0,1,1,0]]
        for vector in ([1,1,1,1,0,0],[0,0,0,0,1,1]):
            self.assertFalse(analysis.can_reach_by_additions(vector,targets))
        for vector in ([1,1,0,1,0,0],[0,1,0,1,1,0]):
            self.assertTrue(analysis.can_reach_by_additions(vector,targets))


if __name__ == "__main__":
    unittest.main()
