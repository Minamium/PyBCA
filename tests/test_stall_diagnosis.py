import importlib.util
from pathlib import Path
import sys
import unittest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0,str(scripts))
try:
    spec = importlib.util.spec_from_file_location("stall_diagnosis",scripts/"diagnose_bca_ip_stalls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


class StallDiagnosisTests(unittest.TestCase):
    def test_joint_stability_excludes_a_change_in_either_unit(self):
        self.assertEqual(module.joint_intervals([400000],[200000,700000],1000000),
                         [(400000,700000),(700000,1000000)])

    def test_equal_objectives_are_not_called_a_ranking_error(self):
        self.assertEqual(module.ranking(10,10,100,500),"objective_tie")
        self.assertEqual(module.ranking(14,10,100,100),"flow_tie")
        self.assertEqual(module.ranking(14,10,100,500),"reverse")
        self.assertEqual(module.ranking(10,14,100,500),"agree")


if __name__ == "__main__":
    unittest.main()
