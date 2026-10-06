"""Parameter and source-preservation checks for the paper cellspace variants.

The longer pulse tests are run by scripts/validate_bca_ip_cellspaces.py and
their results are checked in beside the maps.
"""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_bca_ip_cellspaces as gen
import validate_bca_ip_cellspaces as validate


@pytest.fixture(scope="module")
def maps():
    return {name: gen.read_map(gen.DEFAULT_OUTPUT / name) for name in (
        "BCA-IP_condition1_N1.yaml", "BCA-IP_condition1_N2.yaml", "BCA-IP_condition2_N2.yaml")}


@pytest.mark.parametrize("name,weights,b", [
    ("BCA-IP_condition1_N1.yaml", [4, 3, 2, 1, 5, 5], 6),
    ("BCA-IP_condition1_N2.yaml", [7, 5, 3, 1, 9, 9], 6),
    ("BCA-IP_condition2_N2.yaml", [7, 7, 9, 1, 1, 1], 8),
])
def test_both_units_encode_the_requested_problem(maps, name, weights, b):
    cells = maps[name]
    for i, offset in enumerate(gen.CORE_OFFSETS):
        tokens = [x for x in range(-61, -52) if cells.get((x, offset - 2), 0) == 2]
        assert tokens == list(range(-52 - weights[i % 6], -52))
    for y in (217, 712):
        assert sum(cells.get((gen.shifted_x(x), y), 0) == 2 for x in range(-280, -258)) == b


def test_N_does_not_modify_objective_amplifiers_or_TD(maps):
    a = maps["BCA-IP_condition1_N1.yaml"]
    b = maps["BCA-IP_condition1_N2.yaml"]
    allowed = {(x, o - 2) for o in gen.CORE_OFFSETS for x in range(-61, -52)}
    allowed |= {(x, o + y) for o in gen.CORE_OFFSETS
                for y in range(3, 43) for x in range(-113, -40)}
    changed = {p for p in a.keys() | b.keys() if a.get(p, 0) != b.get(p, 0)}
    assert changed
    assert not changed - allowed


def test_complete_map_audit_including_events():
    report, fixtures = validate.static_audit(gen.DEFAULT_OUTPUT)
    assert set(fixtures) == {1, 2, 3, 4, 5, 7, 9}
    assert len(report["variants"]) == 3
    assert report["legacy_condition2_initial_weights"] == [4, 4, 5, 1, 1, 1]
    assert report["legacy_reset_ladder_gains_by_stage_count"] == [4, 3, 2, 1, 5, 5]


def test_source_patch_rejects_wrong_source():
    patch = {"changes": [{"x": 0, "y": 0, "before": 2, "after": 1}]}
    with pytest.raises(ValueError, match="Source patch does not match"):
        gen.condition1_source({(0, 0): 1}, patch)


def test_widening_rejects_active_component_in_seam():
    with pytest.raises(ValueError, match="Unexpected component"):
        gen.widen({(-65, -13): 2})


def test_duplicate_coordinates_are_not_silently_discarded(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("- coord: {x: 0, y: 0}\n  value: 1\n- coord: {x: 0, y: 0}\n  value: 2\n")
    with pytest.raises(ValueError, match="Duplicate coordinate"):
        gen.read_map(path)


def test_original_cellspace_still_matches_archived_hash():
    patch = json.loads(gen.PATCH.read_text())
    assert gen.sha256(gen.BASE) == patch["base_sha256"]


def test_widening_seam_preserves_wire_phase_outside_replaced_amps():
    cells = gen.read_map(gen.BASE)
    replaced_rows = {o + y for o in gen.CORE_OFFSETS for y in range(3, 43)}
    for y in range(-16, 895):
        if y not in replaced_rows:
            # These columns meet at the repeated tile and its right boundary.
            for left, right in ((-66, -63), (-65, -62), (-64, -61)):
                # Existing tokens move with their original cells; the inserted
                # wire is empty. Compare track geometry, not token occupancy.
                assert min(1, cells.get((left, y), 0)) == min(1, cells.get((right, y), 0)), (left, right, y)


def test_readout_problem_files_have_the_intended_optima():
    from read_bca_ip_fsm_outputs import solve_binary_instance

    for condition, expected in ((1, {"111100"}), (2, {"110101", "110110"})):
        problem = json.loads((gen.DEFAULT_OUTPUT / f"condition{condition}.json").read_text())
        value, vectors = solve_binary_instance(problem)
        assert value == problem["optimum_value"] == 14
        assert {"".join(map(str, x)) for x in vectors} == expected
