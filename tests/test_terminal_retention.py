"""Guard endpoint retention, interrupted episodes and fixed-vector semantics."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_bca_ip_terminal_retention import terminal_episode, combine_units, retained_curve
from read_bca_ip_fsm_outputs import decode_terminal, score_unit

X = [1, 1, 0, 1, 0, 1]
Y = [1, 1, 0, 1, 1, 0]


def regime(start, end, optimal=True, vector=X):
    return dict(start=start, end=end, optimal=optimal, vector=vector)


def test_optimum_lost_before_the_endpoint_does_not_count():
    r = terminal_episode([regime(0, 200000), regime(200000, 300000, False)], 300000, 100000)
    assert not r["qualified"] and r["start"] is None


def test_changed_flow_with_the_same_solution_does_not_reset_duration():
    r = terminal_episode([regime(0, 50000, False), regime(50000, 150000),
                          regime(150000, 200000)], 200000, 100000)
    assert r["qualified"] and r["start"] == 50000 and r["merged_regimes"] == 2


@pytest.mark.parametrize("gap_vector", [[1, 0, 0, 0, 0, 0], [None] * 6])
def test_nonoptimal_or_unreadable_gap_breaks_persistence(gap_vector):
    r = terminal_episode([regime(0, 200000), regime(200000, 250000, False, gap_vector),
                          regime(250000, 300000)], 300000, 100000)
    assert not r["qualified"] and r["start"] == 250000


def test_switching_between_distinct_optima_starts_a_new_solution_episode():
    r = terminal_episode([regime(0, 250000), regime(250000, 300000, vector=Y)], 300000, 100000)
    assert not r["qualified"] and r["vector"] == Y


@pytest.mark.parametrize("start,expected", [(200000, True), (200001, False)])
def test_exact_100k_inclusive_threshold(start, expected):
    r = terminal_episode([regime(0, start, False), regime(start, 300000)], 300000, 100000)
    assert r["qualified"] == expected


def test_two_units_count_once_and_cannot_bridge_a_short_tail():
    a = terminal_episode([regime(0, 150000, False), regime(150000, 300000)], 300000, 100000)
    b = terminal_episode([regime(0, 100000, False), regime(100000, 300000)], 300000, 100000)
    assert combine_units(a, b)["start"] == 100000
    assert combine_units(a, b)["supporting_units"] == ["A", "B"]
    lost = terminal_episode([regime(0, 250000), regime(250000, 300000, False)], 300000, 100000)
    recent = terminal_episode([regime(0, 250000, False), regime(250000, 300000)], 300000, 100000)
    assert not combine_units(lost, recent)["qualified"]


def test_curve_denominator_includes_every_failed_or_unresolved_trial():
    curve = retained_curve([50000, 200000, None, None], trials=4, horizon=300000,
                           minimum_duration=100000, step=50000)
    assert [r["trials"] for r in curve] == [0, 1, 1, 1, 2, 2, 2]
    assert curve[-1]["fraction"] == .5


def test_whole_regime_does_not_hide_an_ambiguous_early_output():
    wires = [np.arange(2500, 600001, 5000) if bit else np.array([], dtype=int) for bit in X]
    wires[2] = np.arange(2500, 200001, 5000)
    instance = dict(a=[1, 1, 2, 2, 4, 4], b=8, c=[2, 2, 1, 5, 5, 5])
    old = score_unit(decode_terminal(wires, 0, 600000, max_window=300000), instance, 14)
    whole = score_unit(decode_terminal(wires, 0, 600000, max_window=600000), instance, 14)
    assert old["optimal"] and not whole["optimal"]


def test_discontinuous_history_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        terminal_episode([regime(0, 100000), regime(110000, 300000)], 300000, 100000)
