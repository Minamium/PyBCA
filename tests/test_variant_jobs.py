"""Protect trial identity and completion gating for costly production jobs."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_bca_ip_variants as checks
from PyBCA.api.streaming import identity


def record(rank=3):
    trial_ids = list(range(rank * 64, (rank + 1) * 64))
    config = checks.config_for("condition1_N2", trials=64, steps=3000000,
                              seed=checks.SEED, trial_ids=trial_ids)
    ident = identity(config)
    ident["trial_offset"] = rank * 64
    return ({"current_step": 3000000, "target_step": 3000000, "stopped": False,
             "rank": rank, "active": True, "trial_ids": trial_ids},
            {"next_step": 3000000, "identity": ident})


def validate(summary, manifest):
    checks.validate_summary(summary, manifest, variant="condition1_N2", rank=3,
                            steps=3000000, seed=checks.SEED)


def test_valid_rank_and_counter_identity():
    validate(*record())


@pytest.mark.parametrize("mutation", ["interrupted", "short", "wrong_seed", "wrong_probability",
                                    "wrong_trial", "wrong_cellspace", "old_events"])
def test_refuses_incomplete_or_mixed_experiment(mutation):
    summary, manifest = copy.deepcopy(record())
    if mutation == "interrupted":
        summary["stopped"] = True
    elif mutation == "short":
        summary["current_step"] -= 1
    elif mutation == "wrong_seed":
        manifest["identity"]["seed"] = checks.PILOT_SEED
    elif mutation == "wrong_probability":
        manifest["identity"]["global_prob"] = 1.0
    elif mutation == "wrong_trial":
        summary["trial_ids"][0] = 0
    elif mutation == "wrong_cellspace":
        manifest["identity"]["inputs"]["cellspace"] = "old"
    elif mutation == "old_events":
        manifest["identity"]["inputs"]["events"] = "old"
    with pytest.raises(ValueError):
        validate(summary, manifest)


def test_event_families_keep_FSM_and_reset_separate():
    assert [checks.event_family(name) for name in ("A_core_input_1", "B_x6output",
            "A_Amp_x2output", "F_value_B", "Reset_Signal_for_A_set",
            "Reset_Signal_for_A_clear")] == ["TD", "FSM", "Amp", "F", "reset", "other"]
