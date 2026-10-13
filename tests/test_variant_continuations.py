"""Check real restart fixtures and reject changed counters or identities."""
import copy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import continue_bca_ip_variants as continuation
from PyBCA.api import Engine
from PyBCA.api.streaming import sha256
from test_core_runtime import config


def test_fixture_matches_uninterrupted_run_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    initial = config(steps=7, stream_dir=str(source), flush_interval=3, checkpoint_interval=6)
    Engine(initial).run()
    checkpoint = source / "checkpoint.pt"
    before = sha256(checkpoint)
    full, split = tmp_path / "full", tmp_path / "split"
    for directory in (full, split):
        assert continuation.verification_checkpoint(checkpoint, directory) == 7
    target = replace(initial, steps=17, stream_dir=str(full), resume_from=str(full / "checkpoint.pt"))
    a = continuation.snapshot_run(target)
    split_config = replace(target, stream_dir=str(split), resume_from=str(split / "checkpoint.pt"))
    continuation.snapshot_run(replace(split_config, steps=11))
    b = continuation.snapshot_run(split_config)
    oracle = Engine(replace(initial, steps=17, stream_dir=None)).run().simulator
    assert torch.equal(a["cells"], b["cells"])
    assert torch.equal(a["cells"], oracle.TCHW.cpu())
    assert a["step"] == b["step"] == 17
    assert continuation.records(full) == continuation.records(split)
    assert all(row[2] >= 7 for row in continuation.records(full))
    assert sha256(checkpoint) == before
    assert torch.load(checkpoint, weights_only=True)["manifest"]["chunks"]
    with pytest.raises(FileExistsError):
        continuation.verification_checkpoint(checkpoint, full)


@pytest.mark.parametrize("change", ["seed", "trial_ids", "probability", "offset", "low_step", "high_step", "history_step"])
def test_refuse_incompatible_restart(tmp_path, change):
    cfg = config(steps=3, stream_dir=str(tmp_path), flush_interval=2)
    Engine(cfg).run()
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert continuation.validate_restart(state, cfg, minimum=3, maximum=10) == 3
    state = copy.deepcopy(state)
    if change == "seed":
        cfg = replace(cfg, seed=32)
    elif change == "trial_ids":
        cfg = replace(cfg, trial_ids=(6, 7, 8))
    elif change == "probability":
        cfg = replace(cfg, global_prob=.25)
    elif change == "offset":
        cfg = replace(cfg, trial_offset=64)
    elif change == "low_step":
        state["next_step"] = 2
    elif change == "high_step":
        state["next_step"] = 11
    elif change == "history_step":
        state["manifest"]["next_step"] = 4
    with pytest.raises(ValueError):
        continuation.validate_restart(state, cfg, minimum=3, maximum=10)
