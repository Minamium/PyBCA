"""A growing history must never leak future events into a fixed endpoint."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from snapshot_bca_ip_readout import history_prefix


def manifest():
    return {"next_step": 3000, "trial_ids": [0, 1], "identity": {"seed": 17},
            "chunks": [{"start_step": i, "next_step": i + 1000,
                        "path": f"history/{i}.jsonl", "sha256": str(i)}
                       for i in range(0, 3000, 1000)]}


def test_snapshot_keeps_exact_prefix_and_does_not_mutate_source():
    source = manifest()
    before = copy.deepcopy(source)
    frozen = history_prefix(source, 2000)
    assert source == before
    assert frozen["next_step"] == 2000 and len(frozen["chunks"]) == 2
    source["chunks"][0]["sha256"] = "replaced"
    assert frozen["chunks"][0]["sha256"] == "0"
    assert frozen["identity"] == {"seed": 17}


@pytest.mark.parametrize("horizon", [0, 1500, 4000])
def test_uncommitted_or_partial_chunk_endpoint_is_rejected(horizon):
    with pytest.raises(ValueError):
        history_prefix(manifest(), horizon)


def test_gapped_history_is_rejected():
    source = manifest()
    source["chunks"][1]["start_step"] = 1001
    with pytest.raises(ValueError, match="gap"):
        history_prefix(source, 2000)
