from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "src").resolve()))

from PyBCA import Config, Engine


def sample_path(*parts: str) -> str:
    return str((ROOT / "Sample" / Path(*parts)).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="torchrun smoke test for Engine")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    event_history_path = outdir / "join_smoke.jsonl"
    run_dir = outdir / "dist"

    config = Config(
        cellspace_path=sample_path("Cellspace", "Join_err", "P0_join.yaml"),
        rule_paths=(
            sample_path("rule", "base-rule.yaml"),
            sample_path("rule", "Join_fork.yaml"),
        ),
        spatial_event_file_path=sample_path("Specialevent", "Join_detect.py"),
        device="cpu",
        trials=args.trials,
        steps=args.steps,
        global_prob=0.5,
        seed=7,
        use_tqdm="false",
        state_gate_enable=True,
        state_gate_interval=10,
        event_history_path=str(event_history_path),
        event_history_format="jsonl_trials",
        event_history_deduplicate=True,
        event_history_return_df=False,
        trial_constant_sweep={
            "join_err_0_input": {"base": 0.1, "delta": 0.05},
            "join_err_1_input": {"base": 0.2, "delta": 0.02},
        },
        distributed_mode="torchrun",
        distributed_backend="gloo",
        distributed_run_dir=str(run_dir),
        distributed_record_configs=True,
        distributed_merge_event_history=True,
    )

    result = Engine(config).run()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        manifest_path = run_dir / "run_manifest.json"
        assert manifest_path.exists(), manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["launcher"]["world_size"] == world_size
        assert manifest["job"]["global_trials"] == args.trials

        q, r = divmod(args.trials, world_size)
        expected_local_trials = [q + (1 if rr < r else 0) for rr in range(world_size)]
        expected_offsets = [rr * q + min(rr, r) for rr in range(world_size)]

        for rr in range(world_size):
            rank_config_path = run_dir / "rank_configs" / f"rank_{rr:04d}.json"
            assert rank_config_path.exists(), rank_config_path
            payload = json.loads(rank_config_path.read_text(encoding="utf-8"))
            part = payload["partition"]
            assert part["local_trials"] == expected_local_trials[rr]
            assert part["trial_offset"] == expected_offsets[rr]

            if expected_local_trials[rr] > 0:
                cfg = payload["config"]
                assert cfg["device"] == "cpu"
                assert cfg["trials"] == expected_local_trials[rr]
                sweep = cfg["trial_constant_sweep"]
                assert sweep["join_err_0_input"]["base"] == 0.1 + (0.05 * expected_offsets[rr])
                assert sweep["join_err_1_input"]["base"] == 0.2 + (0.02 * expected_offsets[rr])

        assert event_history_path.exists(), event_history_path
        merged_lines = event_history_path.read_text(encoding="utf-8").splitlines()
        assert len(merged_lines) == args.trials + 1

        meta_record = json.loads(merged_lines[0])
        meta = meta_record["__meta__"]
        assert meta["parallel_trial"] == args.trials
        assert meta["distributed"]["world_size"] == world_size
        assert meta["distributed"]["rank_shards"][0]["trial_offset"] == 0

        trial_ids = [json.loads(line)["trial"] for line in merged_lines[1:]]
        assert trial_ids == list(range(args.trials)), trial_ids

        merged_result_path = result.meta["distributed"]["paths"]["event_history_merged_path"]
        assert merged_result_path == str(event_history_path)
        assert result.event_history == str(event_history_path)
    else:
        dist_meta = result.meta["distributed"]
        assert dist_meta["context"]["rank"] == rank
        assert dist_meta["context"]["local_rank"] == local_rank
        if dist_meta["partition"]["local_trials"] > 0:
            shard_path = dist_meta["paths"]["event_history_shard_path"]
            assert shard_path is not None
            assert Path(shard_path).exists(), shard_path


if __name__ == "__main__":
    main()
