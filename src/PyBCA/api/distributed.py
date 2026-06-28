from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config, UseTqdm

logger = logging.getLogger("PyBCA")


@dataclass
class DistributedContext:
    enabled: bool = False
    mode: str = "off"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_master: bool = True
    backend: str | None = None
    process_group_initialized: bool = False


@dataclass(frozen=True)
class TrialPartition:
    rank: int
    world_size: int
    global_trials: int
    local_trials: int
    trial_offset: int

    @property
    def trial_end(self) -> int:
        return self.trial_offset + self.local_trials

    @property
    def active(self) -> bool:
        return self.local_trials > 0


@dataclass
class DistributedRunState:
    context: DistributedContext
    original_config: Config
    local_config: Config | None
    partition: TrialPartition
    run_dir: str | None = None
    manifest_path: str | None = None
    rank_config_path: str | None = None
    shard_event_history_path: str | None = None
    merged_event_history_path: str | None = None
    shard_rule_history_path: str | None = None
    merged_rule_history_path: str | None = None

    @property
    def active(self) -> bool:
        return self.partition.active and self.local_config is not None


def resolve_distributed_context(config: Config) -> DistributedContext:
    env = os.environ
    has_torchrun_env = all(key in env for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    mode = config.distributed_mode

    if mode == "off":
        return DistributedContext()

    if mode == "auto" and not has_torchrun_env:
        return DistributedContext()

    if not has_torchrun_env:
        raise ValueError(
            "distributed_mode='torchrun' requires torchrun environment variables "
            "(RANK, LOCAL_RANK, WORLD_SIZE)."
        )

    rank = int(env["RANK"])
    local_rank = int(env["LOCAL_RANK"])
    world_size = int(env["WORLD_SIZE"])
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be >= 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE). got rank={rank}, world_size={world_size}")

    return DistributedContext(
        enabled=True,
        mode="torchrun",
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_master=(rank == 0),
    )


def initialize_process_group(context: DistributedContext, config: Config) -> None:
    if not context.enabled or context.world_size <= 1:
        return

    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        context.backend = dist.get_backend()
        context.process_group_initialized = True
        return

    resolved_device = resolve_device(config.device, context)
    if resolved_device.startswith("cuda"):
        torch.cuda.set_device(context.local_rank)

    backend = config.distributed_backend
    if backend == "auto":
        backend = "nccl" if resolved_device.startswith("cuda") else "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    context.backend = backend
    context.process_group_initialized = True


def barrier(context: DistributedContext) -> None:
    if not context.process_group_initialized:
        return

    import torch.distributed as dist

    dist.barrier()


def shutdown_process_group(context: DistributedContext) -> None:
    if not context.process_group_initialized:
        return

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    context.process_group_initialized = False


def partition_trials(global_trials: int, rank: int, world_size: int) -> TrialPartition:
    q, r = divmod(int(global_trials), int(world_size))
    local_trials = q + (1 if rank < r else 0)
    trial_offset = rank * q + min(rank, r)
    return TrialPartition(
        rank=rank,
        world_size=world_size,
        global_trials=int(global_trials),
        local_trials=int(local_trials),
        trial_offset=int(trial_offset),
    )


def all_trial_partitions(global_trials: int, world_size: int) -> list[TrialPartition]:
    return [partition_trials(global_trials, rank, world_size) for rank in range(world_size)]


def resolve_device(device: str, context: DistributedContext) -> str:
    text = str(device).strip()
    if not context.enabled:
        return text
    if text.lower().startswith("cuda"):
        return f"cuda:{context.local_rank}"
    return text


def resolve_seed(seed: int, config: Config, context: DistributedContext) -> int:
    if not context.enabled:
        return int(seed)
    return int(seed) + (context.rank * int(config.distributed_seed_stride))


def shift_trial_constant_sweep(
    trial_constant_sweep: dict[str, dict[str, float]] | None,
    trial_offset: int,
) -> dict[str, dict[str, float]] | None:
    if trial_constant_sweep is None:
        return None

    shifted = copy.deepcopy(trial_constant_sweep)
    for _alias, cfg in shifted.items():
        base = float(cfg.get("base", 0.0))
        delta = float(cfg.get("delta", 0.0))
        cfg["base"] = float(base + (delta * int(trial_offset)))
        cfg["delta"] = float(delta)
    return shifted


def prepare_distributed_run(config: Config) -> DistributedRunState:
    context = resolve_distributed_context(config)
    initialize_process_group(context, config)

    partition = partition_trials(config.trials, context.rank, context.world_size)
    run_dir = resolve_run_dir(config) if context.enabled else None
    manifest_path = str(Path(run_dir) / "run_manifest.json") if run_dir is not None else None
    rank_config_path = (
        str(Path(run_dir) / "rank_configs" / f"rank_{context.rank:04d}.json")
        if run_dir is not None
        else None
    )
    shard_event_history_path = (
        resolve_shard_event_history_path(config.event_history_path, run_dir, context.rank)
        if context.enabled and partition.active
        else None
    )
    shard_rule_history_path = (
        resolve_shard_rule_history_path(config.rule_history_path, run_dir, context.rank)
        if context.enabled and partition.active
        else None
    )

    local_config: Config | None = None
    if partition.active:
        local_config = replace(
            config,
            device=resolve_device(config.device, context),
            trials=partition.local_trials,
            seed=resolve_seed(config.seed, config, context),
            use_tqdm=config.use_tqdm if context.is_master else UseTqdm.FALSE,
            trial_constant_sweep=shift_trial_constant_sweep(
                config.trial_constant_sweep,
                partition.trial_offset,
            ),
            event_history_path=shard_event_history_path if shard_event_history_path is not None else config.event_history_path,
            rule_history_path=shard_rule_history_path if shard_rule_history_path is not None else config.rule_history_path,
        )

    state = DistributedRunState(
        context=context,
        original_config=config,
        local_config=local_config,
        partition=partition,
        run_dir=run_dir,
        manifest_path=manifest_path,
        rank_config_path=rank_config_path,
        shard_event_history_path=shard_event_history_path,
        merged_event_history_path=config.event_history_path if context.enabled else config.event_history_path,
        shard_rule_history_path=shard_rule_history_path,
        merged_rule_history_path=config.rule_history_path if context.enabled else config.rule_history_path,
    )

    if context.enabled and run_dir is not None and config.distributed_record_configs:
        write_distributed_json_records(state)

    return state


def build_local_save_kwargs(state: DistributedRunState) -> dict[str, Any]:
    if not state.context.enabled:
        return {}
    return {
        "trial_index_offset": state.partition.trial_offset,
        "extra_meta": {
            "distributed": {
                "enabled": True,
                "mode": state.context.mode,
                "backend": state.context.backend,
                "rank": state.context.rank,
                "local_rank": state.context.local_rank,
                "world_size": state.context.world_size,
                "partition": state.original_config.distributed_partition,
                "global_trials": state.original_config.trials,
                "local_trials": state.partition.local_trials,
                "trial_offset": state.partition.trial_offset,
                "trial_end": state.partition.trial_end,
                "global_seed": state.original_config.seed,
                "resolved_seed": state.local_config.seed if state.local_config is not None else None,
                "seed_stride": state.original_config.distributed_seed_stride,
                "event_history_final_path": state.original_config.event_history_path,
                "event_history_shard_path": state.shard_event_history_path,
                "rule_history_final_path": state.original_config.rule_history_path,
                "rule_history_shard_path": state.shard_rule_history_path,
            }
        },
    }


def merge_event_history_shards(state: DistributedRunState) -> str | None:
    if not state.context.enabled:
        return state.original_config.event_history_path
    if not state.context.is_master:
        return None
    if state.original_config.event_history_path is None:
        return None
    if not state.original_config.distributed_merge_event_history:
        return None

    partitions = all_trial_partitions(state.original_config.trials, state.context.world_size)
    active_partitions = [part for part in partitions if part.active]
    if not active_partitions:
        return None

    shard_paths = [
        resolve_shard_event_history_path(state.original_config.event_history_path, state.run_dir, part.rank)
        for part in active_partitions
    ]
    fmt = state.original_config.event_history_format.lower()
    final_path = state.original_config.event_history_path
    meta = build_merged_event_history_meta(state, shard_paths, active_partitions)

    if fmt in ("jsonl_trials", "jsonl_trials_dict"):
        merge_jsonl_trial_shards(final_path, shard_paths, meta)
    elif fmt == "jsonl":
        merge_jsonl_flat_shards(final_path, shard_paths, meta)
    elif fmt == "csv":
        merge_csv_shards(final_path, shard_paths, meta)
    elif fmt == "yaml":
        merge_yaml_shards(final_path, shard_paths, meta)
    elif fmt == "parquet":
        merge_parquet_shards(final_path, shard_paths, meta, state.original_config.event_history_deduplicate)
    else:
        raise ValueError(f"unsupported distributed event_history_format: {state.original_config.event_history_format}")

    return final_path


def merge_rule_history_shards(state: DistributedRunState) -> str | None:
    if not state.context.enabled:
        return state.original_config.rule_history_path
    if not state.context.is_master:
        return None
    if state.original_config.rule_history_path is None:
        return None
    if not state.original_config.distributed_merge_event_history:
        return None

    partitions = all_trial_partitions(state.original_config.trials, state.context.world_size)
    active_partitions = [part for part in partitions if part.active]
    if not active_partitions:
        return None

    shard_paths = [
        resolve_shard_rule_history_path(state.original_config.rule_history_path, state.run_dir, part.rank)
        for part in active_partitions
    ]
    fmt = state.original_config.rule_history_format.lower()
    final_path = state.original_config.rule_history_path
    meta = build_merged_rule_history_meta(state, shard_paths, active_partitions)

    if fmt in ("jsonl_trials", "jsonl_trials_dict"):
        merge_jsonl_trial_shards(final_path, shard_paths, meta)
    elif fmt == "jsonl":
        merge_jsonl_flat_shards(final_path, shard_paths, meta)
    elif fmt == "csv":
        merge_csv_shards(final_path, shard_paths, meta)
    elif fmt == "yaml":
        merge_yaml_shards(final_path, shard_paths, meta)
    elif fmt == "parquet":
        merge_parquet_shards(final_path, shard_paths, meta, state.original_config.rule_history_deduplicate)
    else:
        raise ValueError(f"unsupported distributed rule_history_format: {state.original_config.rule_history_format}")

    return final_path


def resolve_run_dir(config: Config) -> str:
    if config.distributed_run_dir:
        return os.path.abspath(config.distributed_run_dir)
    if config.event_history_path:
        base = Path(config.event_history_path)
        return str((base.parent / f"{base.stem}.dist").resolve())
    if config.rule_history_path:
        base = Path(config.rule_history_path)
        return str((base.parent / f"{base.stem}.dist").resolve())
    return str((Path.cwd() / "pybca_distributed_run").resolve())


def resolve_shard_event_history_path(base_path: str | None, run_dir: str | None, rank: int) -> str | None:
    return resolve_shard_history_path(base_path, run_dir, rank, "event_history_shards", "event_history")


def resolve_shard_rule_history_path(base_path: str | None, run_dir: str | None, rank: int) -> str | None:
    return resolve_shard_history_path(base_path, run_dir, rank, "rule_history_shards", "rule_history")


def resolve_shard_history_path(
    base_path: str | None,
    run_dir: str | None,
    rank: int,
    shard_dir_name: str,
    default_stem: str,
) -> str | None:
    if base_path is None or run_dir is None:
        return None

    base = Path(base_path)
    suffix = base.suffix or ".out"
    stem = base.stem if base.stem else default_stem
    shard_dir = Path(run_dir) / shard_dir_name
    return str((shard_dir / f"{stem}.rank{rank:04d}{suffix}").resolve())


def write_distributed_json_records(state: DistributedRunState) -> None:
    if state.run_dir is None or state.rank_config_path is None:
        return

    run_dir = Path(state.run_dir)
    (run_dir / "rank_configs").mkdir(parents=True, exist_ok=True)

    rank_payload = {
        "schema": "pybca.distributed.rank_config.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "active": state.active,
        "context": context_payload(state.context),
        "partition": partition_payload(state.partition),
        "paths": {
            "run_dir": state.run_dir,
            "manifest_path": state.manifest_path,
            "rank_config_path": state.rank_config_path,
            "event_history_final_path": state.original_config.event_history_path,
            "event_history_shard_path": state.shard_event_history_path,
            "rule_history_final_path": state.original_config.rule_history_path,
            "rule_history_shard_path": state.shard_rule_history_path,
        },
        "config": state.local_config.as_dict if state.local_config is not None else None,
    }
    write_json_file(state.rank_config_path, rank_payload)

    if state.context.is_master and state.manifest_path is not None:
        partitions = all_trial_partitions(state.original_config.trials, state.context.world_size)
        manifest_payload = {
            "schema": "pybca.distributed.run.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "launcher": {
                "kind": state.context.mode,
                "backend": state.context.backend if state.context.backend is not None else state.original_config.distributed_backend,
                "world_size": state.context.world_size,
            },
            "job": {
                "partition": state.original_config.distributed_partition,
                "global_trials": state.original_config.trials,
                "merge_event_history": state.original_config.distributed_merge_event_history,
                "run_dir": state.run_dir,
            },
            "base_config": state.original_config.as_dict,
            "partitions": [partition_payload(part) for part in partitions],
            "rank_config_paths": [
                str((Path(state.run_dir) / "rank_configs" / f"rank_{part.rank:04d}.json").resolve())
                for part in partitions
            ],
        }
        write_json_file(state.manifest_path, manifest_payload)


def build_merged_event_history_meta(
    state: DistributedRunState,
    shard_paths: list[str],
    partitions: list[TrialPartition],
) -> dict[str, Any]:
    first_meta = read_meta_from_sidecar(shard_paths[0]) or {}
    meta = copy.deepcopy(first_meta)
    meta["meta_version"] = 1
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    meta["parallel_trial"] = int(state.original_config.trials)
    meta["device"] = str(state.original_config.device)
    meta["distributed"] = {
        "enabled": True,
        "mode": state.context.mode,
        "backend": state.context.backend,
        "world_size": state.context.world_size,
        "partition": state.original_config.distributed_partition,
        "global_trials": state.original_config.trials,
        "seed_stride": state.original_config.distributed_seed_stride,
        "rank_shards": [
            {
                "rank": part.rank,
                "local_trials": part.local_trials,
                "trial_offset": part.trial_offset,
                "trial_end": part.trial_end,
                "path": resolve_shard_event_history_path(state.original_config.event_history_path, state.run_dir, part.rank),
            }
            for part in partitions
        ],
    }

    if state.original_config.trial_constant_sweep is None:
        meta["probability_sweep"] = None
        return meta

    local_sweep_meta = first_meta.get("probability_sweep") or {}
    sweep_out: dict[str, Any] = {}
    for alias, cfg in state.original_config.trial_constant_sweep.items():
        alias_meta = local_sweep_meta.get(alias, {})
        base = float(cfg.get("base", 0.0))
        delta = float(cfg.get("delta", 0.0))
        sweep_out[str(alias)] = {
            "base": base,
            "delta": delta,
            "prob_by_trial": [clamp01(base + (tt * delta)) for tt in range(state.original_config.trials)],
            "rule_ids": alias_meta.get("rule_ids"),
            "rule_indices": alias_meta.get("rule_indices"),
        }
    meta["probability_sweep"] = sweep_out
    return meta


def build_merged_rule_history_meta(
    state: DistributedRunState,
    shard_paths: list[str],
    partitions: list[TrialPartition],
) -> dict[str, Any]:
    meta = build_merged_event_history_meta(state, shard_paths, partitions)
    meta["history_kind"] = "rule_history"
    meta["distributed"]["rank_shards"] = [
        {
            "rank": part.rank,
            "local_trials": part.local_trials,
            "trial_offset": part.trial_offset,
            "trial_end": part.trial_end,
            "path": resolve_shard_rule_history_path(state.original_config.rule_history_path, state.run_dir, part.rank),
        }
        for part in partitions
    ]
    return meta


def context_payload(context: DistributedContext) -> dict[str, Any]:
    return {
        "enabled": context.enabled,
        "mode": context.mode,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": context.world_size,
        "is_master": context.is_master,
        "backend": context.backend,
        "process_group_initialized": context.process_group_initialized,
    }


def partition_payload(partition: TrialPartition) -> dict[str, int]:
    return {
        "rank": partition.rank,
        "world_size": partition.world_size,
        "global_trials": partition.global_trials,
        "local_trials": partition.local_trials,
        "trial_offset": partition.trial_offset,
        "trial_end": partition.trial_end,
    }


def read_meta_from_sidecar(path: str) -> dict[str, Any] | None:
    meta_path = sidecar_meta_path(path)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sidecar_meta_path(path: str) -> str:
    root, _ext = os.path.splitext(path)
    return root + "_meta.json"


def write_json_file(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_meta_sidecar(path: str, meta: dict[str, Any]) -> None:
    write_json_file(sidecar_meta_path(path), meta)


def merge_jsonl_trial_shards(final_path: str, shard_paths: list[str], meta: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    with open(final_path, "w", encoding="utf-8") as out:
        out.write(json.dumps({"__meta__": meta}, ensure_ascii=False) + "\n")
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as src:
                first_record = True
                for line in src:
                    if first_record:
                        first_record = False
                        try:
                            rec = json.loads(line)
                        except Exception:
                            rec = None
                        if isinstance(rec, dict) and "__meta__" in rec:
                            continue
                    if line.strip():
                        out.write(line)
    write_meta_sidecar(final_path, meta)


def merge_jsonl_flat_shards(final_path: str, shard_paths: list[str], meta: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    with open(final_path, "w", encoding="utf-8") as out:
        out.write(json.dumps({"__meta__": meta}, ensure_ascii=False) + "\n")
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as src:
                first_record = True
                for line in src:
                    if first_record:
                        first_record = False
                        try:
                            rec = json.loads(line)
                        except Exception:
                            rec = None
                        if isinstance(rec, dict) and "__meta__" in rec:
                            continue
                    if line.strip():
                        out.write(line)
    write_meta_sidecar(final_path, meta)


def merge_csv_shards(final_path: str, shard_paths: list[str], meta: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    meta_lines = json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True).splitlines()
    with open(final_path, "w", encoding="utf-8") as out:
        for line in meta_lines:
            out.write("# " + line + "\n")

        header_written = False
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as src:
                header_skipped = False
                for line in src:
                    if line.startswith("#"):
                        continue
                    if not header_written:
                        out.write(line)
                        header_written = True
                        header_skipped = True
                        continue
                    if not header_skipped:
                        header_skipped = True
                        continue
                    out.write(line)
    write_meta_sidecar(final_path, meta)


def merge_yaml_shards(final_path: str, shard_paths: list[str], meta: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("YAML merge requires PyYAML.") from e

    records: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        with open(shard_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or []
            if isinstance(loaded, list):
                records.extend(loaded)

    meta_lines = json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True).splitlines()
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    with open(final_path, "w", encoding="utf-8") as out:
        for line in meta_lines:
            out.write("# " + line + "\n")
        yaml.safe_dump(records, out, allow_unicode=True, sort_keys=False)
    write_meta_sidecar(final_path, meta)


def merge_parquet_shards(
    final_path: str,
    shard_paths: list[str],
    meta: dict[str, Any],
    deduplicate: bool,
) -> None:
    import pandas as pd

    frames = [pd.read_parquet(path) for path in shard_paths]
    if len(frames) == 0:
        merged = pd.DataFrame(columns=["trial", "event", "step"])
    else:
        merged = pd.concat(frames, ignore_index=True)
        if deduplicate and not merged.empty:
            merged = merged.drop_duplicates(subset=["trial", "event", "step"])
        if not merged.empty:
            merged = merged.sort_values(["trial", "event", "step"], kind="mergesort").reset_index(drop=True)

    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    merged.to_parquet(final_path, index=False, compression="snappy")
    write_meta_sidecar(final_path, meta)


def clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
