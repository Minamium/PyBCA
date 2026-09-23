"""Bounded history recording and crash-consistent, atomic checkpoints.

History chunks are immutable step ranges. The manifest is authoritative: after
a restart, chunks newer than the checkpoint are ignored and deterministically
rewritten. A checkpoint never references history that has not been fsynced.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import torch

from PyBCA.core.random import RNG_VERSION

SCHEMA = "pybca-stream-v1"


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()
    atomic_write(path, lambda f: f.write(data))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(config):
    keys = ("model", "scheme", "trials", "global_prob", "seed", "rng_mode",
            "trial_constant_sweep", "trial_offset", "record_rule_history", "rule_history_rule_ids",
            "state_gate_enable", "state_gate_interval")
    source = config.as_dict
    result = {key: source[key] for key in keys}
    result["trial_ids"] = list(config.trial_ids or range(config.trials))
    result["inputs"] = {
        "cellspace": sha256(config.cellspace_path),
        "rules": [sha256(p) for p in config.rule_paths],
        "events": sha256(config.spatial_event_file_path) if config.spatial_event_file_path else None,
    }
    package = Path(__file__).parents[1]
    runtime_files = [p for folder in ("core", "api")
                     for p in (package/folder).rglob("*") if p.suffix in {".py", ".cu"}]
    result["implementation"] = {str(p.relative_to(package)): sha256(p) for p in sorted(runtime_files)}
    result["rng_version"] = RNG_VERSION if config.rng_mode == "independent" else "torch-step-seed-legacy"
    if config.rng_mode == "legacy":
        result["torch"] = str(torch.__version__)
        result["device_type"] = torch.device(config.device).type
    return copy.deepcopy(result)


class HistoryWriter:
    def __init__(self, config, simulator):
        import fcntl
        self.config, self.sim = config, simulator
        self.root = Path(config.stream_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.root/".writer.lock", "a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError(f"Another process owns streaming directory {self.root}")
        self.used = 0
        self.next_step = 0
        self.start_step = 0
        self.trial_ids = list(config.trial_ids or range(config.trials))
        self.events = list(simulator.spatial_event_names or [])
        self.rules = [int(r) for r in simulator.rule_ids]
        self.track = [i for i, r in enumerate(self.rules)
                      if config.record_rule_history and (config.rule_history_rule_ids is None or r in config.rule_history_rule_ids)]
        self.event_buffer = torch.zeros((config.flush_interval, config.trials, len(self.events)),
                                        dtype=torch.bool, device=simulator.device)
        self.rule_buffer = (torch.zeros((config.flush_interval, config.trials, len(self.rules)),
                                       dtype=torch.int32, device=simulator.device) if self.track else None)
        self.manifest = {"schema": SCHEMA, "identity": identity(config), "config": copy.deepcopy(config.as_dict),
                         "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                         "cuda": torch.version.cuda},
                         "trial_ids": self.trial_ids, "events": self.events, "rules": self.rules,
                         "next_step": 0, "chunks": []}
        try:
            if config.resume_from:
                self.restore(config.resume_from)
            elif (self.root/"manifest.json").exists() or (self.root/"checkpoint.pt").exists():
                raise FileExistsError(f"Run directory already contains a run: {self.root}; specify resume_from")
            else:
                atomic_json(self.root/"manifest.json", self.manifest)
        except BaseException:
            self.close()
            raise
        # Streaming owns the history; avoid any unbounded Python lists.
        simulator.event_history = None
        simulator.rule_history = None
        simulator.history_recorder = self

    def begin_step(self, step):
        if step != self.next_step:
            raise RuntimeError(f"History expects step {self.next_step}, got {step}")

    def record_events(self, hits, indices=None):
        if indices is None:
            self.event_buffer[self.used].copy_(hits)
        else:
            self.event_buffer[self.used, :, indices] = hits

    def record_rule_counts(self, counts, rule=None):
        if self.rule_buffer is None:
            return
        if rule is None:
            self.rule_buffer[self.used].copy_(counts)
        else:
            self.rule_buffer[self.used, :, rule].copy_(counts)

    def end_step(self):
        self.used += 1
        self.next_step += 1
        if self.used == self.config.flush_interval:
            self.flush()

    def flush(self):
        if not self.used:
            return
        events = self.event_buffer[:self.used].cpu()
        rules = self.rule_buffer[:self.used].cpu() if self.rule_buffer is not None else None
        relative = f"history/{self.start_step:012d}-{self.next_step:012d}.jsonl"
        records = 0

        def write(f):
            nonlocal records
            f.write((json.dumps({"__chunk__": {"schema": SCHEMA, "start_step": self.start_step,
                                              "next_step": self.next_step}}) + "\n").encode())
            for step, trial, event in events.nonzero().tolist():
                row = {"kind": "event", "trial": self.trial_ids[trial], "step": self.start_step+step,
                       "name": self.events[event], "count": 1}
                f.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
                records += 1
            if rules is not None:
                selected = rules[:, :, self.track]
                for step, trial, column in selected.nonzero().tolist():
                    r = self.track[column]
                    row = {"kind": "rule", "trial": self.trial_ids[trial], "step": self.start_step+step,
                           "name": f"rule_{self.rules[r]}", "count": int(rules[step, trial, r])}
                    f.write((json.dumps(row) + "\n").encode())
                    records += 1

        atomic_write(self.root/relative, write)
        memory = {"cuda_allocated_bytes": None, "cuda_peak_allocated_bytes": None, "host_peak_rss_bytes": None}
        if str(self.sim.device).startswith("cuda"):
            memory["cuda_allocated_bytes"] = torch.cuda.memory_allocated(self.sim.device)
            memory["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.sim.device)
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            memory["host_peak_rss_bytes"] = int(rss * (1 if sys.platform == "darwin" else 1024))
        except ImportError:
            pass
        self.manifest["chunks"].append({"path": relative, "start_step": self.start_step,
                                         "next_step": self.next_step, "records": records,
                                         "sha256": sha256(self.root/relative), "memory": memory})
        self.manifest["next_step"] = self.next_step
        atomic_json(self.root/"manifest.json", self.manifest)
        self.start_step = self.next_step
        self.used = 0
        self.event_buffer.zero_()
        if self.rule_buffer is not None:
            self.rule_buffer.zero_()

    def checkpoint(self):
        self.flush()
        s = self.sim
        if s._current_step != self.next_step:
            raise RuntimeError("Checkpoint must be taken between complete simulation steps")
        state = {"schema": SCHEMA, "manifest": copy.deepcopy(self.manifest),
                 "next_step": s._current_step, "cells": s.TCHW.detach().cpu(),
                 "offset": [int(s.offset_x), int(s.offset_y)],
                 "rule_probs": s.rule_probs_tensor.detach().cpu(), "rng": s.rng.get_state()}
        if hasattr(s, "rng_event"):
            state["rng_event"] = s.rng_event.get_state()
        atomic_write(self.root/"checkpoint.pt", lambda f: torch.save(state, f))

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("schema") != SCHEMA or state["manifest"]["identity"] != self.manifest["identity"]:
            raise ValueError("Checkpoint inputs, trial IDs, dynamics, RNG or implementation do not match")
        if list(state["cells"].shape) != list(self.sim.TCHW.shape) or state["offset"] != [self.sim.offset_x, self.sim.offset_y]:
            raise ValueError("Checkpoint grid shape/origin mismatch")
        if state["next_step"] > self.config.steps:
            raise ValueError("On resume, steps is the absolute target and must be >= the checkpoint step")
        source_root = Path(path).resolve().parent
        if source_root != self.root:
            raise ValueError("Resume in the checkpoint's original stream_dir (history chunks are referenced there)")
        manifest = state["manifest"]
        for chunk in manifest["chunks"]:
            if sha256(self.root/chunk["path"]) != chunk["sha256"]:
                raise ValueError(f"Checkpoint history chunk is corrupted: {chunk['path']}")
        self.sim.TCHW.copy_(state["cells"].to(self.sim.device))
        self.sim.rule_probs_tensor = state["rule_probs"].to(self.sim.device)
        # Counter RNG has no Torch generator state. CPU and CUDA generator
        # state formats differ, so restore these only for the legacy mode.
        if self.config.rng_mode == "legacy":
            self.sim.rng.set_state(state["rng"])
        if self.config.rng_mode == "legacy" and "rng_event" in state:
            self.sim.rng_event = torch.Generator(device=self.sim.device)
            self.sim.rng_event.set_state(state["rng_event"])
        self.next_step = self.start_step = self.sim._current_step = int(state["next_step"])
        self.manifest = manifest
        self.manifest["config"] = self.config.as_dict
        # Roll back only the manifest. Unreferenced crash-tail files are ignored.
        atomic_json(self.root/"manifest.json", self.manifest)

    def close(self):
        if getattr(self, "lock", None) is not None:
            self.lock.close()
            self.lock = None


def iter_history(stream_dir, verify=False):
    """Read only committed chunks, yielding count-compressed records lazily."""
    root = Path(stream_dir)
    if not (root/"manifest.json").exists() and (root/"run_manifest.json").exists():
        distributed = json.loads((root/"run_manifest.json").read_text())
        for part in distributed["partitions"]:
            if part["local_trials"]:
                yield from iter_history(root/f"rank_{part['rank']:04d}", verify)
        return
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported history schema")
    for chunk in manifest["chunks"]:
        if verify and sha256(root/chunk["path"]) != chunk["sha256"]:
            raise ValueError(f"Corrupt history chunk: {chunk['path']}")
        with (root/chunk["path"]).open() as f:
            for line in f:
                row = json.loads(line)
                if "__chunk__" not in row:
                    yield row
