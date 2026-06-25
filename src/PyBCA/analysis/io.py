"""I/O utilities for PyBCA event-history JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_trials_with_meta(filepath: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load ``jsonl_trials`` output and split the optional metadata record."""
    meta: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []

    with Path(filepath).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue

            if "__meta__" in rec and meta is None:
                value = rec.get("__meta__")
                meta = value if isinstance(value, dict) else None
                continue

            if rec.get("type") == "meta" and meta is None:
                value = rec.get("meta")
                meta = value if isinstance(value, dict) else None
                continue

            trials.append(rec)

    return meta, trials


def events_to_dict(events: Any) -> dict[str, list[int]]:
    """Normalize an event-history payload to ``{event_name: [step, ...]}``."""
    if isinstance(events, dict):
        return {str(name): [int(step) for step in (steps or [])] for name, steps in events.items()}

    if isinstance(events, list):
        out: dict[str, list[int]] = {}
        for item in events:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            name, steps = item
            out[str(name)] = [int(step) for step in (steps or [])]
        return out

    raise ValueError("Unsupported events format. Expected dict or list of [name, steps] pairs.")


def event_history_for_trial(filepath: str | Path, trial: int = 0) -> dict[str, list[int]]:
    """Load one trial's event history from a PyBCA JSONL event-history file."""
    _meta, trials = load_jsonl_trials_with_meta(filepath)
    if not trials:
        raise ValueError(f"No trial records found in {filepath}.")

    trial_map: dict[int, dict[str, Any]] = {}
    has_trial_ids = True
    for rec in trials:
        if "trial" not in rec:
            has_trial_ids = False
            break
        try:
            trial_map[int(rec["trial"])] = rec
        except Exception:
            has_trial_ids = False
            break

    if has_trial_ids:
        if trial not in trial_map:
            raise IndexError(f"trial={trial} was not found in {filepath}.")
        trial_data = trial_map[trial]
    else:
        if trial < 0 or trial >= len(trials):
            raise IndexError(f"trial={trial} is out of range (0..{len(trials) - 1}).")
        trial_data = trials[trial]

    return events_to_dict(trial_data.get("events", {}))


def format_sweep_for_trial(meta: dict[str, Any] | None, trial: int) -> str | None:
    """Return a compact sweep-probability label for a trial, if metadata has it."""
    if not isinstance(meta, dict):
        return None

    sweep = meta.get("probability_sweep") or meta.get("trial_constant_sweep")
    if not isinstance(sweep, dict):
        return None

    parts: list[str] = []
    for alias, info in sweep.items():
        if not isinstance(info, dict):
            continue
        probs = info.get("prob_by_trial")
        if isinstance(probs, list) and 0 <= trial < len(probs):
            try:
                parts.append(f"{alias}={float(probs[trial]):.6g}")
            except Exception:
                continue
        elif "base" in info and "delta" in info:
            try:
                parts.append(f"{alias}={float(info['base']) + float(info['delta']) * trial:.6g}")
            except Exception:
                continue

    if not parts:
        return None
    return "sweep: " + ", ".join(parts)
