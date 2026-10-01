"""Describe retained nonoptimal outputs, resets, and earlier optimum readouts.

Readouts describe output-flow regimes, not exact internal CA states. Constant
vector durations refer to adjacent decoded regimes, with unknowns breaking a
run. They are approximate at the change-point grid resolution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from read_bca_ip_fsm_outputs import (
    decode_terminal, score_unit, sha256, solve_binary_instance, split_wire_times,
)


def same_vector_tail_duration(regimes, horizon):
    if not regimes or not regimes[-1]["stable_readout"]:
        return 0
    vector, duration = regimes[-1]["vector"], 0
    for r in reversed(regimes):
        if not r["stable_readout"] or r["vector"] != vector:
            break
        duration = horizon-r["segment_start"]
    return duration


def can_reach_by_additions(vector, targets):
    return any(all(x <= y for x, y in zip(vector, target)) for target in targets)


def build_dynamics(wires, baseline_rows, terminal_rows, resets, horizon, instance):
    optimum, _ = solve_binary_instance(instance)
    units, all_regimes = [], []
    for trial, row in enumerate(baseline_rows):
        for j, unit in enumerate("AB"):
            idx = trial*2+j
            edges = [0, *row[unit]["change_points"], horizon]
            regimes = []
            for start, end in zip(edges, edges[1:]):
                decoded = decode_terminal(wires[idx*6:(idx+1)*6], start, end,
                                          min_duration=10_000, max_window=300_000)
                decoded = score_unit(decoded, instance, optimum)
                decoded |= {"trial_id": trial, "unit": unit,
                            "segment_start": start, "segment_end": end}
                regimes.append(decoded)
            known = [r for r in regimes if r["stable_readout"]]
            changes = [b["segment_start"] for a, b in zip(known, known[1:]) if a["vector"] != b["vector"]]
            # The baseline boundary fit and exact event-time counts must reproduce
            # the short-duration terminal output, including ambiguous bits.
            for key in ("vector", "counts", "quarter_counts", "objective_value", "stable_readout"):
                assert regimes[-1][key] == terminal_rows[trial][unit][key], (trial, unit, key)
            units.append({"trial_id": trial, "unit": unit,
                          "terminal": terminal_rows[trial][unit],
                          "known_vector_change_regime_starts": changes,
                          "terminal_same_vector_duration": same_vector_tail_duration(regimes, horizon),
                          "reset_steps": resets[(resets[:,0] == trial) & (resets[:,1] == j),2].tolist()})
            all_regimes.extend(regimes)
    return units, all_regimes


def group_characteristics(ids, units, horizon, targets):
    terminal_vectors, best_values, completion = Counter(), Counter(), Counter()
    reset_counts, plateaus, records = [], [], []
    late_changes = late_resets = 0
    for trial in sorted(ids):
        pair = units[2*trial:2*trial+2]
        stable = [u for u in pair if u["terminal"]["stable_readout"]]
        best = max((u["terminal"]["objective_value"] for u in stable), default=None)
        leaders = [u for u in stable if u["terminal"]["objective_value"] == best]
        duration = max((u["terminal_same_vector_duration"] for u in leaders), default=0)
        changes = any(any(t >= horizon-500_000 for t in u["known_vector_change_regime_starts"]) for u in pair)
        all_resets = [t for u in pair for t in u["reset_steps"]]
        reset_counts.append(len(all_resets))
        plateaus.append(duration)
        late_changes += changes
        late_resets += any(t > horizon-500_000 for t in all_resets)
        for u in stable:
            terminal_vectors["".join(map(str, u["terminal"]["vector"]))] += 1
        best_values[str(best)] += 1
        add = [can_reach_by_additions(u["terminal"]["vector"], targets) for u in leaders]
        category = ("no_known_terminal_vector" if not leaders else
                    "best_can_complete_by_adding" if any(add) else "best_requires_removing_a_selected_bit")
        completion[category] += 1
        records.append({"trial_id": trial, "best_readable_terminal_value": best,
                        "best_terminal_vector_same_duration": duration,
                        "best_completion_category": category,
                        "new_known_vector_in_last_500k": bool(changes),
                        "resets": len(all_resets), "A": pair[0], "B": pair[1]})
    summary = {
        "trials": len(ids), "trial_ids": sorted(ids),
        "best_readable_terminal_values": dict(sorted(best_values.items())),
        "terminal_unit_vectors": dict(sorted(terminal_vectors.items())),
        "best_completion_categories": dict(completion),
        "reset_total": sum(reset_counts), "trials_with_reset": sum(n > 0 for n in reset_counts),
        "reset_median": float(np.median(reset_counts)) if reset_counts else None,
        "reset_min_max": [min(reset_counts), max(reset_counts)] if reset_counts else None,
        "trials_with_reset_last_500k": late_resets,
        "trials_with_new_known_vector_last_500k": late_changes,
        "best_terminal_vector_same_for_at_least_1m": sum(d >= 1_000_000 for d in plateaus),
        "best_terminal_vector_same_and_other_changes_late": sum(
            r["best_terminal_vector_same_duration"] >= 1_000_000 and r["new_known_vector_in_last_500k"]
            for r in records),
        "best_terminal_same_duration_median": float(np.median(plateaus)) if plateaus else None,
        "reset_counts": reset_counts, "best_terminal_same_durations": plateaus,
    }
    return summary, records


def raw_reset_context(run, trial, reset_step):
    directory = run/f"rank_{trial//64:04d}"
    manifest = json.loads((directory/"manifest.json").read_text())
    records = []
    for chunk in manifest["chunks"]:
        if chunk["next_step"] <= reset_step-300_000 or chunk["start_step"] >= reset_step:
            continue
        path = directory/chunk["path"]
        if sha256(path) != chunk["sha256"]:
            raise ValueError(f"Changed raw history: {path}")
        for line in path.read_text().splitlines()[1:]:
            row = json.loads(line)
            if row["trial"] == trial:
                records.append(row)
    result = {"trial_id": trial, "reset_step": reset_step, "counts_before": {}}
    for length in (300_000, 100_000, 50_000):
        counts = Counter()
        for row in records:
            if reset_step-length < row["step"]+1 <= reset_step:
                counts[row["name"]] += row["count"]
        result["counts_before"][str(length)] = {name: counts[name] for name in
                ("F_value_A", "F_value_B", "Comparate_A-B", "Comparate_B-A")}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("readout", type=Path)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source, sweep, output = args.readout, args.sweep, args.output
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((source/"summary.json").read_text())
    original = [json.loads(s) for s in (source/"trials.jsonl").read_text().splitlines()]
    comparison = json.loads((sweep/"summary.json").read_text())
    details = json.loads((sweep/"trial-details.json").read_text())
    terminal = details["original_10000"]["terminal"]
    raw = np.load(source/"fsm-event-times.npz")
    wires = split_wire_times(raw["events"], baseline["trials"])
    targets = baseline["optimum_vectors"]
    units, regimes = build_dynamics(wires, original, terminal, raw["resets"],
                                    baseline["horizon"], baseline["instance"])
    final = comparison["tables"]["original"][-1]
    ever = set(final["ever_supported_trial_ids"])
    retained = set(final["terminal_trial_ids"])
    groups = {"never_supported": set(range(baseline["trials"]))-ever,
              "retained": retained, "earlier_only": ever-retained}
    summaries, record_groups = {}, {}
    for name, ids in groups.items():
        summaries[name], record_groups[name] = group_characteristics(ids, units, baseline["horizon"], targets)
    assert sum(s["trials"] for s in summaries.values()) == baseline["trials"]
    # Independently classify all qualifying segments against the sweep's roster.
    assert {r["trial_id"] for r in regimes if r["optimal"]} == ever
    reset_contexts = []
    for trial in sorted(groups["earlier_only"]):
        for j, unit in enumerate("AB"):
            optima = [r for r in regimes if r["trial_id"] == trial and r["unit"] == unit and r["optimal"]]
            if not optima:
                continue
            end = max(r["segment_end"] for r in optima)
            for reset in units[2*trial+j]["reset_steps"]:
                if abs(reset-end) > 10_000:
                    continue
                context = raw_reset_context(Path(baseline["provenance"]["run"]), trial, reset)
                context["reset_unit"] = unit
                context["pre_reset_fsm_readouts"] = {}
                for other_j, other in enumerate("AB"):
                    start = max(t for t in [0, *original[trial][other]["change_points"]] if t < reset)
                    index = 2*trial+other_j
                    decoded = decode_terminal(wires[index*6:(index+1)*6], start, reset,
                                              min_duration=10_000, max_window=300_000)
                    context["pre_reset_fsm_readouts"][other] = score_unit(decoded, baseline["instance"],
                                                                          baseline["optimum_value"])
                reset_contexts.append(context)
    result = {
        "schema": "pybca-nonhit-output-dynamics-v1", "horizon": baseline["horizon"],
        "minimum_duration": 10_000, "groups": summaries,
        "notes": ["Best value uses readable terminal units; a companion unit may be unresolved.",
                  "Durations and pattern changes use retrospective 10,000-update rate regimes, not exact internal-state change times.",
                  "Unknown intervening regimes break same-vector duration runs."],
        "provenance": {"analysis_sha256": sha256(__file__), "sweep_summary_sha256": sha256(sweep/"summary.json"),
                       "baseline_summary_sha256": sha256(source/"summary.json")},
    }
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    for name, records in record_groups.items():
        (output/f"{name}-trials.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (output/"all-regime-readouts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in regimes))
    (output/"reset-context.json").write_text(json.dumps(reset_contexts, indent=2)+"\n")
    print(json.dumps({name:{k:v for k,v in values.items() if k not in
          ("trial_ids", "reset_counts", "best_terminal_same_durations", "terminal_unit_vectors")}
          for name,values in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
