"""Shorten the duration threshold while keeping the FSM signal test fixed.

Reports terminal retention separately from a supported optimum in any detected
regime. Treating the latter as permanent retention is an explicit assumption.
Segment-end evidence is retrospective; it is not an exact first-hit timestamp.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from read_bca_ip_fsm_outputs import (
    decode_terminal, evaluate, score_unit, segment_rates, sha256,
    solve_binary_instance, split_wire_times,
)


def collect_regime_evidence(wire_times, boundaries, width, instance):
    """Decode all regimes without using the target vector to choose their edges."""
    optimum, _ = solve_binary_instance(instance)
    evidence = []
    for u, edges in enumerate(boundaries):
        for start, end in zip(edges, edges[1:]):
            out = decode_terminal(wire_times[u*6:(u+1)*6], start*width, end*width,
                                  min_duration=1, max_window=300_000)
            out = score_unit(out, instance, optimum)
            if out["optimal"]:
                evidence.append(out | {"trial_id": u//2, "unit": "AB"[u%2],
                                       "segment_start": start*width, "segment_end": end*width})
    return sorted(evidence, key=lambda r: (r["trial_id"], r["segment_end"], r["unit"]))


def classify_duration(evidence, terminal_rows, minimum_duration):
    eligible = [e for e in evidence if e["regime_duration"] >= minimum_duration]
    reached = sorted({e["trial_id"] for e in eligible})
    retained = sorted(r["trial_id"] for r in terminal_rows if r["status"] == "optimal")
    if not set(retained).issubset(reached):
        raise ValueError("Terminal optimum is missing from regime evidence")
    first_support = {}
    for record in eligible:
        trial = record["trial_id"]
        if trial not in first_support or record["segment_end"] < first_support[trial]["segment_end"]:
            first_support[trial] = record
    n = len(terminal_rows)
    statuses = Counter(r["status"] for r in terminal_rows)
    return {
        "minimum_duration": minimum_duration,
        "terminal_optimal_trials": len(retained), "terminal_optimal_fraction": len(retained)/n,
        "terminal_nonoptimal_trials": statuses["stable_nonoptimal"],
        "terminal_unresolved_trials": statuses["unresolved"],
        "ever_supported_optimum_trials": len(reached), "ever_supported_optimum_fraction": len(reached)/n,
        "terminal_trial_ids": retained, "ever_supported_trial_ids": reached,
        "earlier_supported_but_not_terminal_ids": sorted(set(reached)-set(retained)),
    }, first_support


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("readout", type=Path, help="Output directory of read_bca_ip_fsm_outputs.py")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source, output = args.readout, args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / "summary.json").read_text())
    rows = [json.loads(s) for s in (source / "trials.jsonl").read_text().splitlines()]
    events = np.load(source / "fsm-event-times.npz")["events"]
    n, horizon, instance = summary["trials"], summary["horizon"], summary["instance"]
    if len(events) != summary["provenance"]["fsm_events"]:
        raise ValueError("FSM event total does not match the original audit")
    if [r["trial_id"] for r in rows] != list(range(n)):
        raise ValueError("Missing or duplicate trial rows")
    if sha256(Path(__file__).with_name("read_bca_ip_fsm_outputs.py")) != summary["provenance"]["decoder_sha256"]:
        raise ValueError("Baseline decoder has changed")
    wires = split_wire_times(events, n)
    # Original split points reproduce the baseline exactly. The finer fit also
    # removes its 30,000-update segment floor, permitting short terminal regimes.
    original = [[v//10_000 for v in (0, *r[unit]["change_points"], horizon)]
                for r in rows for unit in "AB"]
    fine_counts = np.zeros((n*2, horizon//5000, 6), dtype=np.int32)
    np.add.at(fine_counts, (events[:,0]*2+events[:,1]//6,
                           (events[:,2]-1)//5000, events[:,1]%6), 1)
    fine = segment_rates(fine_counts, penalty=20., min_bins=1)
    durations = [100_000, 75_000, 50_000, 40_000, 30_000, 20_000, 10_000]
    tables, details = {}, {}
    for label, boundaries, width in (("original", original, 10_000), ("fine", fine, 5000)):
        evidence = collect_regime_evidence(wires, boundaries, width, instance)
        table = []
        for duration in durations:
            terminal, _ = evaluate(wires, boundaries, width, n, horizon, instance,
                                    min_duration=duration, max_window=300_000)
            stats, first_support = classify_duration(evidence, terminal, duration)
            table.append(stats)
            if duration in (100_000, 10_000):
                details[f"{label}_{duration}"] = {
                    "terminal": terminal, "earliest_supporting_segment": first_support}
            print(json.dumps({"fit": label, **{k:v for k,v in stats.items() if not k.endswith("ids")}},
                             ensure_ascii=False), flush=True)
        tables[label] = table
        (output / f"{label}-optimal-regimes.jsonl").write_text(
            "".join(json.dumps(r)+"\n" for r in evidence))
    baseline = tables["original"][0]
    assert baseline["terminal_trial_ids"] == summary["trial_ids_by_status"]["optimal"]
    for table in tables.values():
        for previous, current in zip(table, table[1:]):
            assert set(previous["terminal_trial_ids"]).issubset(current["terminal_trial_ids"])
            assert set(previous["ever_supported_trial_ids"]).issubset(current["ever_supported_trial_ids"])
    window_control = []
    for duration in durations:
        _, stats = evaluate(wires, original, 10_000, n, horizon, instance,
                            min_duration=duration, max_window=duration)
        window_control.append({"minimum_duration": duration, "max_window": duration,
                               **{k: stats[k] for k in ("optimal_trials", "optimal_fraction",
                                      "stable_nonoptimal_trials", "unresolved_trials")}})
    result = {
        "schema": "pybca-short-stability-sweep-v1", "trials": n, "horizon": horizon,
        "assumption": "Once supported, the optimum continues; applied only to the ever-supported scenario, not asserted as an observed fact.",
        "signal_policy_unchanged": {k: summary["readout_policy"][k] for k in
             ("max_window", "active_events", "off_events", "occupied_quarters")},
        "fits": {"original": {"bin_width": 10_000, "min_segment_bins": 3, "penalty": 20.},
                 "fine": {"bin_width": 5000, "min_segment_bins": 1, "penalty": 20.}},
        "tables": tables,
        "observation_window_control": window_control,
        "duration_note": "Main tables change only the minimum persistence duration; using exclusively a short observation window is a separate control.",
        "new_terminal_ids": sorted(set(tables["original"][-1]["terminal_trial_ids"])-
                                   set(baseline["terminal_trial_ids"])),
        "first_hit_times": None,
        "evidence_time_note": "Earliest supporting segment END, not an exact first detection or a causal online timestamp.",
        "provenance": {"baseline_readout": str(source.resolve()), "sweep_sha256": sha256(__file__),
                       **{name:sha256(source/name) for name in
                          ("summary.json", "trials.jsonl", "fsm-event-times.npz", "fsm-output-counts.npz")}},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (output / "trial-details.json").write_text(json.dumps(details)+"\n")


if __name__ == "__main__":
    main()
