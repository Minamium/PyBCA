"""Compare persistent FSM readouts at two horizons of the same trial cohort.

Whole-trajectory segmentation is retrospective. Preserve previously observed
evidence separately if refitting the longer trajectory revises an earlier hit.
This script does not infer an exact first-hit time from a segment boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_row(before, after, trials):
    if before["minimum_duration"] != after["minimum_duration"]:
        raise ValueError("Different persistence thresholds")
    old_hit, new_hit = (set(r["ever_supported_trial_ids"]) for r in (before, after))
    old_final, new_final = (set(r["terminal_trial_ids"]) for r in (before, after))
    universe = set(range(trials))
    for row, hits, final in ((before, old_hit, old_final), (after, new_hit, new_final)):
        if not final <= hits <= universe:
            raise ValueError("Inconsistent hit or trial IDs")
        if (len(hits) != row["ever_supported_optimum_trials"] or
                len(final) != row["terminal_optimal_trials"]):
            raise ValueError("Counts disagree with distinct trial IDs")
    groups = {
        "terminal_at_both": old_final & new_final,
        "new_terminal": new_final - old_final,
        "no_longer_terminal": old_final - new_final,
        "terminal_at_neither": universe - (old_final | new_final),
        "newly_supported_in_longer_fit": new_hit - old_hit,
        "prior_support_not_reproduced_by_longer_fit": old_hit - new_hit,
        "cumulative_supported": old_hit | new_hit,
        "cumulative_never_supported": universe - (old_hit | new_hit),
        "cumulative_supported_not_terminal": (old_hit | new_hit) - new_final,
        "previously_supported_only_now_terminal": (old_hit - old_final) & new_final,
        "previously_never_supported_now_terminal": new_final - old_hit,
    }
    assert sum(len(groups[k]) for k in (
        "terminal_at_both", "new_terminal", "no_longer_terminal", "terminal_at_neither")) == trials
    return {
        "minimum_duration": before["minimum_duration"],
        "previous_terminal_count": len(old_final), "current_terminal_count": len(new_final),
        "previous_ever_count": len(old_hit), "current_ever_count": len(new_hit),
        "current_terminal_fraction": len(new_final) / trials,
        "current_ever_fraction": len(new_hit) / trials,
        "current_stable_nonoptimal_count": after["terminal_nonoptimal_trials"],
        "current_unresolved_count": after["terminal_unresolved_trials"],
        "counts": {k: len(v) for k, v in groups.items()},
        "trial_ids": {k: sorted(v) for k, v in groups.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path, help="Earlier short-stability output")
    parser.add_argument("current", type=Path, help="Later short-stability output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [p / "summary.json" for p in (args.previous, args.current)]
    old, new = [json.loads(p.read_text()) for p in paths]
    if old["trials"] != new["trials"] or old["horizon"] >= new["horizon"]:
        raise ValueError("Different cohort sizes or unordered horizons")
    for key in ("fits", "signal_policy_unchanged"):
        if old[key] != new[key]:
            raise ValueError(f"Changed readout setting: {key}")
    tables = {}
    for fit in old["tables"]:
        a = old["tables"][fit]
        b = new["tables"][fit]
        if len(a) != len(b):
            raise ValueError("Different threshold sweeps")
        evidence = [json.loads(s) for s in (args.current / f"{fit}-optimal-regimes.jsonl").read_text().splitlines()]
        rows = []
        for previous, current in zip(a, b):
            row = compare_row(previous, current, old["trials"])
            new_ids = set(row["trial_ids"]["newly_supported_in_longer_fit"])
            later = {e["trial_id"] for e in evidence
                     if e["regime_duration"] >= row["minimum_duration"]
                     and e["segment_end"] > old["horizon"]}
            row["new_ids_with_support_after_previous_horizon"] = sorted(new_ids & later)
            row["new_ids_supported_only_before_previous_horizon"] = sorted(new_ids - later)
            rows.append(row)
        tables[fit] = rows
    output = {
        "schema": "pybca-horizon-comparison-v1", "trials": old["trials"],
        "previous_horizon": old["horizon"], "current_horizon": new["horizon"],
        "signal_policy": old["signal_policy_unchanged"], "fits": old["fits"],
        "tables": tables,
        "same_ids_between_fits": all(
            set(new["tables"]["original"][i][key]) == set(new["tables"]["fine"][i][key])
            for i in range(len(new["tables"]["original"]))
            for key in ("terminal_trial_ids", "ever_supported_trial_ids")),
        "note": "Readout observation counts, not causal first-hit times. A/B count as one trial. Cohort identity and exact history-prefix equality are checked separately.",
        "provenance": {"script_sha256": sha256(__file__),
                       "previous_summary_sha256": sha256(paths[0]),
                       "current_summary_sha256": sha256(paths[1])},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"same_ids_between_fits": output["same_ids_between_fits"],
                      "original": [{k: v for k, v in row.items() if k != "trial_ids"}
                                   for row in tables["original"]]}, indent=2))


if __name__ == "__main__":
    main()
