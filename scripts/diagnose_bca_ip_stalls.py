"""Trace stage flows and resets around persistent nonoptimal FSM readouts.

All counts come from the recorded full circuit. Rankings use stable joint
output regimes; they are observational comparisons, not a causal Amp ablation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from read_bca_ip_fsm_outputs import decode_terminal, score_unit, split_wire_times


def extract_counts(run, trials, horizon, width=10_000):
    first = json.loads((run / "rank_0000/manifest.json").read_text())
    names = sorted(first["events"])
    index = {name: j for j, name in enumerate(names)}
    counts = np.zeros((trials, horizon // width, len(names)), dtype=np.int32)
    checked = records = 0
    for rank in sorted(run.glob("rank_????")):
        m = json.loads((rank / "manifest.json").read_text())
        assert m["next_step"] == horizon and sorted(m["events"]) == names
        for chunk in m["chunks"]:
            raw = (rank / chunk["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == chunk["sha256"]
            lines = raw.splitlines()[1:]
            assert len(lines) == chunk["records"]
            for line in lines:
                r = json.loads(line)
                assert r["trial"] in m["trial_ids"] and r["count"] == 1
                counts[r["trial"], r["step"] // width, index[r["name"]]] += r["count"]
            records += len(lines)
            checked += 1
        print(f"Counted {rank.name}", flush=True)
    assert int(counts.sum()) == records
    return counts, names, {"chunks": checked, "records": records}


def ranking(objective_a, objective_b, count_a, count_b):
    truth = int(np.sign(objective_a-objective_b))
    flow = int(np.sign(count_a-count_b))
    return "objective_tie" if truth == 0 else "flow_tie" if flow == 0 else "agree" if truth == flow else "reverse"


def joint_intervals(a, b, horizon, minimum=300_000):
    edges = sorted({0, horizon, *a, *b})
    return [(start, end) for start, end in zip(edges, edges[1:]) if end-start >= minimum]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="Analysis root containing fsm-readout and nonhit-dynamics")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--previous-nonhits", type=Path, default=Path(__file__).resolve().parents[1] /
                    "docs/experiments/2026-09-28-production-p05/nonhit-trials.jsonl")
    args = ap.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "fsm-readout/summary.json").read_text())
    baseline = [json.loads(s) for s in (root / "fsm-readout/trials.jsonl").read_text().splitlines()]
    groups = json.loads((root / "nonhit-dynamics/summary.json").read_text())["groups"]
    never = [json.loads(s) for s in (root / "nonhit-dynamics/never_supported-trials.jsonl").read_text().splitlines()]
    raw = np.load(root / "fsm-readout/fsm-event-times.npz")
    wires = split_wire_times(raw["events"], summary["trials"])
    horizon, width = summary["horizon"], 10_000
    counts, names, audit = extract_counts(Path(summary["provenance"]["run"]), summary["trials"], horizon, width)
    index = {name: j for j, name in enumerate(names)}
    np.savez_compressed(out / "stage-counts.npz", counts=counts, names=np.array(names), bin_width=width)

    def stage(trial, start, end):
        amount = counts[trial, start//width:end//width].sum(axis=0)
        return {name: int(amount[j]) for j, name in enumerate(names)}

    intervals = []
    for trial, row in enumerate(baseline):
        for start, end in joint_intervals(row["A"]["change_points"], row["B"]["change_points"], horizon):
            pair = []
            for j in range(2):
                w = wires[(trial*2+j)*6:(trial*2+j+1)*6]
                d = decode_terminal(w, start, end, min_duration=300_000, max_window=300_000)
                pair.append(score_unit(d, summary["instance"], summary["optimum_value"]))
            if not all(d["stable_readout"] and d["feasible"] for d in pair):
                continue
            read_start = end-300_000
            n = stage(trial, read_start, end)
            resets = {u: n[f"Reset_Signal_for_{u}_set"] for u in "AB"}
            intervals.append({"trial_id": trial, "start": read_start, "end": end,
                              "A_vector": pair[0]["vector"], "B_vector": pair[1]["vector"],
                              "A_value": pair[0]["objective_value"], "B_value": pair[1]["objective_value"],
                              "unit_ranking": ranking(pair[0]["objective_value"], pair[1]["objective_value"], n["F_value_A"], n["F_value_B"]),
                              "comparator_ranking": ranking(pair[0]["objective_value"], pair[1]["objective_value"], n["Comparate_A-B"], n["Comparate_B-A"]),
                              "resets": resets, "events": n})
    (out / "joint-stable-intervals.jsonl").write_text("".join(json.dumps(r)+"\n" for r in intervals))
    ranking_groups = {}
    for group, desc in groups.items():
        ids = set(desc["trial_ids"])
        selected = [r for r in intervals if r["trial_id"] in ids]
        ranking_groups[group] = {"intervals": len(selected), "trials": len({r["trial_id"] for r in selected}),
            "unit_ranking": dict(Counter(r["unit_ranking"] for r in selected)),
            "comparator_ranking": dict(Counter(r["comparator_ranking"] for r in selected)),
            "trials_with_unit_reversal": sorted({r["trial_id"] for r in selected if r["unit_ranking"] == "reverse"}),
            "unit_reversals_without_reset_in_window": sum(r["unit_ranking"] == "reverse" and not any(r["resets"].values()) for r in selected)}

    leaders = []
    for r in never:
        best = r["best_readable_terminal_value"]
        choices = [u for u in "AB" if r[u]["terminal"]["objective_value"] == best]
        # A tie in decoded objective uses the longest observed constant tail.
        leader = max(choices, key=lambda u: r[u]["terminal_same_vector_duration"])
        other = "B" if leader == "A" else "A"
        start = horizon-r[leader]["terminal_same_vector_duration"]
        leaders.append({"trial_id": r["trial_id"], "leader": leader,
                        "vector": r[leader]["terminal"]["vector"], "value": best,
                        "same_vector_start": start, "same_vector_duration": horizon-start,
                        "leader_resets_during_tail": [t for t in r[leader]["reset_steps"] if t > start],
                        "other_resets_during_tail": [t for t in r[other]["reset_steps"] if t > start],
                        "last_300k_events": stage(r["trial_id"], horizon-300_000, horizon)})
    (out / "nonhit-leaders.jsonl").write_text("".join(json.dumps(r)+"\n" for r in leaders))
    best_vectors = Counter("".join(map(str, r["vector"])) for r in leaders)
    changes = json.loads((root / "horizon-comparison.json").read_text())["tables"]["original"][-1]
    old_nonhits = [json.loads(s) for s in args.previous_nonhits.read_text().splitlines()]
    arrivals = set(changes["trial_ids"]["newly_supported_in_longer_fit"])
    arrival_by_prior_value = {}
    for value in sorted({r["best_readable_terminal_value"] for r in old_nonhits}):
        selected = [r for r in old_nonhits if r["best_readable_terminal_value"] == value]
        arrival_by_prior_value[str(value)] = {"at_3m": len(selected), "reached_by_6m": sum(r["trial_id"] in arrivals for r in selected)}
    group_input = {}
    for name, desc in groups.items():
        ids = desc["trial_ids"]
        n = counts[ids].sum(axis=(0,1))
        group_input[name] = {"trials":len(ids), "core_input_counts": [sum(int(n[index[f"{u}_core_input_{j}"]]) for u in "AB") for j in range(1,7)]}
    result = {"schema":"pybca-stall-diagnosis-v1", "horizon":horizon, "audit":audit,
              "best_nonhit_vectors":dict(best_vectors),
              "leader_reset_during_constant_tail_trials":sum(bool(r["leader_resets_during_tail"]) for r in leaders),
              "other_reset_during_constant_tail_trials":sum(bool(r["other_resets_during_tail"]) for r in leaders),
              "arrival_by_3m_best_value":arrival_by_prior_value, "ranking_groups":ranking_groups,
              "td_input_by_group":group_input,
              "provenance":{"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "previous_nonhits_sha256":hashlib.sha256(args.previous_nonhits.read_bytes()).hexdigest(),
                            "readout_summary_sha256":hashlib.sha256((root/"fsm-readout/summary.json").read_bytes()).hexdigest(),
                            "dynamics_summary_sha256":hashlib.sha256((root/"nonhit-dynamics/summary.json").read_bytes()).hexdigest()},
              "notes":["Joint intervals have no detected rate change in either unit for at least 300k updates; readout and flow counts use their last 300k.",
                       "Intervals from the same trial are not independent trials; rank counts are descriptive.",
                       "A reset inside a constant decoded tail does not prove the internal state stayed unchanged.",
                       "TD monitor events are not direct measurements of completed acceptance into the FSM."]}
    (out / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
