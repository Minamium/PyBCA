"""Read terminal BCA-IP solutions from persistent, unamplified FSM outputs.

The signal decoder does not use feasibility or optimality. It first finds the
last joint rate change, then reads each of the six wires in the terminal regime.
An instance is used only afterwards to score the resulting binary vectors.
This estimates the solution retained at the horizon, not the first-ever hit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_events(run, expected_trials, horizon):
    """Recheck all immutable chunks; retain FSM events and logical reset sets."""
    names = {f"{unit}_x{x+1}output": u*6+x
             for u, unit in enumerate("AB") for x in range(6)}
    rows, resets, ids, manifests = [], [], [], {}
    checked_chunks = raw_records = 0
    for rank in sorted(run.glob("rank_????")):
        manifest = rank / "manifest.json"
        m = json.loads(manifest.read_text())
        manifests[rank.name] = sha256(manifest)
        if m["next_step"] != horizon:
            raise ValueError(f"Incomplete rank: {rank}")
        ids.extend(m["trial_ids"])
        previous = 0
        for chunk in m["chunks"]:
            raw = (rank / chunk["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
                raise ValueError(f"History hash mismatch: {rank / chunk['path']}")
            lines = raw.splitlines()
            if (chunk["start_step"] != previous or
                    len(lines)-1 != chunk["records"]):
                raise ValueError("Noncontiguous history or incorrect row count")
            header = json.loads(lines[0])["__chunk__"]
            if (header["start_step"] != previous or
                    header["next_step"] != chunk["next_step"]):
                raise ValueError("Chunk header mismatch")
            raw_records += len(lines)-1
            checked_chunks += 1
            for line in lines[1:]:
                if not any(s in line for s in
                           (b'"A_x', b'"B_x', b'Reset_Signal_for_')):
                    continue
                r = json.loads(line)
                if (r["kind"] != "event" or r["count"] != 1 or
                        r["trial"] not in m["trial_ids"] or
                        not previous <= r["step"] < chunk["next_step"]):
                    raise ValueError("Invalid FSM/reset event")
                if r["name"] in names:
                    rows.append((r["trial"], names[r["name"]], r["step"]+1))
                elif r["name"].endswith("_set"):
                    resets.append((r["trial"], "AB".index(r["name"].split("_")[3]),
                                   r["step"]+1))
            previous = chunk["next_step"]
        if previous != horizon:
            raise ValueError("History does not reach the horizon")
        print(f"Read and checked {rank.name}", flush=True)
    if sorted(ids) != list(range(expected_trials)):
        raise ValueError("Missing or duplicate trial IDs")
    events = np.asarray(rows, dtype=np.int32).reshape(-1, 3)
    order = np.lexsort((events[:, 2], events[:, 1], events[:, 0]))
    events = events[order]
    if len(events) > 1 and np.any(np.all(events[1:] == events[:-1], axis=1)):
        raise ValueError("Duplicate FSM events")
    return events, np.asarray(resets, dtype=np.int32).reshape(-1, 3), {
        "manifest_sha256": manifests, "checked_chunks": checked_chunks,
        "raw_records": raw_records, "fsm_events": len(events),
        "reset_set_events": len(resets), "time_convention": "completed CA updates; raw step + 1",
    }


def segment_rates(counts, penalty=20., min_bins=3):
    """Joint piecewise-constant Poisson-rate fit, by exact dynamic programming.

    counts has shape (units, time bins, wires). Ignoring partition-independent
    terms, the negative maximized log likelihood of a segment is
    -sum_x C_x log(C_x / length). The penalty is charged per extra segment.
    This is an operational change detector, not a claim that Brownian arrivals
    are independent Poisson events or a calibrated significance test.
    """
    counts = np.asarray(counts)
    if counts.ndim != 3 or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Require finite nonnegative counts shaped units/time/wires")
    units, n, wires = counts.shape
    if not units or not wires or min_bins < 1 or n < min_bins or penalty < 0:
        raise ValueError("Invalid segmentation dimensions or parameters")
    cumulative = np.concatenate([np.zeros((units, 1, wires)),
                                 counts.cumsum(axis=1)], axis=1)
    cost = np.full((units, n+1), np.inf)
    cost[:, 0] = -penalty
    previous = np.zeros((units, n+1), dtype=np.int32)
    unit_ids = np.arange(units)
    for end in range(min_bins, n+1):
        starts = np.r_[0, np.arange(min_bins, end-min_bins+1)]
        amount = cumulative[:, end:end+1] - cumulative[:, starts]
        length = end-starts
        fit = -xlogy(amount, amount/length[None, :, None]).sum(axis=-1)
        scores = cost[:, starts] + fit + penalty
        best = scores.argmin(axis=1)
        cost[:, end] = scores[unit_ids, best]
        previous[:, end] = starts[best]
    boundaries = []
    for unit in range(units):
        end, edges = n, [n]
        while end:
            end = int(previous[unit, end])
            edges.append(end)
        boundaries.append(edges[::-1])
    return boundaries


def split_wire_times(events, trials):
    keys = events[:, 0]*12 + events[:, 1]
    index = np.searchsorted(keys, np.arange(trials*12+1))
    return [events[index[i]:index[i+1], 2] for i in range(trials*12)]


def decode_terminal(wires, last_change, horizon, min_duration=100_000,
                    max_window=300_000, active_events=4, off_events=2,
                    occupied_quarters=3):
    """Read wires without consulting the optimization problem; None is ambiguous.

    Intervals are (start, end] in completed updates. Quarter counts use exact
    event times, even though rate-change locations use a coarser grid.
    """
    if not 0 <= last_change < horizon or min_duration <= 0 or max_window < min_duration:
        raise ValueError("Invalid terminal interval")
    if not 0 <= off_events < active_events or not 1 <= occupied_quarters <= 4:
        raise ValueError("Invalid signal thresholds")
    start = max(last_change, horizon-max_window)
    edges = np.array([start + (horizon-start)*i//4 for i in range(5)])
    quarters = np.array([np.diff(np.searchsorted(t, edges, side="right")) for t in wires])
    counts = quarters.sum(axis=1)
    on = (counts >= active_events) & ((quarters > 0).sum(axis=1) >= occupied_quarters)
    off = counts <= off_events
    vector = [1 if a else 0 if b else None for a, b in zip(on, off)]
    sufficiently_long = horizon-last_change >= min_duration
    readable = sufficiently_long and all(v is not None for v in vector)
    return {
        "last_rate_change": int(last_change), "readout_start": int(start),
        "readout_end": int(horizon), "regime_duration": int(horizon-last_change),
        "quarter_edges": edges.tolist(), "counts": counts.tolist(),
        "quarter_counts": quarters.tolist(), "vector": vector,
        "sufficient_duration": bool(sufficiently_long), "stable_readout": bool(readable),
    }


def solve_binary_instance(instance):
    a, c, b = np.asarray(instance["a"]), np.asarray(instance["c"]), instance["b"]
    if a.shape != (6,) or c.shape != (6,) or not np.isfinite([*a, *c, b]).all():
        raise ValueError("Require finite six-variable instance coefficients")
    candidates = np.array(list(itertools.product((0, 1), repeat=6)))
    feasible = candidates[candidates @ a <= b]
    if not len(feasible):
        raise ValueError("Instance has no feasible binary vector")
    values = feasible @ c
    optimum = values.max().item()
    return optimum, feasible[values == optimum].tolist()


def score_unit(readout, instance, optimum):
    """Never repair a decoded vector to make it feasible or optimal."""
    if not readout["stable_readout"]:
        return readout | {"constraint_value": None, "objective_value": None,
                          "feasible": None, "optimal": False}
    x = np.array(readout["vector"])
    constraint = (x @ np.asarray(instance["a"])).item()
    value = (x @ np.asarray(instance["c"])).item()
    feasible = constraint <= instance["b"]
    return readout | {"constraint_value": constraint, "objective_value": value,
                      "feasible": feasible, "optimal": feasible and value == optimum}


def evaluate(wire_times, boundaries, bin_width, trials, horizon, instance, **params):
    optimum, solutions = solve_binary_instance(instance)
    units = []
    for u, edges in enumerate(boundaries):
        readout = decode_terminal(wire_times[u*6:(u+1)*6], edges[-2]*bin_width,
                                  horizon, **params)
        row = score_unit(readout, instance, optimum)
        units.append(row | {"trial_id": u//2, "unit": "AB"[u%2],
                            "change_points": [int(v*bin_width) for v in edges[1:-1]]})
    trial_rows = []
    for t in range(trials):
        a, b = units[2*t:2*t+2]
        status = ("optimal" if a["optimal"] or b["optimal"] else
                  "stable_nonoptimal" if a["stable_readout"] and b["stable_readout"] else
                  "unresolved")
        trial_rows.append({"trial_id": t, "status": status, "A": a, "B": b})
    counts = Counter(r["status"] for r in trial_rows)
    summary = {
        "trials": trials, "horizon": horizon, "optimum_value": optimum,
        "optimum_vectors": solutions,
        "optimal_trials": counts["optimal"],
        "stable_nonoptimal_trials": counts["stable_nonoptimal"],
        "unresolved_trials": counts["unresolved"],
        "optimal_fraction": counts["optimal"]/trials,
        "not_confirmed_optimal_fraction": 1-counts["optimal"]/trials,
        "stable_units": sum(u["stable_readout"] for u in units),
        "stable_infeasible_units": sum(u["feasible"] is False for u in units),
        "both_units_optimal_trials": sum(r["A"]["optimal"] and r["B"]["optimal"] for r in trial_rows),
        "optimal_A_only_trials": sum(r["A"]["optimal"] and not r["B"]["optimal"] for r in trial_rows),
        "optimal_B_only_trials": sum(r["B"]["optimal"] and not r["A"]["optimal"] for r in trial_rows),
        "stable_vector_counts": dict(sorted(Counter("".join(map(str, u["vector"]))
            for u in units if u["stable_readout"]).items())),
        "trial_ids_by_status": {s: [r["trial_id"] for r in trial_rows if r["status"] == s]
                                for s in ("optimal", "stable_nonoptimal", "unresolved")},
    }
    return trial_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-trials", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=3_000_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trials, horizon, width = args.expected_trials, args.horizon, 10_000
    if horizon % width or horizon < 100_000:
        raise ValueError("Horizon must be a multiple of 10,000 and at least 100,000")
    instance = json.loads(args.instance.read_text())
    events, resets, provenance = extract_events(args.run, trials, horizon)
    counts = np.zeros((trials*2, horizon//width, 6), dtype=np.int32)
    np.add.at(counts, (events[:, 0]*2+events[:, 1]//6,
                      (events[:, 2]-1)//width, events[:, 1]%6), 1)
    assert int(counts.sum()) == len(events)
    wire_times = split_wire_times(events, trials)
    fits = {p: segment_rates(counts, penalty=p) for p in (10., 20., 30.)}
    policy = {"min_duration": 100_000, "max_window": 300_000,
              "active_events": 4, "off_events": 2, "occupied_quarters": 3}
    rows, summary = evaluate(wire_times, fits[20.], width, trials, horizon, instance, **policy)
    sensitivity = []
    choices = [(20., policy, "primary")]
    choices += [(p, policy, f"penalty_{p:g}") for p in (10., 30.)]
    choices += [(20., policy | {"min_duration": d}, f"min_duration_{d}")
                for d in (50_000, 150_000, 200_000)]
    choices += [(20., policy | {"max_window": w}, f"max_window_{w}")
                for w in (200_000, horizon)]
    choices += [(20., policy | {"off_events": n}, f"off_events_{n}") for n in (0, 1)]
    choices += [(20., policy | {"occupied_quarters": 4}, "all_four_quarters")]
    for penalty, settings, label in choices:
        _, s = evaluate(wire_times, fits[penalty], width, trials, horizon, instance, **settings)
        sensitivity.append({"label": label, "penalty": penalty, **settings,
                            **{k: s[k] for k in ("optimal_trials", "optimal_fraction",
                                  "stable_nonoptimal_trials", "unresolved_trials")}})
    summary |= {
        "schema": "pybca-terminal-fsm-readout-v1", "instance": instance,
        "readout_policy": policy | {"bin_width": width, "change_penalty": 20., "min_segment_bins": 3},
        "provenance": provenance | {"run": str(args.run.resolve()),
                                    "decoder_sha256": sha256(__file__),
                                    "instance_sha256": sha256(args.instance)},
        "sensitivity": sensitivity,
        "interpretation": "Post-hoc terminal persistent-output readout; not an internal-state or first-hit measurement.",
        "first_hit_steps": None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (args.output / "trials.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    np.savez_compressed(args.output / "fsm-output-counts.npz", counts=counts, bin_width=width, resets=resets)
    np.savez_compressed(args.output / "fsm-event-times.npz", events=events, resets=resets)
    print(json.dumps({k: summary[k] for k in ("optimal_trials", "optimal_fraction",
          "stable_nonoptimal_trials", "unresolved_trials", "stable_infeasible_units",
          "optimal_A_only_trials", "optimal_B_only_trials", "both_units_optimal_trials")}, indent=2))
    print(json.dumps(sensitivity, indent=2))


if __name__ == "__main__":
    main()
