"""Find the last uninterrupted optimal FSM output interval ending at T.

This is a retrospective analysis conditional on a fixed observation endpoint.
The curve counts only trials retaining an optimum through T for >=100k steps;
it is neither an ever-hit curve nor a causal online first-detection curve.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from plot_bca_ip_retention import wilson_interval
from read_bca_ip_fsm_outputs import (
    decode_terminal, score_unit, segment_rates, sha256, solve_binary_instance,
    split_wire_times,
)

ROOT = Path(__file__).resolve().parents[1]


def terminal_episode(regimes, horizon, minimum_duration):
    """Merge adjacent optimal regimes only if the six-bit vector is identical.

    An unreadable/nonoptimal regime or a different optimum ends persistence.
    A flow-rate change alone does not end persistence. Intervals are (start,end].
    """
    if minimum_duration <= 0 or not regimes:
        raise ValueError("Require regimes and a positive minimum duration")
    previous = 0
    for row in regimes:
        if row["start"] != previous or row["end"] <= row["start"]:
            raise ValueError("Regimes must cover a contiguous history from step zero")
        previous = row["end"]
    if previous != horizon:
        raise ValueError("Regimes must end at the observation horizon")
    final = regimes[-1]
    if not final["optimal"]:
        return {"qualified": False, "start": None, "duration": 0, "vector": None,
                "merged_regimes": 0, "reason": "terminal_output_not_confirmed_optimal"}
    start, merged = final["start"], 1
    for row in reversed(regimes[:-1]):
        if not row["optimal"] or row["vector"] != final["vector"]:
            break
        start, merged = row["start"], merged + 1
    duration = horizon - start
    return {"qualified": duration >= minimum_duration, "start": start,
            "duration": duration, "vector": final["vector"], "merged_regimes": merged,
            "reason": "qualified" if duration >= minimum_duration else "terminal_optimum_too_recent"}


def combine_units(a, b):
    """A single unit must support the complete interval; count each trial once."""
    candidates = [(unit, row) for unit, row in (("A", a), ("B", b)) if row["qualified"]]
    if not candidates:
        return {"qualified": False, "start": None, "supporting_units": [], "selected_unit": None}
    unit, row = min(candidates, key=lambda x: (x[1]["start"], x[0]))
    return {"qualified": True, "start": row["start"], "selected_unit": unit,
            "supporting_units": [name for name, _ in candidates]}


def retained_curve(starts, *, trials, horizon, minimum_duration, step=10000):
    if len(starts) != trials or horizon < minimum_duration or step < 1:
        raise ValueError("Invalid curve dimensions")
    qualified = sorted(int(s) for s in starts if s is not None)
    if any(s < 0 or s > horizon - minimum_duration for s in qualified):
        raise ValueError("A retained interval is shorter than the minimum")
    times = np.unique(np.r_[np.arange(0, horizon + 1, step), qualified,
                            horizon - minimum_duration, horizon]).astype(np.int64)
    counts = np.searchsorted(qualified, times, side="right")
    lo, hi = wilson_interval(counts, trials)
    return [{"step": int(t), "trials": int(n), "fraction": float(n / trials),
             "wilson95": [float(a), float(b)]} for t, n, a, b in zip(times, counts, lo, hi)]


def analyze_case(spec, *, minimum_duration):
    source = ROOT / spec["readout"]
    meta = json.loads((source / "summary.json").read_text())
    previous_rows = [json.loads(s) for s in (source / "trials.jsonl").read_text().splitlines()]
    trials, horizon = meta["trials"], spec.get("horizon", meta["horizon"])
    if horizon > meta["horizon"] or horizon % 10000 or horizon < minimum_duration:
        raise ValueError("Requested horizon must be available and align with the 10k bins")
    if [r["trial_id"] for r in previous_rows] != list(range(trials)):
        raise ValueError("Source trial IDs are missing, duplicated or reordered")
    if sha256(ROOT / "scripts/read_bca_ip_fsm_outputs.py") != meta["provenance"]["decoder_sha256"]:
        raise ValueError("The source decoder has changed")
    all_events = np.load(source / "fsm-event-times.npz")["events"]
    if len(all_events) != meta["provenance"]["fsm_events"]:
        raise ValueError("Source FSM count mismatch")
    if np.any(all_events[:, 2] < 1) or np.any(all_events[:, 2] > meta["horizon"]):
        raise ValueError("Event outside the source horizon")
    events = all_events[all_events[:, 2] <= horizon]
    wires = split_wire_times(events, trials)
    if horizon == meta["horizon"]:
        edges = [[0, *row[unit]["change_points"], horizon]
                 for row in previous_rows for unit in "AB"]
    else:
        # Never use later change points when analyzing an earlier endpoint.
        counts = np.zeros((trials * 2, horizon // 10000, 6), dtype=np.int32)
        np.add.at(counts, (events[:, 0] * 2 + events[:, 1] // 6,
                          (events[:, 2] - 1) // 10000, events[:, 1] % 6), 1)
        edges = [[int(x * 10000) for x in row] for row in segment_rates(counts)]
    optimum, vectors = solve_binary_instance(meta["instance"])
    all_regimes, unit_episodes = [], []
    for unit, boundaries in enumerate(edges):
        regimes = []
        for start, end in zip(boundaries, boundaries[1:]):
            # Inspect the WHOLE regime, not just its last 300k steps. Otherwise
            # earlier ambiguous or nonoptimal output could be silently hidden.
            decoded = decode_terminal(wires[unit * 6:(unit + 1) * 6], start, end,
                                      min_duration=1, max_window=end - start)
            scored = score_unit(decoded, meta["instance"], optimum)
            regimes.append({"start": start, "end": end, **scored})
        episode = terminal_episode(regimes, horizon, minimum_duration)
        unit_episodes.append(episode)
        all_regimes.append({"trial_id": unit // 2, "unit": "AB"[unit % 2],
                            "terminal_episode": episode, "regimes": regimes})
    rows = []
    for trial in range(trials):
        a, b = unit_episodes[trial * 2:trial * 2 + 2]
        combined = combine_units(a, b)
        rows.append({"trial_id": trial, **combined, "A": a, "B": b})
    qualified_ids = [r["trial_id"] for r in rows if r["qualified"]]
    statuses = Counter("qualified" if row["qualified"] else
                       "terminal_optimum_too_recent" if any(row[u]["start"] is not None for u in "AB")
                       else "terminal_output_not_confirmed_optimal" for row in rows)
    curve = retained_curve([r["start"] for r in rows], trials=trials, horizon=horizon,
                           minimum_duration=minimum_duration)
    if curve[-1]["trials"] != len(qualified_ids):
        raise AssertionError("Curve endpoint and trial classification disagree")
    baseline = (set(meta["trial_ids_by_status"]["optimal"])
                if horizon == meta["horizon"] else None)
    result = {"case": spec["id"], "label": spec["label"], "condition": spec["condition"],
              "N": spec["N"], "horizon": horizon, "trials": trials,
              "minimum_duration": minimum_duration, "qualified_trials": len(qualified_ids),
              "qualified_fraction": len(qualified_ids) / trials, "qualified_trial_ids": qualified_ids,
              "status_counts": dict(statuses), "optimum_value": optimum, "optimal_vectors": vectors,
              "legacy_reset_mismatch": spec.get("legacy_reset_mismatch", False),
              "interim": spec.get("interim", False), "curve": curve,
              "baseline_policy_difference": None if baseline is None else {
                  "previous_terminal_regime_minimum_100k": meta["optimal_trials"],
                  "included_now": sorted(set(qualified_ids) - baseline),
                  "excluded_now": sorted(baseline - set(qualified_ids))},
              "provenance": {"source_readout": spec["readout"], "source_horizon": meta["horizon"],
                  "source_summary_sha256": sha256(source / "summary.json"),
                  "source_trials_sha256": sha256(source / "trials.jsonl"),
                  "source_events_sha256": sha256(source / "fsm-event-times.npz"),
                  "fsm_events_used": len(events), "script_sha256": sha256(__file__),
                  "future_events_beyond_requested_horizon_used": False}}
    return result, rows, all_regimes


def plot_cases(results, keys, output, *, title, ymax=100, legacy_note=False,
               legend_location="lower right"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    fig.subplots_adjust(left=.095, right=.98, top=.84, bottom=.24)
    max_t = max(results[key]["horizon"] for key in keys)
    for i, key in enumerate(keys):
        case = results[key]
        curve = case["curve"]
        x = np.array([r["step"] for r in curve]) / 1e6
        y = np.array([r["fraction"] for r in curve]) * 100
        interval = np.array([r["wilson95"] for r in curve]) * 100
        label = f"{case['label']}: {case['qualified_trials']}/{case['trials']} ({y[-1]:.2f}%)"
        color = colors[i % len(colors)]
        style = "--" if case["legacy_reset_mismatch"] else "-"
        ax.step(x, y, where="post", label=label, color=color, ls=style, lw=2)
        ax.fill_between(x, interval[:, 0], interval[:, 1], step="post", color=color, alpha=.10)
        ax.scatter(x[-1], y[-1], s=24, color=color, zorder=3)
    if len({results[k]["horizon"] for k in keys}) == 1:
        t = results[keys[0]]["horizon"]
        cutoff = t - results[keys[0]]["minimum_duration"]
        ax.axvspan(cutoff / 1e6, t / 1e6, color=".5", alpha=.10)
        ax.axvline(cutoff / 1e6, color=".4", lw=.8, ls=":")
    ax.set(xlim=(0, max_t / 1e6), ylim=(0, ymax),
           xlabel="Start of the final uninterrupted optimal-output interval (million steps)",
           ylabel="Trials retaining that optimum through T (%)")
    ax.grid(alpha=.20)
    ax.legend(loc=legend_location, frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=15, y=.96)
    durations = sorted({results[key]["minimum_duration"] for key in keys})
    trial_counts = sorted({results[key]["trials"] for key in keys})
    subtitle = "Minimum terminal duration: " + "/".join(f"{d:,}" for d in durations) + " steps"
    subtitle += "   |   " + "/".join(str(n) for n in trial_counts) + " trials per condition"
    fig.text(.5, .895, subtitle, ha="center")
    note = "Retrospective curve: only intervals continuing through the fixed endpoint T count.\n"
    note += "The same unit and optimal vector must persist. Shading: pointwise 95% Wilson intervals."
    if legacy_note:
        note += "\nDashed legacy N=1 has mismatched reset gains; this is not a comparison changing only N."
    fig.text(.095, .055, note, fontsize=9, va="bottom", color=".3")
    for extension in ("png", "svg", "pdf"):
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=200)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in config["cases"]:
        result, rows, regimes = analyze_case(case, minimum_duration=config["minimum_duration"])
        name = case["id"]
        results[name] = result
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        for suffix, data in (("trials", rows), ("regimes", regimes)):
            (args.output / f"{name}-{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
        print(json.dumps({k: result[k] for k in ("case", "horizon", "qualified_trials", "qualified_fraction",
                                                 "baseline_policy_difference")}), flush=True)
    for chart in config["plots"]:
        plot_cases(results, chart["cases"], args.output / chart["file"], title=chart["title"],
                   ymax=chart.get("ymax", 100), legacy_note=chart.get("legacy_note", False),
                   legend_location=chart.get("legend_location", "lower right"))
    limitations = ["Continuous output is inferred from sparse events and rate regimes, not observed at every CA step.",
                   "Estimated episode starts are not exact internal first-hit times or causal confirmation times."]
    if any(r["legacy_reset_mismatch"] for r in results.values()):
        limitations.append("The historical condition-2 N=1 circuit has reset gains [4,3,2,1,5,5] despite initial Weights [4,4,5,1,1,1]; it is not a matched control for the new condition-2 N=2 circuit.")
    if any(r["interim"] for r in results.values()):
        limitations.append("An interim endpoint cannot establish persistence beyond its observation horizon.")
    aggregate = {"created_at": datetime.now(timezone.utc).isoformat(),
                 "schema": "pybca-terminal-uninterrupted-optimum-v1",
                 "minimum_duration": config["minimum_duration"],
                 "curve_definition": "F_T(t) = number of trials with s <= t and T-s >= minimum_duration, divided by all trials; s is the beginning of an uninterrupted same-vector optimum episode in one unit that ends at T.",
                 "policy": {"bin_width": 10000, "change_penalty": 20, "min_segment_bins": 3,
                            "active_events": 4, "off_events": 2, "occupied_quarters": 3,
                            "read_entire_rate_regime": True, "merge_same_optimal_vector": True,
                            "unknown_or_nonoptimal_breaks_continuity": True,
                            "different_optimum_breaks_continuity": True,
                            "switching_units_to_bridge_a_gap": False,
                            "counts_require_final_horizon_information": True},
                 "limitations": limitations,
                 "cases": [{k: r[k] for k in ("case", "label", "condition", "N", "horizon", "trials",
                                               "qualified_trials", "qualified_fraction", "interim",
                                               "legacy_reset_mismatch", "baseline_policy_difference")}
                           for r in results.values()],
                 "config_sha256": sha256(args.config), "script_sha256": sha256(__file__)}
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    main()
