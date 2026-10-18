"""Verify completed 6M readouts and reproduce the condition-wise figures.

The event-level cross-check retains the 15 October report's readout policy.
No simulation dynamics or optimum classification parameters are changed.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "docs/experiments/2026-10-15-interim-6m"
THREE = ROOT / "docs/experiments/2026-10-11-final-3m"
sys.path.insert(0, str(ROOT / "scripts"))
from plot_bca_ip_progress_summary import curve_figure, read_verified, save
from read_bca_ip_fsm_outputs import split_wire_times


def load(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(config):
    checks = []
    for spec in config["cases"]:
        source = ROOT / spec["readout"]
        directory = source.parent
        snap = load(directory / "snapshot.json")
        variant = snap["variant"]
        assert snap["passed"] and snap["horizon"] == 6000000 and snap["trials"] == 512
        assert snap["snapshot_script_sha256"] == digest(ROOT / "scripts/snapshot_bca_ip_readout.py")
        assert snap["extraction"]["decoder_sha256"] == digest(ROOT / "scripts/read_bca_ip_fsm_outputs.py")
        for name, expected in snap["files"].items():
            assert digest(directory / name) == expected, (variant, name)
        for name, expected in snap["source"]["implementation"].items():
            assert digest(ROOT / "src/PyBCA" / name) == expected, name
        audit_path = (OLD / "condition1_N1-completion-audit.json" if variant == "condition1_N1"
                      else OUT / "readout-source" / f"{variant}-completion-audit.json")
        audit, meta = load(audit_path), load(source / "summary.json")
        assert audit["passed"] and audit["steps"] == 6000000 and audit["trials"] == 512
        assert len(audit["ranks"]) == 8 and all(r["step"] == 6000000 and r["trials"] == 64 for r in audit["ranks"])
        for family, key in [("FSM", "fsm_events"), ("reset", "reset_set_events")]:
            assert audit["events_by_family"][family] == meta["provenance"][key]
        prefixes = []
        if variant == "condition1_N2":
            prefixes.append((OLD / "readout-source/condition1_N2/readout", 5300000))
            prefixes.append((THREE / "readout-source/condition1_N2/readout", 3000000))
        elif variant == "condition2_N2":
            prefixes.append((THREE / "readout-source/condition2_N2/readout", 3000000))
        else:
            prefixes.append((ROOT / "docs/experiments/2026-10-07-variants/status-20261008/condition1_N1-3m/readout", 3000000))
        with np.load(source / "fsm-event-times.npz") as current:
            for previous, horizon in prefixes:
                with np.load(previous / "fsm-event-times.npz") as earlier:
                    for kind in ("events", "resets"):
                        assert np.array_equal(current[kind][current[kind][:, 2] <= horizon], earlier[kind]), (variant, kind)
        checks.append({"variant": variant, "passed": True, "files_sha256_verified": len(snap["files"]),
                       "checked_chunks": snap["extraction"]["checked_chunks"],
                       "completion_audit_sha256": digest(audit_path),
                       "prior_FSM_and_reset_prefixes_exact": [h for _, h in prefixes],
                       "newly_collected": variant != "condition1_N1"})
    return checks


def verify_case(name):
    case = load(OUT / f"{name}.json")
    source = ROOT / case["provenance"]["source_readout"]
    meta = load(source / "summary.json")
    events = np.load(source / "fsm-event-times.npz")["events"]
    events = events[events[:, 2] <= case["horizon"]]
    wires = split_wire_times(events, 512)
    regimes = jsonl(OUT / f"{name}-regimes.jsonl")
    trials = jsonl(OUT / f"{name}-trials.jsonl")
    assert [r["trial_id"] for r in trials] == list(range(512))
    assert [(r["trial_id"], r["unit"]) for r in regimes] == [(t, u) for t in range(512) for u in "AB"]
    checked = 0
    for unit, row in enumerate(regimes):
        previous = 0
        for segment in row["regimes"]:
            start, end = segment["start"], segment["end"]
            assert start == previous and end > start
            previous = end
            edges = [start + (end - start) * i // 4 for i in range(5)]
            q = [np.diff(np.searchsorted(wires[unit * 6 + i], edges, side="right")).tolist()
                 for i in range(6)]
            bits = [1 if sum(v) >= 4 and sum(n > 0 for n in v) >= 3
                    else 0 if sum(v) <= 2 else None for v in q]
            assert segment["quarter_counts"] == q and segment["vector"] == bits
            readable = None not in bits
            optimal = readable and sum(x*a for x, a in zip(bits, meta["instance"]["a"])) <= meta["instance"]["b"] and sum(x*c for x, c in zip(bits, meta["instance"]["c"])) == case["optimum_value"]
            assert segment["optimal"] == optimal
            checked += 1
        assert previous == case["horizon"]
        last = row["regimes"][-1]
        start = None
        if last["optimal"]:
            start = last["start"]
            for segment in row["regimes"][-2::-1]:
                if not segment["optimal"] or segment["vector"] != last["vector"]:
                    break
                start = segment["start"]
        qualified = start is not None and case["horizon"] - start >= 100000
        episode = row["terminal_episode"]
        assert episode["start"] == start and episode["qualified"] == qualified
        assert trials[unit // 2]["AB"[unit % 2]] == episode
    for row in trials:
        starts = [row[u]["start"] for u in "AB" if row[u]["qualified"]]
        assert row["qualified"] == bool(starts)
        assert row["start"] == (min(starts) if starts else None)
    ids = [r["trial_id"] for r in trials if r["qualified"]]
    assert ids == case["qualified_trial_ids"]
    for point in case["curve"]:
        count = sum(r["qualified"] and r["start"] <= point["step"] for r in trials)
        assert point["trials"] == count and point["fraction"] == count / 512
    breakdown = Counter()
    for t, row in enumerate(trials):
        if row["qualified"]:
            label = "qualified"
        elif any(row[u]["start"] is not None for u in "AB"):
            label = "terminal_optimum_too_recent"
        elif all(regimes[t * 2 + i]["regimes"][-1]["stable_readout"] for i in range(2)):
            label = "both_readable_nonoptimal"
        else:
            label = "includes_unresolved_readout"
        breakdown[label] += 1
    return {"case": name, "passed": True, "trials": 512, "regimes_checked": checked,
            "FSM_events_checked": len(events), "qualified_trials": len(ids),
            "terminal_status_counts": dict(breakdown)}


def status_plot(cases, checks):
    import matplotlib.pyplot as plt
    keys = ["qualified", "terminal_optimum_too_recent", "both_readable_nonoptimal", "includes_unresolved_readout"]
    categories = ["Optimum retained >=100k", "Optimum retained <100k",
                  "Both units readable, nonoptimal", "Includes unresolved readout"]
    colors = ["#0072b2", "#e6ab02", "#a6a6a6", "#ececec"]
    values = np.array([[row["terminal_status_counts"].get(key, 0) for key in keys] for row in checks])
    assert np.all(values.sum(axis=1) == 512)
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    fig.subplots_adjust(left=.23, right=.97, bottom=.31, top=.80)
    left = np.zeros(len(cases))
    for i, (label, color) in enumerate(zip(categories, colors)):
        widths = values[:, i] / 512 * 100
        ax.barh(range(len(cases)), widths, left=left, height=.58, color=color,
                edgecolor="white", linewidth=.6, label=label, hatch="///" if i == 3 else None)
        if i == 0:
            for j, width in enumerate(widths):
                ax.text(width / 2, j, f"{values[j, i]}/512  ({width:.2f}%)", ha="center", va="center", color="white", fontweight="bold")
        left += widths
    ax.set(yticks=range(len(cases)), yticklabels=[f"Instance {c['condition']}, N={c['N']}" for c in cases],
           xlim=(0, 100), xlabel="Fraction of trials (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.suptitle("Terminal output status at 6 million steps", x=.05, y=.96, ha="left", fontsize=17, fontweight="bold")
    fig.text(.05, .87, "512 trials per case  |  Same-unit, same-vector persistence for at least 100,000 steps", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.04, .105), ncol=2, frameon=False, fontsize=9.5)
    fig.text(.05, .025, "Output classifications inferred from FSM events.\nUnresolved or short outputs are not proof of internal-state failure.", fontsize=9, color=".3", linespacing=1.5)
    save(fig, OUT / "terminal-status-6m")


def main():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    config = load(OUT / "config.json")
    sources = verify_sources(config)
    jobs = load(OUT / "readout-source/job-status.json")
    production = load(OUT / "readout-source/production-completion-audit.json")
    assert jobs["passed"] and len(jobs["jobs"]) == 3
    assert all(j["job_state"] == "F" and str(j["Exit_status"]) == "0" for j in jobs["jobs"])
    assert production["passed"] and len(production["conditions"]) == 3
    assert all(c["passed"] and c["current_20_core_files_match"] and c["source_stamp_match"]
               for c in production["conditions"])
    names = [c["id"] for c in config["cases"]]
    cases = [load(OUT / f"{name}.json") for name in names]
    checks = [verify_case(name) for name in names]
    old_n1 = read_verified(OLD / "condition1-N1-6m.json")
    assert cases[0] == old_n1, "Previously published completed N1 result changed"
    legacy_path = ROOT / "docs/experiments/2026-10-08-terminal-retention/condition2-legacy-N1-6m.json"
    legacy = read_verified(legacy_path)
    comparisons = []
    for case in [*cases, legacy]:
        name = case["case"]
        old = read_verified(THREE / f"{name.replace('-6m', '-3m')}.json")
        before, after = set(old["qualified_trial_ids"]), set(case["qualified_trial_ids"])
        comparisons.append({"case": name, "previous_horizon": 3000000, "horizon": 6000000,
                            "previous_qualified": len(before), "qualified": len(after),
                            "previous_fraction": len(before) / 512, "fraction": len(after) / 512,
                            "wilson95": case["curve"][-1]["wilson95"], "retained": len(before & after),
                            "newly_qualified": sorted(after - before), "no_longer_qualified": sorted(before - after),
                            "neither": 512 - len(before | after), "legacy_reference": case["legacy_reset_mismatch"],
                            "interpretation": "Endpoint classification changes, not exact first hits or proven physical loss."})
    before, after = (set(c["qualified_trial_ids"]) for c in cases[:2])
    pairing = {"both": len(before & after), "N1_only": len(before - after),
               "N2_only": len(after - before), "neither": 512 - len(before | after)}
    interim = read_verified(OLD / "condition1-N2-5p3m.json")
    interim_ids = set(interim["qualified_trial_ids"])
    n2_interim_change = {"previous_horizon": 5300000, "horizon": 6000000,
                         "previous_qualified": len(interim_ids), "qualified": len(after),
                         "newly_qualified": sorted(after - interim_ids),
                         "no_longer_qualified": sorted(interim_ids - after)}
    curve_figure(cases[:2], "Instance 1 | Completed 6 million steps", OUT / "condition1-6m")
    curve_figure([cases[2], legacy], "Instance 2 | Completed 6 million steps", OUT / "condition2-6m", legacy=True)
    status_plot(cases, checks)
    for path in OUT.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    comparison = {"versus_3m": comparisons, "condition1_matched_6m_pairing": pairing,
                  "condition1_N2_vs_5p3m": n2_interim_change, "terminal_details": checks}
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    validation = {"passed": True, "source_checks": sources, "checks": checks,
                  "previous_condition1_N1_6m_aggregate_exact": True,
                  "same_policy_as_previous_results": True, "future_events_used": False,
                  "legacy_aggregate_sha256": digest(legacy_path), "script_sha256": digest(__file__),
                  "aggregation_script_sha256": digest(ROOT / "scripts/plot_bca_ip_terminal_retention.py"),
                  "plot_helper_sha256": digest(ROOT / "scripts/plot_bca_ip_progress_summary.py"),
                  "collection_record_sha256": {str(p.relative_to(OUT)): digest(p)
                      for p in sorted((OUT / "readout-source").glob("*.json"))}}
    (OUT / "event-level-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
