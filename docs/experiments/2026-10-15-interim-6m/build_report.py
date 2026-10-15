"""Verify event-level readouts and draw the 15 October fixed-endpoint report."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_bca_ip_progress_summary import curve_figure, read_verified
from read_bca_ip_fsm_outputs import split_wire_times


def load(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_downloads_and_prefixes():
    old_roots = {
        "condition1_N1": ROOT / "docs/experiments/2026-10-07-variants/status-20261008/condition1_N1-3m/readout",
        "condition1_N2": ROOT / "docs/experiments/2026-10-11-final-3m/readout-source/condition1_N2/readout",
    }
    reports = []
    for variant, old_root in old_roots.items():
        directory = OUT / "readout-source" / variant
        snapshot = load(directory / "snapshot.json")
        assert snapshot["passed"] and snapshot["trials"] == 512
        assert snapshot["snapshot_script_sha256"] == digest(ROOT / "scripts/snapshot_bca_ip_readout.py")
        assert snapshot["extraction"]["decoder_sha256"] == digest(ROOT / "scripts/read_bca_ip_fsm_outputs.py")
        for path, expected in snapshot["files"].items():
            assert digest(directory / path) == expected
        new = np.load(directory / "readout/fsm-event-times.npz")
        old = np.load(old_root / "fsm-event-times.npz")
        for kind in ("events", "resets"):
            prefix = new[kind][new[kind][:, 2] <= 3000000]
            assert np.array_equal(prefix, old[kind]), (variant, kind)
        reports.append({"variant": variant, "passed": True,
                        "files_sha256_verified": len(snapshot["files"]),
                        "history_chunks_SHA256_verified": snapshot["extraction"]["checked_chunks"],
                        "prior_3m_FSM_and_reset_prefix_exact": True})
    meta = load(OUT / "readout-source/condition1_N1/readout/summary.json")
    audit = load(OUT / "condition1_N1-completion-audit.json")
    assert audit["passed"] and audit["steps"] == 6000000
    assert audit["events_by_family"]["FSM"] == meta["provenance"]["fsm_events"]
    assert audit["events_by_family"]["reset"] == meta["provenance"]["reset_set_events"]
    return reports


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


def main():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    names = ["condition1-N1-6m", "condition1-N1-5p3m", "condition1-N2-5p3m"]
    cases = {name: load(OUT / f"{name}.json") for name in names}
    downloads = verify_downloads_and_prefixes()
    checks = [verify_case(name) for name in names]
    previous = ROOT / "docs/experiments/2026-10-11-final-3m"
    comparisons = []
    for name, case in cases.items():
        old = read_verified(previous / f"condition1-N{case['N']}-3m.json")
        before, after = set(old["qualified_trial_ids"]), set(case["qualified_trial_ids"])
        comparisons.append({"case": name, "previous_horizon": 3000000,
                            "horizon": case["horizon"], "previous_qualified": len(before),
                            "qualified": len(after), "retained": len(before & after),
                            "newly_qualified": sorted(after - before),
                            "no_longer_qualified": sorted(before - after),
                            "neither": len(set(range(512)) - (before | after)),
                            "interpretation": "Change of endpoint classification, not exact first hits or proven physical loss."})
    curve_figure([cases[names[1]], cases[names[2]]],
                 "Instance 1 | Interim comparison at 5.3 million steps", OUT / "condition1-matched-5p3m")
    curve_figure([cases[names[0]]], "Instance 1, N = 1 | Completed 6 million steps",
                 OUT / "condition1-N1-completed-6m")
    old2 = read_verified(previous / "condition2-legacy-N1-3m.json")
    n2 = read_verified(previous / "condition2-N2-3m.json")
    curve_figure([n2, old2], "Instance 2 | Available results at 3 million steps",
                 OUT / "condition2-unchanged-3m", legacy=True)
    old6 = read_verified(ROOT / "docs/experiments/2026-10-08-terminal-retention/condition2-legacy-N1-6m.json")
    curve_figure([old6], "Instance 2 | Historical circuit at 6 million steps",
                 OUT / "condition2-legacy-6m", legacy=True)
    for path in OUT.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    pairing = {"both": len(set(cases[names[1]]["qualified_trial_ids"]) & set(cases[names[2]]["qualified_trial_ids"])),
               "N1_only": len(set(cases[names[1]]["qualified_trial_ids"]) - set(cases[names[2]]["qualified_trial_ids"])),
               "N2_only": len(set(cases[names[2]]["qualified_trial_ids"]) - set(cases[names[1]]["qualified_trial_ids"])),
               "neither": 512 - len(set(cases[names[1]]["qualified_trial_ids"]) | set(cases[names[2]]["qualified_trial_ids"]))}
    (OUT / "comparison.json").write_text(json.dumps({"versus_3m": comparisons,
        "matched_5p3m_pairing": pairing, "terminal_details": checks}, indent=2) + "\n")
    (OUT / "event-level-validation.json").write_text(json.dumps({"passed": True,
        "checks": checks, "downloads_and_prefixes": downloads, "script_sha256": digest(__file__),
        "same_policy_as_previous_results": True, "future_events_used": False}, indent=2) + "\n")
    print(json.dumps({"comparison": comparisons, "matched_pairing": pairing, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
