"""Shareable figures from the verified terminal-retention aggregates.

No new readout or classification is performed. Fixed 3M and 6M endpoints are
drawn separately because a retrospective retention curve depends on its end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "docs/experiments/2026-10-11-final-3m"
EARLIER = ROOT / "docs/experiments/2026-10-08-terminal-retention"
COLORS = {"N1": "#0072b2", "N2": "#d55e00", "legacy": "#737373"}


def read_verified(path):
    expected = None
    for line in (path.parent / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.lstrip("*") == path.name:
            expected = digest
            break
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Unverified aggregate: {path}")
    return json.loads(path.read_text())


def save(fig, output):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{ext}"), dpi=220, facecolor="white")
    plt.close(fig)


def curve_figure(cases, title, output, *, legacy=False):
    horizons = {row["horizon"] for row in cases}
    if len(horizons) != 1:
        raise ValueError("Do not overlay retention curves with different fixed endpoints")
    horizon = horizons.pop()
    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    fig.subplots_adjust(left=.115, right=.975, bottom=.29, top=.81)
    for case in cases:
        points = case["curve"]
        if (case["trials"] != 512 or case["minimum_duration"] != 100000 or
                points[-1]["trials"] != case["qualified_trials"] or
                points[-1]["step"] != horizon):
            raise ValueError("Unexpected cohort or retention policy")
        old = case.get("legacy_reset_mismatch", False)
        label = f"{'Legacy ' if old else ''}N={case['N']}{'*' if old else ''}"
        label += f"   {case['qualified_trials']}/512 ({case['qualified_fraction'] * 100:.2f}%)"
        color = COLORS["legacy" if old else f"N{case['N']}"]
        x = np.array([r["step"] for r in points]) / 1e6
        y = np.array([r["fraction"] for r in points]) * 100
        limits = np.array([r["wilson95"] for r in points]) * 100
        ax.step(x, y, where="post", color=color, lw=2.2, ls="--" if old else "-", label=label)
        ax.fill_between(x, limits[:, 0], limits[:, 1], step="post", color=color, alpha=.12)
        ax.plot(x[-1], y[-1], "o", color=color, ms=4, clip_on=False)
    ax.axvspan((horizon - 100000) / 1e6, horizon / 1e6, color=".5", alpha=.10)
    ax.set(xlim=(0, horizon / 1e6), ylim=(0, 100),
           xlabel="Start step of final optimal-output interval (millions)",
           ylabel="Trials retaining an optimum through T (%)")
    ax.grid(alpha=.18)
    fig.suptitle(title, x=.115, y=.965, ha="left", fontsize=17, fontweight="bold")
    fig.text(.115, .88, f"512 trials per case  |  T = {horizon / 1e6:.1f} million CA steps  |  Minimum persistence: 100,000 steps", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.105, .125),
               ncol=len(cases), frameon=False, fontsize=10, columnspacing=2)
    note = "Retrospective: only intervals persisting through T count; the same unit and vector must persist.\n"
    note += "Bands: pointwise 95% Wilson intervals. Persistence is inferred from FSM output events."
    if legacy:
        note += "\n*Legacy reset gains differ from initial Weights; this is not an N-only comparison."
    fig.text(.115, .025, note, fontsize=8.8, color=".3", va="bottom", linespacing=1.5)
    save(fig, output)


def status_figure(comparison, output):
    rows = comparison["rows"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.subplots_adjust(left=.23, right=.97, bottom=.31, top=.8)
    labels = [f"Instance {r['condition']}, N={r['N']}" + (" (legacy*)" if r['legacy_reset_mismatch'] else "") for r in rows]
    values = np.array([[r["qualified_trials"], r["status_counts"]["terminal_optimum_too_recent"],
                        r["neither_unit_terminal_optimal_breakdown"]["both_readable_nonoptimal"],
                        r["neither_unit_terminal_optimal_breakdown"]["at_least_one_unreadable"]]
                       for r in rows])
    if not np.all(values.sum(axis=1) == 512):
        raise ValueError("Terminal classes must partition all 512 trials")
    categories = ["Optimum retained >=100k", "Optimum retained <100k",
                  "Both units readable, nonoptimal", "Includes unresolved readout"]
    colors = ["#0072b2", "#e6ab02", "#a6a6a6", "#ececec"]
    left = np.zeros(len(rows))
    for i, (name, color) in enumerate(zip(categories, colors)):
        width = values[:, i] / 512 * 100
        ax.barh(range(len(rows)), width, left=left, color=color, height=.60,
                edgecolor="white", linewidth=.6, label=name, hatch="///" if i == 3 else None)
        if i == 0:
            for j in range(len(rows)):
                ax.text(width[j] / 2, j, f"{values[j, i]}/512  ({width[j]:.2f}%)",
                        ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left += width
    ax.set(yticks=range(len(rows)), yticklabels=labels, xlim=(0, 100), xlabel="Fraction of trials (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.suptitle("Terminal output status at 3 million steps", x=.05, y=.96,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.05, .87, "512 trials per case  |  All cases use the same 100,000-step terminal persistence criterion", fontsize=10)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower left", bbox_to_anchor=(.04, .12), ncol=2,
               frameon=False, fontsize=9.5)
    fig.text(.05, .025, "Unresolved or short outputs are not proof of internal-state failure.\n"
             "*Legacy Instance 2 has mismatched reset gains; it is a reference, not a matched N=1 control.",
             fontsize=9, color=".3", linespacing=1.5)
    save(fig, output)
    return values.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/experiments/2026-10-13-continuation-6m")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    paths = [FINAL / f"{name}.json" for name in ("condition1-N1-3m", "condition1-N2-3m",
             "condition2-N2-3m", "condition2-legacy-N1-3m")]
    paths += [EARLIER / "condition2-legacy-N1-6m.json", FINAL / "comparison.json"]
    a, b, c, old3, old6, comparison = [read_verified(p) for p in paths]
    curve_figure([a, b], "Instance 1 | N = 1 and N = 2", args.output / "instance1-3m")
    curve_figure([c, old3], "Instance 2 | Current circuit and historical reference",
                 args.output / "instance2-3m", legacy=True)
    curve_figure([old6], "Instance 2 | Historical circuit at 6 million steps",
                 args.output / "instance2-legacy-6m", legacy=True)
    statuses = status_figure(comparison, args.output / "terminal-status-3m")
    record = {"source_files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "classification_changed": False, "future_continuation_results_included": False,
              "fixed_horizons_overlaid": False, "status_counts": statuses,
              "outputs": ["instance1-3m", "instance2-3m", "instance2-legacy-6m", "terminal-status-3m"]}
    (args.output / "figure-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
