"""Summarize certified per-trial first hits without treating censoring as failure.

Input JSONL: one row per independent trial, with trial_id, observed_steps,
first_hit_step (integer or null), and decoder_sha256. A/B circuit blocks belong
to the SAME trial. This module does not infer solutions from output events.
Steps count completed CA updates; an optimum in the initial state is step 0.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def wilson_interval(successes, total, z=1.959963984540054):
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def summarize(rows, horizon, expected_trials):
    if horizon <= 0 or expected_trials <= 0 or len(rows) != expected_trials:
        raise ValueError("Require the complete expected trial roster and a positive horizon")
    seen, decoders, hits = set(), set(), []
    failed = incomplete = 0
    for row in rows:
        trial, observed, hit = row["trial_id"], row["observed_steps"], row["first_hit_step"]
        if type(trial) is not int or not 0 <= trial < 2**64 or trial in seen:
            raise ValueError("Trial IDs must be unique unsigned 64-bit integers")
        if type(observed) is not int or observed < 0:
            raise ValueError("observed_steps must be a non-negative integer")
        if hit is not None and (type(hit) is not int or not 0 <= hit <= observed):
            raise ValueError("first_hit_step must be null or an observed non-negative step")
        decoder = row.get("decoder_sha256", "")
        if not isinstance(decoder, str) or len(decoder) != 64 or any(c not in "0123456789abcdef" for c in decoder):
            raise ValueError("Each trial must identify its confirmed decoder with SHA-256")
        seen.add(trial)
        decoders.add(decoder)
        if hit is not None and hit <= horizon:
            hits.append(hit)
        elif observed >= horizon:
            failed += 1
        else:
            incomplete += 1
    if len(decoders) != 1:
        raise ValueError("Do not combine different optimum definitions/decoders")
    complete = incomplete == 0
    counts = Counter(hits)
    cdf, count = [], 0
    for step in sorted(counts):
        count += counts[step]
        cdf.append({"step": step, "observed_reached_trials": count,
                    "observed_reached_fraction": count/expected_trials})
    return {"schema": "pybca-first-hit-summary-v1", "horizon_steps": horizon,
            "independent_trials": expected_trials, "decoder_sha256": next(iter(decoders)),
            "successes": len(hits), "confirmed_failures_at_horizon": failed,
            "incomplete_without_observed_hit": incomplete, "horizon_outcomes_complete": complete,
            "failure_rate_at_horizon": failed/expected_trials if complete else None,
            "failure_rate_wilson_95": wilson_interval(failed, expected_trials) if complete else None,
            "failure_rate_bounds_with_incomplete": [failed/expected_trials, (failed+incomplete)/expected_trials],
            "mean_hit_steps_among_observed_successes": statistics.mean(hits) if hits else None,
            "median_hit_steps_among_observed_successes": statistics.median(hits) if hits else None,
            "sd_hit_steps_among_observed_successes": statistics.stdev(hits) if len(hits) > 1 else None,
            "restricted_mean_steps_at_horizon": ((sum(hits)+failed*horizon)/expected_trials if complete else None),
            "unrestricted_mean_hit_steps": statistics.mean(hits) if len(hits) == expected_trials else None,
            "observed_reached_fraction_curve": cdf}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials_jsonl", type=Path)
    parser.add_argument("--horizon", type=int, default=3_000_000)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trials_jsonl.read_text().splitlines() if line.strip()]
    summary = summarize(rows, args.horizon, args.expected_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "observed_reached_fraction_curve"}, indent=2))


if __name__ == "__main__":
    main()
