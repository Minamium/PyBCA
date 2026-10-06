"""Plot persistent optimum readouts using only the history available at each time.

The prefix segmentation uses the existing dynamic-programming criterion. Future
bins never contribute to the last boundary of an earlier prefix. The curve is
pointwise retention under an operational signal policy, not a first-hit CDF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy

from read_bca_ip_fsm_outputs import decode_terminal, score_unit, split_wire_times


def prefix_last_changes(counts, penalty=20., min_bins=3):
    """Return the last boundary for every prefix; -1 means insufficient history."""
    counts = np.asarray(counts)
    if counts.ndim != 3 or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Require finite nonnegative units/time/wires counts")
    units, n, wires = counts.shape
    if not units or not wires or min_bins < 1 or n < min_bins or penalty < 0:
        raise ValueError("Invalid segmentation settings")
    cumulative = np.concatenate([np.zeros((units, 1, wires)), counts.cumsum(axis=1)], axis=1)
    cost = np.full((units, n+1), np.inf)
    cost[:, 0] = -penalty
    last = np.full((units, n+1), -1, dtype=np.int32)
    unit_ids = np.arange(units)
    for end in range(min_bins, n+1):
        starts = np.r_[0, np.arange(min_bins, end-min_bins+1)]
        amount = cumulative[:, end:end+1] - cumulative[:, starts]
        length = end-starts
        fit = -xlogy(amount, amount/length[None, :, None]).sum(axis=-1)
        scores = cost[:, starts] + fit + penalty
        best = scores.argmin(axis=1)
        cost[:, end] = scores[unit_ids, best]
        last[:, end] = starts[best]
    return last


def wilson_interval(successes, trials, z=1.959963984540054):
    p = np.asarray(successes, dtype=float)/trials
    denominator = 1+z*z/trials
    center = (p+z*z/(2*trials))/denominator
    half = z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return np.maximum(0, center-half), np.minimum(1, center+half)


def read_prefix(wires, last, end, width, trials, instance, optimum, durations=(10_000, 100_000)):
    # 0 = unresolved, 1 = readable nonoptimal, 2 = readable optimum.
    unit_states = np.zeros((len(durations), trials*2), dtype=np.int8)
    for unit, start in enumerate(last[:, end//width]):
        if start < 0:
            continue
        readout = decode_terminal(wires[unit*6:(unit+1)*6], int(start)*width, end,
                                  min_duration=min(durations), max_window=300_000)
        scored = score_unit(readout, instance, optimum)
        for j, duration in enumerate(durations):
            if scored['stable_readout'] and scored['regime_duration'] >= duration:
                unit_states[j, unit] = 2 if scored['optimal'] else 1
    pairs = unit_states.reshape(len(durations), trials, 2)
    return np.where((pairs == 2).any(axis=2), 2,
                    np.where((pairs == 1).all(axis=2), 1, 0)).astype(np.int8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('readout', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sample-step', type=int, default=50_000)
    args = ap.parse_args()
    source, out = args.readout, args.output
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source/'summary.json').read_text())
    trials, horizon = summary['trials'], summary['horizon']
    width, durations = 10_000, (10_000, 100_000)
    if args.sample_step < 30_000 or args.sample_step % width or horizon % args.sample_step:
        raise ValueError('Sampling must align with 10k bins, be at least 30k, and divide horizon')
    events = np.load(source/'fsm-event-times.npz')['events']
    if (events[:, 2] < 1).any() or (events[:, 2] > horizon).any():
        raise ValueError('Event outside available history')
    counts = np.zeros((trials*2, horizon//width, 6), dtype=np.int32)
    np.add.at(counts, (events[:, 0]*2+events[:, 1]//6,
                      (events[:, 2]-1)//width, events[:, 1]%6), 1)
    assert int(counts.sum()) == len(events) == summary['provenance']['fsm_events']
    wires = split_wire_times(events, trials)
    print('Fitting prefixes without future observations', flush=True)
    last = prefix_last_changes(counts)
    times = np.arange(args.sample_step, horizon+1, args.sample_step)
    states = np.empty((len(durations), trials, len(times)), dtype=np.int8)
    rows = []
    for i, end in enumerate(times):
        states[:, :, i] = read_prefix(wires, last, int(end), width, trials,
                                     summary['instance'], summary['optimum_value'], durations)
        row = {'step':int(end)}
        for j, duration in enumerate(durations):
            n = np.bincount(states[j, :, i], minlength=3)
            low, high = wilson_interval(int(n[2]), trials)
            row[str(duration)] = dict(optimal=int(n[2]), readable_nonoptimal=int(n[1]),
                                      unresolved=int(n[0]), wilson95=[float(low), float(high)])
        rows.append(row)
        if end % 1_000_000 == 0:
            print(json.dumps(row), flush=True)
    np.savez_compressed(out/'retention-trial-states.npz', states=states, steps=times,
                        durations=np.array(durations), last_changes=last)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = {'schema':'pybca-prefix-retention-v1', 'trials':trials, 'horizon':horizon,
              'sample_step':args.sample_step, 'durations':list(durations),
              'policy':summary['readout_policy'] | {'min_duration':min(durations)},
              'segmentation':{'penalty':20., 'bin_width':width, 'min_bins':3,
                              'future_observations_used':False},
              'rows':rows, 'notes':[
                  'All trials remain in the denominator, including unresolved readouts.',
                  'Either unit can support success; each trial counts only once.',
                  '95% intervals are pointwise, not a simultaneous confidence band.',
                  'Steps within a trial are correlated, not additional independent trials.',
                  'No first-hit or cumulative-ever curve is inferred from these samples.'],
              'provenance':{'readout_summary_sha256':digest(source/'summary.json'),
                            'event_times_sha256':digest(source/'fsm-event-times.npz'),
                            'script_sha256':digest(__file__)}}
    (out/'retention-curve.json').write_text(json.dumps(result, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(9,5.5), constrained_layout=True)
    x = times/1e6
    n = (states[0] == 2).sum(axis=0)
    low, high = wilson_interval(n, trials)
    ax.fill_between(x, 100*low, 100*high, color='#0072b2', alpha=.18,
                    label='Pointwise 95% Wilson interval')
    ax.plot(x, 100*n/trials, color='#0072b2', lw=2,
            label='Minimum regime duration: 10k updates')
    ax.plot(x, 100*(states[1] == 2).mean(axis=0), color='#d55e00', lw=1.3, ls='--',
            label='Sensitivity: minimum duration 100k')
    for t in (3_000_000, 6_000_000):
        if t in times:
            i = int(np.searchsorted(times,t))
            ax.scatter([x[i]], [100*n[i]/trials], color='#0072b2', zorder=5)
            ax.annotate(f'{n[i]}/{trials} = {100*n[i]/trials:.2f}%',
                        (x[i], 100*n[i]/trials), xytext=(-8, 15), textcoords='offset points',
                        ha='right', fontsize=10)
    ax.set(xlim=(0,horizon/1e6+.05), ylim=(0,100),
           xlabel='Completed CA updates (millions)',
           ylabel='Trials with a confirmed persistent optimum (%)',
           title=f'BCA-IP, Instance 2: optimum retention over time ({trials} trials)')
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    fig.savefig(out/'optimum-retention.png', dpi=200)
    fig.savefig(out/'optimum-retention.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
