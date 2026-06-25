"""Plotting helpers for PyBCA event-history JSONL files."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import event_history_for_trial, events_to_dict, format_sweep_for_trial, load_jsonl_trials_with_meta


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig: Any, filename: str | Path, output_dir: str | Path = ".", dpi: int = 120) -> Path:
    """Save a Matplotlib figure under ``output_dir`` and return the written path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def first_seen_histogram(
    filepath: str | Path,
    target_events: Iterable[str] | None = None,
    bin_size: int = 1,
) -> tuple[dict[str, Counter[int]], int, int, list[str]]:
    """Build first-firing-step histograms for selected events in one JSONL file."""
    hist: dict[str, Counter[int]] = {}
    total_trials = 0
    max_seen_step = 0
    all_event_names: set[str] = set()
    target = set(target_events) if target_events is not None else None

    _meta, trials = load_jsonl_trials_with_meta(filepath)
    for rec in trials:
        total_trials += 1
        evdict = events_to_dict(rec.get("events", {}))
        all_event_names.update(evdict)
        names = target if target is not None else evdict.keys()

        for name in names:
            steps = evdict.get(str(name), [])
            if not steps:
                continue
            first = int(min(steps))
            if first < 0:
                continue
            max_seen_step = max(max_seen_step, first)
            hist.setdefault(str(name), Counter())[first // bin_size] += 1

    return hist, total_trials, max_seen_step, sorted(all_event_names)


def _counter_to_cdf(
    counter: Counter[int],
    total_trials: int,
    x_max: int,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    nbins = x_max // bin_size + 1
    counts = np.zeros(nbins, dtype=np.int64)
    for bin_idx, count in counter.items():
        if 0 <= bin_idx < nbins:
            counts[bin_idx] += int(count)

    csum = counts.cumsum()
    x = np.arange(nbins) * bin_size
    y = csum / max(total_trials, 1)
    return x, y, int(csum[-1])


def plot_first_seen_cdf_overlay(
    files: Iterable[str | Path],
    events: Iterable[str] | None = None,
    x_max: int | None = None,
    bin_size: int = 1,
    title: str = "Event first-firing CDF",
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    fig_width_per_col: float = 5.0,
    fig_height_per_row: float = 3.2,
    dpi: int = 120,
    x_label: str = "step",
    y_label: str = "fraction of trials",
    legend_loc: str = "lower right",
    legend_fontsize: int = 9,
    suptitle_size: int = 14,
    axes_title_size: int = 12,
    output_dir: str | Path = ".",
    filename: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Path | None]:
    """Plot first-firing CDF overlays across files, optionally saving the figure."""
    plt = _pyplot()
    file_list = [Path(p) for p in files]
    if not file_list:
        raise ValueError("At least one JSONL file is required.")

    event_filter = list(events) if events is not None else None
    per_file = []
    global_max_seen = 0
    event_name_sets: list[set[str]] = []

    for path in file_list:
        hist, total, mx, names = first_seen_histogram(path, target_events=event_filter, bin_size=bin_size)
        per_file.append((hist, total, mx, names, path))
        global_max_seen = max(global_max_seen, mx)
        event_name_sets.append(set(names if event_filter is None else event_filter))

    if event_filter is None:
        common = set.intersection(*event_name_sets) if event_name_sets else set()
        events_to_plot = sorted(common)
    else:
        events_to_plot = event_filter

    if not events_to_plot:
        raise ValueError("No plottable event names were found.")
    if x_max is None:
        x_max = int(global_max_seen)

    event_count = len(events_to_plot)
    if ncols is None:
        ncols = min(event_count, 2 if event_count > 1 else 1)
    nrows = math.ceil(event_count / ncols)
    if figsize is None:
        figsize = (fig_width_per_col * ncols, fig_height_per_row * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    fig.suptitle(title, fontsize=suptitle_size)

    for idx, event_name in enumerate(events_to_plot):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for hist, total, _mx, _names, path in per_file:
            counter = hist.get(event_name, Counter())
            x, y, seen = _counter_to_cdf(counter, total, x_max, bin_size)
            ax.plot(x, y, label=f"{path.name}  (seen {seen}/{total})")

        ax.set_title(event_name, fontsize=axes_title_size)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=True)

    for idx in range(event_count, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row][col])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    saved_path = save_figure(fig, filename, output_dir=output_dir, dpi=dpi) if filename is not None else None
    if show:
        plt.show()
    return fig, saved_path


def plot_cumulative_events(
    event_source: str | Path | dict[str, list[int]],
    event_names: Iterable[str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str = "Event Firing History",
    xlim: tuple[int, int] | None = None,
    trial: int = 0,
    output_dir: str | Path = ".",
    filename: str | Path | None = None,
    dpi: int = 120,
    show: bool = False,
) -> tuple[Any, Path | None]:
    """Plot cumulative firing counts for one trial or for a provided event dict."""
    plt = _pyplot()
    meta = None

    if isinstance(event_source, (str, Path)):
        meta, _trials = load_jsonl_trials_with_meta(event_source)
        event_dict = event_history_for_trial(event_source, trial=trial)
        sweep_text = format_sweep_for_trial(meta, trial)
        if sweep_text is None:
            plot_title = f"{title} (trial={trial})"
        else:
            plot_title = f"{title} (trial={trial} | {sweep_text})"
    elif isinstance(event_source, dict):
        event_dict = event_source
        plot_title = title
    else:
        raise TypeError("event_source must be a filepath or an event dictionary.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    names = list(event_names) if event_names is not None else list(event_dict.keys())

    for event_name in names:
        if event_name not in event_dict:
            print(f"Warning: '{event_name}' not found")
            continue
        steps = event_dict[event_name]
        if not steps:
            continue
        steps_sorted = np.sort(np.asarray(steps, dtype=np.int64))
        cumulative = np.arange(1, len(steps_sorted) + 1, dtype=np.int64)
        ax.step(steps_sorted, cumulative, where="post", label=event_name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Firings")
    ax.set_title(plot_title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()
    saved_path = save_figure(fig, filename, output_dir=output_dir, dpi=dpi) if filename is not None else None
    if show:
        plt.show()
    return fig, saved_path
