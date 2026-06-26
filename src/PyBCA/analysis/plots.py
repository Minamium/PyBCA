"""Plotting helpers for PyBCA event-history JSONL files."""

from __future__ import annotations

import math
import re
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


def _first_seen_histogram_with_meta(
    filepath: str | Path,
    target_events: Iterable[str] | None = None,
    bin_size: int = 1,
) -> tuple[dict[str, Any] | None, dict[str, Counter[int]], int, int, list[str]]:
    hist: dict[str, Counter[int]] = {}
    total_trials = 0
    max_seen_step = 0
    all_event_names: set[str] = set()
    target = set(target_events) if target_events is not None else None

    meta, trials = load_jsonl_trials_with_meta(filepath)
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

    return meta, hist, total_trials, max_seen_step, sorted(all_event_names)


def first_seen_histogram(
    filepath: str | Path,
    target_events: Iterable[str] | None = None,
    bin_size: int = 1,
) -> tuple[dict[str, Counter[int]], int, int, list[str]]:
    """Build first-firing-step histograms for selected events in one JSONL file."""
    _meta, hist, total_trials, max_seen_step, all_event_names = _first_seen_histogram_with_meta(
        filepath,
        target_events=target_events,
        bin_size=bin_size,
    )
    return hist, total_trials, max_seen_step, all_event_names


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


def _apply_axis_font_sizes(
    ax: Any,
    label_fontsize: int | None = None,
    tick_labelsize: int | None = None,
) -> None:
    if label_fontsize is not None:
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", labelsize=tick_labelsize)


def _format_probability(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+03d}"


def _probability_from_meta(
    meta: dict[str, Any] | None,
    aliases: Iterable[str] | None = None,
) -> float | None:
    if not isinstance(meta, dict):
        return None

    sweep = meta.get("probability_sweep") or meta.get("trial_constant_sweep")
    if not isinstance(sweep, dict):
        return None

    alias_order = list(aliases) if aliases is not None else list(sweep)
    for alias in alias_order:
        info = sweep.get(alias)
        if not isinstance(info, dict):
            continue
        probs = info.get("prob_by_trial")
        if isinstance(probs, list) and probs:
            try:
                return float(probs[0])
            except Exception:
                pass
        if "base" in info:
            try:
                return float(info["base"])
            except Exception:
                pass

    return None


def _condition_from_path(path: Path) -> tuple[str | None, str | None, str | None]:
    match = re.match(r"^(?P<kind>join|fork)_(?P<tag>.+)_(?P<condition>P\d+)$", path.stem)
    if match is None:
        return None, None, None
    return match.group("kind"), match.group("tag"), match.group("condition")


def _global_probability_from_tag(
    tag: str | None,
    tag_map: dict[str, float] | None = None,
) -> float | None:
    if tag is None:
        return None
    if tag_map is not None and tag in tag_map:
        return float(tag_map[tag])

    # Convention used in Join/Fork calibration jobs:
    #   g5e1 -> 5e-1 = 0.5
    #   g1e1 -> 1e-1 = 0.1
    #   g5e2 -> 5e-2 = 0.05
    match = re.match(r"^g(?P<mantissa>\d+(?:p\d+)?)e(?P<exponent>\d+)$", tag)
    if match is None:
        return None
    mantissa = float(match.group("mantissa").replace("p", "."))
    exponent = int(match.group("exponent"))
    return mantissa * 10 ** (-exponent)


def _global_probability_from_path(
    path: Path,
    tag: str | None,
    tag_map: dict[str, float] | None = None,
) -> float | None:
    candidates: list[str] = []
    if tag is not None:
        candidates.append(tag.split("_", 1)[0])
    candidates.append(path.parent.name.split("_", 1)[0])
    candidates.append(path.parent.parent.name.split("_", 1)[0])

    for candidate in candidates:
        probability = _global_probability_from_tag(candidate, tag_map=tag_map)
        if probability is not None:
            return probability
    return None


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
    font_size: int | None = None,
    label_fontsize: int | None = None,
    tick_labelsize: int | None = None,
    output_dir: str | Path = ".",
    filename: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Path | None]:
    """Plot first-firing CDF overlays across files, optionally saving the figure."""
    plt = _pyplot()
    if font_size is not None:
        if legend_fontsize == 9:
            legend_fontsize = font_size
        if suptitle_size == 14:
            suptitle_size = font_size + 2
        if axes_title_size == 12:
            axes_title_size = font_size
        if label_fontsize is None:
            label_fontsize = font_size
        if tick_labelsize is None:
            tick_labelsize = font_size

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
        _apply_axis_font_sizes(ax, label_fontsize=label_fontsize, tick_labelsize=tick_labelsize)
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


def plot_first_seen_cdf_by_condition(
    files: Iterable[str | Path],
    event: str,
    conditions: Iterable[str] | None = None,
    probability_aliases: Iterable[str] | None = None,
    x_max: int | None = None,
    bin_size: int = 1,
    title: str = "First-firing probability",
    figsize: tuple[float, float] = (8.0, 4.8),
    dpi: int = 120,
    font_size: int = 12,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
    tick_labelsize: int | None = None,
    legend_fontsize: int | None = None,
    x_label: str = "step",
    y_label: str = "fraction of trials",
    condition_colors: dict[str, str] | None = None,
    condition_order: Iterable[str] | None = None,
    marker_cycle: Iterable[str] = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">"),
    markevery: int | tuple[int, int] | None = 30,
    markersize: float = 5.0,
    linewidth: float = 1.8,
    condition_legend_title: str = "condition",
    probability_legend_title: str = "error probability",
    legend_outside: bool = True,
    output_dir: str | Path = ".",
    filename: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Path | None]:
    """Plot CDFs with input condition as color and error probability as marker."""
    plt = _pyplot()
    file_list = [Path(p) for p in files]
    if not file_list:
        raise ValueError("At least one JSONL file is required.")

    condition_filter = set(conditions) if conditions is not None else None
    records: list[dict[str, Any]] = []
    global_max_seen = 0
    for path in file_list:
        _kind, tag, condition = _condition_from_path(path)
        if condition is None:
            continue
        if condition_filter is not None and condition not in condition_filter:
            continue

        meta, hist, total, max_seen, _names = _first_seen_histogram_with_meta(
            path,
            target_events=[event],
            bin_size=bin_size,
        )
        probability = _probability_from_meta(meta, aliases=probability_aliases)
        records.append(
            {
                "path": path,
                "tag": tag,
                "condition": condition,
                "probability": probability,
                "counter": hist.get(event, Counter()),
                "total": total,
            }
        )
        global_max_seen = max(global_max_seen, max_seen)

    if not records:
        raise ValueError("No matching Join/Fork JSONL files were found.")
    if x_max is None:
        x_max = int(global_max_seen)

    if condition_order is None:
        condition_order = ("P2", "P1", "P0", "P3")
    order_map = {condition: idx for idx, condition in enumerate(condition_order)}
    records.sort(
        key=lambda rec: (
            order_map.get(str(rec["condition"]), len(order_map)),
            float("inf") if rec["probability"] is None else float(rec["probability"]),
            str(rec["tag"]),
        )
    )

    default_colors = {
        "P2": "tab:blue",
        "P1": "tab:orange",
        "P0": "tab:green",
        "P3": "tab:red",
    }
    color_map = {**default_colors, **(condition_colors or {})}

    unique_probabilities = sorted({rec["probability"] for rec in records if rec["probability"] is not None})
    marker_values = list(marker_cycle)
    marker_map = {
        probability: marker_values[idx % len(marker_values)]
        for idx, probability in enumerate(unique_probabilities)
    }

    if title_fontsize is None:
        title_fontsize = font_size + 2
    if label_fontsize is None:
        label_fontsize = font_size
    if tick_labelsize is None:
        tick_labelsize = font_size
    if legend_fontsize is None:
        legend_fontsize = max(font_size - 1, 6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for rec in records:
        probability = rec["probability"]
        marker = marker_map.get(probability, "o")
        condition = str(rec["condition"])
        color = color_map.get(condition, None)
        x, y, _seen = _counter_to_cdf(rec["counter"], int(rec["total"]), x_max, bin_size)
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=markersize,
            linewidth=linewidth,
            label=f"{condition}, p={_format_probability(probability)}",
        )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_axis_font_sizes(ax, label_fontsize=label_fontsize, tick_labelsize=tick_labelsize)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    present_conditions = []
    for rec in records:
        condition = str(rec["condition"])
        if condition not in present_conditions:
            present_conditions.append(condition)

    condition_handles = [
        Line2D([0], [0], color=color_map.get(condition, "black"), lw=linewidth, label=condition)
        for condition in present_conditions
    ]
    probability_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=marker_map[probability],
            linestyle="None",
            markersize=markersize + 1,
            label=f"p={_format_probability(probability)}",
        )
        for probability in unique_probabilities
    ]

    if legend_outside:
        legend1 = ax.legend(
            handles=condition_handles,
            title=condition_legend_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=probability_handles,
            title=probability_legend_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            frameon=True,
        )
    else:
        legend1 = ax.legend(
            handles=condition_handles,
            title=condition_legend_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="upper left",
            frameon=True,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=probability_handles,
            title=probability_legend_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="lower right",
            frameon=True,
        )

    fig.tight_layout()
    saved_path = save_figure(fig, filename, output_dir=output_dir, dpi=dpi) if filename is not None else None
    if show:
        plt.show()
    return fig, saved_path


def plot_join_fork_accuracy_sweep(
    root_dir: str | Path,
    kind: str,
    event: str | None = None,
    conditions: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, Path | None]:
    """Scan a Join/Fork accuracy-sweep directory and plot condition/probability CDFs."""
    root = Path(root_dir)
    if kind not in {"join", "fork"}:
        raise ValueError("kind must be 'join' or 'fork'.")

    if event is None:
        event = "output" if kind == "join" else "output_1"
    if conditions is None:
        conditions = ("P2", "P1", "P0") if kind == "join" else ("P1", "P0")

    probability_aliases = kwargs.pop("probability_aliases", None)
    if probability_aliases is None:
        probability_aliases = ("join_err_0_input",) if kind == "join" else ("fork_err_0_input",)

    files = sorted(root.glob(f"p*/{kind}_*_P*.jsonl"))
    if not files:
        files = sorted(root.glob(f"{kind}_*_P*.jsonl"))

    return plot_first_seen_cdf_by_condition(
        files,
        event=event,
        conditions=conditions,
        probability_aliases=probability_aliases,
        title=kwargs.pop("title", f"{kind.capitalize()} firing probability"),
        **kwargs,
    )


def plot_join_fork_global_calibration_sweep(
    root_dir: str | Path,
    kind: str,
    event: str | None = None,
    conditions: Iterable[str] | None = None,
    global_probability_tags: dict[str, float] | None = None,
    global_probability_order: Iterable[float] | None = None,
    probability_aliases: Iterable[str] | None = None,
    x_max: int | None = None,
    bin_size: int = 1,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 120,
    font_size: int = 12,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
    tick_labelsize: int | None = None,
    legend_fontsize: int | None = None,
    x_label: str = "step",
    y_label: str = "fraction of trials",
    condition_colors: dict[str, str] | None = None,
    condition_order: Iterable[str] | None = None,
    marker_cycle: Iterable[str] = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">"),
    markevery: int | tuple[int, int] | None = 60,
    markersize: float = 5.0,
    linewidth: float = 1.8,
    condition_legend_title: str = "condition",
    probability_legend_title: str = "error probability",
    global_title_prefix: str = "global_prob",
    output_dir: str | Path = ".",
    filename: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Path | None]:
    """Plot Join/Fork accuracy sweeps faceted by global probability.

    The expected layout is the one produced by the global calibration jobs::

        root_dir/g5e1_p1e5/join_g5e1_p1e5_P0.jsonl
        root_dir/g5e1_p1e5/join_g5e1_p1e5_P1.jsonl
        root_dir/g5e1_p1e5/join_g5e1_p1e5_P2.jsonl

    Global probability tags such as ``g5e1`` are decoded as ``5e-1``.  A custom
    mapping can be supplied through ``global_probability_tags``.
    """
    plt = _pyplot()
    root = Path(root_dir)
    if kind not in {"join", "fork"}:
        raise ValueError("kind must be 'join' or 'fork'.")

    if event is None:
        event = "output" if kind == "join" else "output_1"
    if conditions is None:
        conditions = ("P2", "P1", "P0") if kind == "join" else ("P1", "P0")
    if condition_order is None:
        condition_order = tuple(conditions)
    if probability_aliases is None:
        probability_aliases = ("join_err_0_input",) if kind == "join" else ("fork_err_0_input",)

    files = sorted(root.glob(f"**/{kind}_*_P*.jsonl"))
    if not files:
        raise ValueError(f"No {kind} JSONL files were found under {root}.")

    condition_filter = set(conditions)
    records: list[dict[str, Any]] = []
    global_max_seen = 0
    for path in files:
        _kind, tag, condition = _condition_from_path(path)
        if condition is None or condition not in condition_filter:
            continue
        global_probability = _global_probability_from_path(
            path,
            tag,
            tag_map=global_probability_tags,
        )
        if global_probability is None:
            continue

        meta, hist, total, max_seen, _names = _first_seen_histogram_with_meta(
            path,
            target_events=[event],
            bin_size=bin_size,
        )
        records.append(
            {
                "path": path,
                "tag": tag,
                "condition": condition,
                "global_probability": global_probability,
                "probability": _probability_from_meta(meta, aliases=probability_aliases),
                "counter": hist.get(event, Counter()),
                "total": total,
            }
        )
        global_max_seen = max(global_max_seen, max_seen)

    if not records:
        raise ValueError("No matching global calibration JSONL files were found.")
    if x_max is None:
        x_max = int(global_max_seen)

    if global_probability_order is None:
        global_values = sorted({float(rec["global_probability"]) for rec in records}, reverse=True)
    else:
        global_values = [float(value) for value in global_probability_order]
    if not global_values:
        raise ValueError("No global probability values were found.")

    unique_probabilities = sorted({rec["probability"] for rec in records if rec["probability"] is not None})
    marker_values = list(marker_cycle)
    marker_map = {
        probability: marker_values[idx % len(marker_values)]
        for idx, probability in enumerate(unique_probabilities)
    }

    # Match the physical-system figures by default: P2 green, P1 red, P0 blue.
    default_colors = {
        "P2": "tab:green",
        "P1": "tab:red",
        "P0": "tab:blue",
        "P3": "tab:purple",
    }
    color_map = {**default_colors, **(condition_colors or {})}
    condition_order_tuple = tuple(condition_order)
    order_map = {condition: idx for idx, condition in enumerate(condition_order_tuple)}

    if title_fontsize is None:
        title_fontsize = font_size + 2
    if label_fontsize is None:
        label_fontsize = font_size
    if tick_labelsize is None:
        tick_labelsize = font_size
    if legend_fontsize is None:
        legend_fontsize = max(font_size - 1, 6)
    if figsize is None:
        figsize = (4.7 * len(global_values), 4.8)
    if title is None:
        title = f"{kind.capitalize()} firing probability by global probability"

    fig, axes = plt.subplots(1, len(global_values), figsize=figsize, dpi=dpi, sharey=True, squeeze=False)
    axis_list = list(axes[0])
    for ax, global_probability in zip(axis_list, global_values):
        subset = [
            rec
            for rec in records
            if math.isclose(float(rec["global_probability"]), global_probability)
        ]
        subset.sort(
            key=lambda rec: (
                order_map.get(str(rec["condition"]), len(order_map)),
                float("inf") if rec["probability"] is None else float(rec["probability"]),
                str(rec["tag"]),
            )
        )

        for rec in subset:
            probability = rec["probability"]
            condition = str(rec["condition"])
            x, y, _seen = _counter_to_cdf(rec["counter"], int(rec["total"]), x_max, bin_size)
            ax.plot(
                x,
                y,
                color=color_map.get(condition, None),
                marker=marker_map.get(probability, "o"),
                markevery=markevery,
                markersize=markersize,
                linewidth=linewidth,
                label=f"{condition}, p={_format_probability(probability)}",
            )

        ax.set_title(f"{global_title_prefix}={global_probability:g}", fontsize=title_fontsize)
        ax.set_xlabel(x_label)
        _apply_axis_font_sizes(ax, label_fontsize=label_fontsize, tick_labelsize=tick_labelsize)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    axis_list[0].set_ylabel(y_label)
    fig.suptitle(title, fontsize=title_fontsize + 1)

    from matplotlib.lines import Line2D

    present_conditions: list[str] = []
    for rec in records:
        condition = str(rec["condition"])
        if condition not in present_conditions:
            present_conditions.append(condition)
    present_conditions.sort(key=lambda condition: order_map.get(condition, len(order_map)))

    condition_handles = [
        Line2D([0], [0], color=color_map.get(condition, "black"), lw=linewidth, label=condition)
        for condition in present_conditions
    ]
    probability_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=marker_map[probability],
            linestyle="None",
            markersize=markersize + 1,
            label=f"p={_format_probability(probability)}",
        )
        for probability in unique_probabilities
    ]

    last_axis = axis_list[-1]
    legend1 = last_axis.legend(
        handles=condition_handles,
        title=condition_legend_title,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    last_axis.add_artist(legend1)
    last_axis.legend(
        handles=probability_handles,
        title=probability_legend_title,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
    font_size: int | None = None,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
    tick_labelsize: int | None = None,
    legend_fontsize: int | None = None,
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

    if font_size is not None:
        if title_fontsize is None:
            title_fontsize = font_size + 2
        if label_fontsize is None:
            label_fontsize = font_size
        if tick_labelsize is None:
            tick_labelsize = font_size
        if legend_fontsize is None:
            legend_fontsize = font_size

    ax.set_xlabel("Step", fontsize=label_fontsize)
    ax.set_ylabel("Cumulative Firings", fontsize=label_fontsize)
    ax.set_title(plot_title, fontsize=title_fontsize)
    _apply_axis_font_sizes(ax, label_fontsize=label_fontsize, tick_labelsize=tick_labelsize)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()
    saved_path = save_figure(fig, filename, output_dir=output_dir, dpi=dpi) if filename is not None else None
    if show:
        plt.show()
    return fig, saved_path
