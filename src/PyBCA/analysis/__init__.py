"""Analysis helpers for PyBCA event-history JSONL files."""

from .io import (
    event_history_for_trial,
    events_to_dict,
    format_sweep_for_trial,
    load_jsonl_trials_with_meta,
)
from .plots import (
    first_seen_histogram,
    plot_cumulative_events,
    plot_first_seen_cdf_overlay,
    plot_first_seen_cdf_by_condition,
    plot_join_fork_accuracy_sweep,
    save_figure,
)

__all__ = [
    "event_history_for_trial",
    "events_to_dict",
    "first_seen_histogram",
    "format_sweep_for_trial",
    "load_jsonl_trials_with_meta",
    "plot_cumulative_events",
    "plot_first_seen_cdf_overlay",
    "plot_first_seen_cdf_by_condition",
    "plot_join_fork_accuracy_sweep",
    "save_figure",
]
