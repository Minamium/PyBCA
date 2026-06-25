"""Command line entry points for PyBCA analysis plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .plots import plot_cumulative_events, plot_first_seen_cdf_overlay, plot_join_fork_accuracy_sweep


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_figsize(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must be formatted like '8,5' or '8x5'.")
    return float(parts[0]), float(parts[1])


def plot_cdf_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot first-firing CDF overlays from PyBCA JSONL files.")
    parser.add_argument("files", nargs="+", help="JSONL files to overlay.")
    parser.add_argument("--events", help="Comma-separated event names. Defaults to common events.")
    parser.add_argument("--x-max", type=int, default=None)
    parser.add_argument("--bin-size", type=int, default=1)
    parser.add_argument("--title", default="Event first-firing CDF")
    parser.add_argument("--output-dir", default=".", help="Directory for the saved plot. Defaults to cwd.")
    parser.add_argument("--filename", default="first_seen_cdf.png")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--figsize", type=_parse_figsize, help="Figure size, e.g. '8,5'.")
    parser.add_argument("--font-size", type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    _fig, saved = plot_first_seen_cdf_overlay(
        files=args.files,
        events=_split_csv(args.events),
        x_max=args.x_max,
        bin_size=args.bin_size,
        title=args.title,
        output_dir=args.output_dir,
        filename=args.filename,
        dpi=args.dpi,
        figsize=args.figsize,
        font_size=args.font_size,
        show=args.show,
    )
    if saved is not None:
        print(saved)


def plot_cumulative_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot cumulative event histories from one PyBCA JSONL file.")
    parser.add_argument("file", help="JSONL file.")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--events", help="Comma-separated event names. Defaults to all events in the trial.")
    parser.add_argument("--xlim", nargs=2, type=int, metavar=("MIN", "MAX"))
    parser.add_argument("--title", default="Event Firing History")
    parser.add_argument("--output-dir", default=".", help="Directory for the saved plot. Defaults to cwd.")
    parser.add_argument("--filename", default=None)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--figsize", type=_parse_figsize, help="Figure size, e.g. '12,6'.")
    parser.add_argument("--font-size", type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    filename = args.filename
    if filename is None:
        stem = Path(args.file).stem
        filename = f"{stem}_trial{args.trial}_cumulative.png"

    _fig, saved = plot_cumulative_events(
        event_source=args.file,
        event_names=_split_csv(args.events),
        xlim=tuple(args.xlim) if args.xlim else None,
        trial=args.trial,
        title=args.title,
        output_dir=args.output_dir,
        filename=filename,
        dpi=args.dpi,
        figsize=args.figsize or (12, 6),
        font_size=args.font_size,
        show=args.show,
    )
    if saved is not None:
        print(saved)


def plot_accuracy_sweep_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot Join/Fork accuracy sweeps with condition as color and probability as marker."
    )
    parser.add_argument("root_dir", help="Sweep result directory containing p*/{join,fork}_*_P*.jsonl files.")
    parser.add_argument("--kind", choices=("join", "fork"), required=True)
    parser.add_argument("--event", help="Event name. Defaults to output for join and output_1 for fork.")
    parser.add_argument("--conditions", help="Comma-separated condition labels, e.g. P2,P1,P0.")
    parser.add_argument("--x-max", type=int, default=300)
    parser.add_argument("--bin-size", type=int, default=1)
    parser.add_argument("--title")
    parser.add_argument("--output-dir", default=".", help="Directory for the saved plot. Defaults to cwd.")
    parser.add_argument("--filename")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--figsize", type=_parse_figsize, default=(8.0, 4.8), help="Figure size, e.g. '8,5'.")
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--legend-inside", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    filename = args.filename
    if filename is None:
        filename = f"{args.kind}_accuracy_sweep.png"

    plot_kwargs = {
        "x_max": args.x_max,
        "bin_size": args.bin_size,
        "output_dir": args.output_dir,
        "filename": filename,
        "dpi": args.dpi,
        "figsize": args.figsize,
        "font_size": args.font_size,
        "legend_outside": not args.legend_inside,
        "show": args.show,
    }
    if args.title is not None:
        plot_kwargs["title"] = args.title

    _fig, saved = plot_join_fork_accuracy_sweep(
        root_dir=args.root_dir,
        kind=args.kind,
        event=args.event,
        conditions=_split_csv(args.conditions),
        **plot_kwargs,
    )
    if saved is not None:
        print(saved)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: pybca-plot {cdf,cumulative,accuracy-sweep} ...")
        print()
        print("commands:")
        print("  cdf             Plot first-firing CDF overlays.")
        print("  cumulative      Plot cumulative event histories.")
        print("  accuracy-sweep  Plot Join/Fork accuracy sweeps.")
        return

    command, rest = args[0], args[1:]
    if command == "cdf":
        plot_cdf_main(rest)
        return
    if command == "cumulative":
        plot_cumulative_main(rest)
        return
    if command == "accuracy-sweep":
        plot_accuracy_sweep_main(rest)
        return
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
