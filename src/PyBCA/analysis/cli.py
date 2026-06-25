"""Command line entry points for PyBCA analysis plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .plots import plot_cumulative_events, plot_first_seen_cdf_overlay


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


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
        show=args.show,
    )
    if saved is not None:
        print(saved)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: pybca-plot {cdf,cumulative} ...")
        print()
        print("commands:")
        print("  cdf         Plot first-firing CDF overlays.")
        print("  cumulative  Plot cumulative event histories.")
        return

    command, rest = args[0], args[1:]
    if command == "cdf":
        plot_cdf_main(rest)
        return
    if command == "cumulative":
        plot_cumulative_main(rest)
        return
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
