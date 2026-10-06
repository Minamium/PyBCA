"""Build paper conditions 1/N=1, 1/N=2 and 2/N=2 with matched reset wiring.

The archived condition-1 map is reconstructed from a checked-in coordinate
patch. The current BCA-IP.yaml is never edited. These maps require the generated
events file because widening the reset corridor moves the TD/reset coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Sample/Cellspace/BCA-IP-variants"
BASE = ROOT / "Sample/Cellspace/BCA-IP.yaml"
PATCH = DEFAULT_OUTPUT / "condition1-source-patch.json"
EVENTS = ROOT / "Sample/Specialevent/BCA-IP_event.py"
WIDTH_ADDED = 48
CUT_X = -65
CORE_OFFSETS = tuple(66 * i for i in range(6)) + tuple(495 + 66 * i for i in range(6))
CONDITIONS = {
    1: {"a": [1, 1, 2, 2, 4, 4], "b": 6, "c": [2, 3, 4, 5, 1, 1]},
    2: {"a": [1, 1, 2, 2, 4, 4], "b": 8, "c": [2, 2, 1, 5, 5, 5]},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_map(path: Path) -> dict[tuple[int, int], int]:
    items = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
    result = {}
    for item in items:
        coord = item["coord"]["x"], item["coord"]["y"]
        if coord in result:
            raise ValueError(f"Duplicate coordinate {coord} in {path}")
        result[coord] = int(item["value"])
    return result


def write_map(path: Path, cells: dict[tuple[int, int], int]) -> None:
    # Keep world coordinates. The legacy NumPy writer would lose the offset.
    path.write_text("".join(
        f"- coord: {{x: {x}, y: {y}}}\n  value: {value}\n"
        for (x, y), value in sorted(cells.items(), key=lambda item: (item[0][1], item[0][0]))
        if value != 0
    ))


def condition1_source(base: dict, patch: dict) -> dict:
    result = dict(base)
    for item in patch["changes"]:
        coord = item["x"], item["y"]
        if result.get(coord, 0) != item["before"]:
            raise ValueError(f"Source patch does not match at {coord}")
        result[coord] = item["after"]
    return result


def shifted_x(x: int) -> int:
    return x - WIDTH_ADDED if x < CUT_X else x


def widen(cells: dict) -> dict:
    """Insert 16 periods of the existing three-column signal-wire pattern."""
    result = {(shifted_x(x), y): v for (x, y), v in cells.items()}
    ys = {y for _, y in cells}
    allowed = {o + y for o in CORE_OFFSETS for y in (-14, -13, -12, 35, 36, 37, 48, 49, 50)}
    allowed.update((388, 395, 467, 474))  # Unit/TD return wires between A and B.
    for y in ys:
        tile = [cells.get((CUT_X + dx, y), 0) for dx in range(3)]
        if any(tile):
            if y not in allowed or any(v not in (0, 1) for v in tile):
                raise ValueError(f"Unexpected component across widening seam at y={y}: {tile}")
            for dx in range(WIDTH_ADDED):
                v = tile[dx % 3]
                if v:
                    result[CUT_X - WIDTH_ADDED + dx, y] = v
    return result


def simple_amp(base: dict, gain: int) -> dict:
    """Native CJoin/recycle-bin ladder with a single input and gain outputs.

    Ports are (-65,36), (-41,35). All intermediate stages retain the original
    seven-row CJoin/ratchet pitch; a six-row pitch does not preserve behavior.
    """
    if not 1 <= gain <= 9:
        raise ValueError("Supported gains are 1..9")
    if gain == 1:
        cells = {(x, y): base.get((x, y + 198), 0)
                 for y in range(29, 40) for x in range(-66, -40)}
        # Terminate the bypass in the same port phase as the amplified cases.
        for x in range(-49, -40):
            cells[x, 35] = 1
            cells[x, 34] = int(x == -42)
            cells[x, 36] = int(x == -42)
        return {p: v for p, v in cells.items() if v}
    source = {(x, y): base.get((x, y + 264), 0)
              for y in range(8, 40) for x in range(-66, -40)}
    cells = {p: v for p, v in source.items() if 33 <= p[1] <= 39 and v}
    for j in range(1, gain - 1):
        cx, cy = -60 + j, 36 - 7 * j
        for (x, y), v in source.items():
            if 19 <= y <= 25 and v:
                if -61 <= x <= -55:
                    cells[x + cx + 58, y + cy - 22] = v
                if -44 <= x <= -42:
                    cells[x, y + cy - 22] = v
        for x in range(cx + 4, -44):
            cells[x, cy - 1] = 1
    cx, cy = -60 + gain - 2, 36 - 7 * (gain - 2)
    # The gain-2 cap matches this collector's wire phase at every stage count.
    for y in range(29, 33):
        for x in range(-66, -40):
            v = base.get((x, y + 132), 0)
            if v:
                if -60 <= x <= -58:
                    cells[x + cx + 60, y + cy - 36] = v
                if -44 <= x <= -42:
                    cells[x, y + cy - 36] = v
    for x in range(cx + 3, -44):
        cells[x, cy - 6] = 1
    cells[-41, 35] = 1
    return cells


def routed_amp(base: dict, gain: int) -> dict:
    """Transpose the ladder into the widened corridor and connect both ports."""
    cells = {(y - 81, x + 76): v for (x, y), v in simple_amp(base, gain).items()}

    def wire(x: int, y: int) -> None:
        if cells.get((x, y), 1) != 1:
            raise ValueError(f"Wire overlaps a bin or key token at {(x, y)}")
        cells[x, y] = 1

    def cross(x: int, y: int) -> None:
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            wire(x + dx, y + dy)

    # A bend consists of two diagonally adjacent crosses, not a one-cell L.
    for x in range(-113, -107):
        wire(x, 36)
    cross(-108, 36)
    cross(-107, 35)
    for y in range(5, 35):
        wire(-107, y)
    cross(-107, 5)
    cross(-106, 4)
    for x in range(-105, -46):
        wire(x, 4)
    cross(-46, 4)
    cross(-45, 5)
    for y in range(6, 12):
        wire(-45, y)
    # Use the same three-cell directed wire motif as the archived map on the
    # long new leads. A plain Brownian line adds an avoidable quadratic delay.
    for y in range(30, 8, -3):
        cross(-107, y)
        wire(-108, y - 1)
    for x in range(-102, -48, 3):
        cross(x, 4)
        wire(x + 1, 3)
    cross(-45, 35)
    for x in range(-44, -40):
        wire(x, 35)
    cross(-42, 35)
    return cells


def build_variant(source: dict, base: dict, condition: int, n: int) -> tuple[dict, list[int]]:
    params = CONDITIONS[condition]
    weights = [n * (max(params["c"]) - c) + 1 for c in params["c"]]
    result = widen(source)
    for i, offset in enumerate(CORE_OFFSETS):
        gain = weights[i % 6]
        for x in range(-61, -52):
            result[x, offset - 2] = 2 if x >= -52 - gain else 0
        for y in range(3, 43):
            for x in range(-113, -40):
                result.pop((x, offset + y), None)
        for (x, y), v in routed_amp(base, gain).items():
            if not (-113 <= x <= -41 and 3 <= y <= 42):
                raise ValueError(f"Amp escaped its reserved rectangle: {(x,y)}")
            result[x, y + offset] = v
    return {p: v for p, v in result.items() if v}, weights


def generate(output: Path = DEFAULT_OUTPUT) -> dict:
    patch = json.loads(PATCH.read_text())
    if sha256(BASE) != patch["base_sha256"]:
        raise ValueError("BCA-IP.yaml changed; review templates and source patch first")
    output.mkdir(parents=True, exist_ok=True)
    base = read_map(BASE)
    source1 = condition1_source(base, patch)
    events = runpy.run_path(str(EVENTS))["events"]
    moved = [(name, (shifted_x(ref[0]), ref[1]), state,
              (shifted_x(dst[0]), dst[1]), value, *rest)
             for name, ref, state, dst, value, *rest in events]
    event_path = output / "BCA-IP_wide_events.py"
    event_path.write_text(
        '# Generated by scripts/generate_bca_ip_cellspaces.py.\n'
        '# Use only with BCA-IP-variants maps (48-column reset corridor).\n'
        'events = [\n' + ''.join(f'    {event!r},\n' for event in moved) + ']\n'
    )
    manifest = {"schema_version": 1, "base_sha256": sha256(BASE),
                "condition1_archive_sha256": patch["source_sha256"],
                "condition1_patch_sha256": sha256(PATCH),
                "width_added": WIDTH_ADDED, "global_prob": 0.5,
                "events": event_path.name, "events_sha256": sha256(event_path),
                "rules": "Sample/rule/base-rule.yaml",
                "rules_sha256": sha256(ROOT / "Sample/rule/base-rule.yaml"),
                "layout": "48-column corridor in every map; within condition 1, only initial Weights and reset Amp ladders differ with N.",
                "validation": "See validation.json; generation alone is not dynamic certification.",
                "variants": []}
    for condition, n in ((1, 1), (1, 2), (2, 2)):
        source = source1 if condition == 1 else base
        cells, weights = build_variant(source, base, condition, n)
        path = output / f"BCA-IP_condition{condition}_N{n}.yaml"
        write_map(path, cells)
        xs, ys = zip(*cells)
        manifest["variants"].append({
            "file": path.name, "condition": condition, "N": n,
            **CONDITIONS[condition], "weights": weights, "reset_gains": weights,
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "nonzero_cells": len(cells), "tokens": sum(v == 2 for v in cells.values()),
            "sha256": sha256(path),
        })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2))
