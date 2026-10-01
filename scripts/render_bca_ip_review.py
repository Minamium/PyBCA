"""Render verified local checkpoint review data, without rerunning simulations."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import yaml

from PyBCA.core.io import extract_cellspace_and_offset, load_cell_space_yaml_to_numpy

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#8960a6", "#ffffff", "#acb7bf", "#ca283d"]


def encode(rows):
    delta = rows.copy()
    delta[1:] = rows[1:] ^ rows[:-1]
    return base64.b64encode(gzip.compress(delta.tobytes(), 9, mtime=0)).decode("ascii")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("review", type=Path)
    p.add_argument("--inline", type=Path, required=True)
    args = p.parse_args()
    folder = args.review.resolve()
    data = json.loads((folder / "viewer-data.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    state = np.load(folder / "selected-states.npz")
    initial = state["initial"]
    ox, oy = map(int, state["offset"])
    h, w = initial.shape
    wire = initial.ravel() > 0
    event_rows = runpy.run_path(str(ROOT / "Sample/Specialevent/BCA-IP_event.py"))["events"]
    data["events"] = [[r[0], *r[1]] for r in event_rows]
    # The full 128-step traces remain in NPZ. Show 24 consecutive steps online,
    # without temporal decimation that could hide a period-two oscillation.
    for entry in data["replays"]:
        frames = np.load(folder / f"replay-{entry['label']}.npz")["tokens"][:25]
        entry["frames"] = len(frames)
        entry["bits"] = encode(frames)
    template = (ROOT / "scripts/templates/bca_ip_cells.html").read_text()
    html = template.replace("__BCA_CELL_DATA__", json.dumps(data, separators=(",", ":")))
    assert len(html.encode()) < 1_000_000
    args.inline.parent.mkdir(parents=True, exist_ok=True)
    args.inline.write_text(html)
    # Retain an inspectable project-local fragment as well as the inline one.
    (folder / "cellspace-viewer.fragment.html").write_text(html)

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm([-1.5, -.5, .5, 1.5, 2.5], 4)
    extent = [ox-.5, ox+w-.5, oy+h-.5, oy-.5]
    p1, p05 = summary["replays"]
    for key, meta in (("p1", p1), ("p05", p05)):
        a = state[key]
        y, x = np.nonzero(a)
        entries = [{"coord": {"x": int(xx+ox), "y": int(yy+oy)}, "value": int(a[yy,xx])}
                   for yy, xx in zip(y, x)]
        # Preserve both the original coordinates and the complete rectangle.
        for xx, yy in ((0, 0), (w-1, h-1)):
            if a[yy,xx] == 0:
                entries.append({"coord": {"x": xx+ox, "y": yy+oy}, "value": 0})
        path = folder / f"{key}-trial-{meta['trial_id']:04d}-step-{meta['source_step']}.yaml"
        path.write_text(yaml.dump(entries, Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper), sort_keys=False))
        decoded, dx, dy = extract_cellspace_and_offset(load_cell_space_yaml_to_numpy(str(path)))
        assert np.array_equal(decoded, a) and [dx, dy] == [ox, oy]
    cases = [("initial", "Initial / 0 updates"),
             ("p1", f"Stopped p=1.0 / trial {p1['trial_id']} / {p1['source_step']:,} updates"),
             ("p05", f"Diagnostic p=0.5 / trial {p05['trial_id']} / {p05['source_step']:,} updates")]
    legend = [Line2D([], [], color=COLORS[2], lw=3, label="Wire"),
              Line2D([], [], color=COLORS[3], marker="o", ls="", label="Token (state 2)"),
              Line2D([], [], color=COLORS[0], marker="s", ls="", label="Token bin (-1)")]

    def show(ax, a, bounds=None):
        ax.imshow(a, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        if bounds:
            x0,x1,y0,y1 = bounds
            ax.set_xlim(x0-.5,x1+.5); ax.set_ylim(y1+.5,y0-.5)
        ax.set_xlabel("x (cell)"); ax.set_ylabel("y (cell)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    for ax, (key, label) in zip(axes, cases):
        show(ax, state[key]); ax.set_title(label, fontsize=9)
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(folder / "overview.png", dpi=200); plt.close(fig)

    for region, title in [("A4", "A4: TD input, FSM-core and output"),
                           ("A4_core", "A4: FSM-core"),
                           ("comparator_rsm", "Comparator and reset signal module")]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
        for ax, (key, label) in zip(axes, cases):
            show(ax, state[key], data["regions"][region]); ax.set_title(label, fontsize=10)
            if region == "A4":
                for x,y,text in [(-87,209,"TD counter"),(187,207,"FSM counter")]:
                    ax.plot(x,y,"o",mfc="none",mec="#258598",ms=7)
                    ax.annotate(text, (x,y), (x-4 if x>0 else x+3,y+18), fontsize=8,
                                ha="right" if x>0 else "left",
                                arrowprops={"arrowstyle":"-","color":"#258598"})
        fig.suptitle(title, fontsize=12)
        fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
        fig.savefig(folder / f"{region}-comparison.png", dpi=200); plt.close(fig)

    # Visualize which physical cells actually changed in the full 128 updates.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, key in enumerate(("p1", "p05")):
        packed = np.load(folder / f"replay-{key}.npz")["tokens"]
        bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :wire.sum()]
        change_counts = (bits[1:] != bits[:-1]).sum(axis=0)
        image = np.zeros(initial.size, dtype=float)
        image[wire] = change_counts / (len(bits)-1)
        image = image.reshape(initial.shape)
        for col, region in enumerate(("A4_core", "comparator_rsm")):
            ax=axes[row,col]
            show(ax, np.where(initial==2,1,initial), data["regions"][region])
            m = ax.imshow(np.ma.masked_equal(image, 0), cmap="magma_r", vmin=0, vmax=1,
                          extent=extent, interpolation="nearest")
            ax.set_title(f"p={'1.0' if key=='p1' else '0.5'} / {region} / 128 diagnostic updates", fontsize=10)
    fig.colorbar(m, ax=axes, label="Fraction of updates with a cell-state change", shrink=.75)
    fig.savefig(folder / "motion-128-steps.png", dpi=200); plt.close(fig)
    print(json.dumps({"inline": str(args.inline), "bytes": len(html.encode()),
                      "verified_states": sum(a["shape"][0] for a in summary["audits"])}))


if __name__ == "__main__":
    main()
