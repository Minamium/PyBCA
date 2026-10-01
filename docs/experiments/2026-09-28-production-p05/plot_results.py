"""Plot verified module activity; these are not optimum hitting-time curves."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = np.load(HERE / "event-counts.npz")
audit = json.loads((HERE / "audit.json").read_text())
counts = data["bins"].sum(axis=0)
x = np.r_[0, np.arange(1, counts.shape[1]+1)*int(data["bin_width"])]/1e6
titles = ["TD to FSM", "FSM to Amp", "Amp output", "Unit output", "Comparator output", "RSM reset injection"]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)
for j, (ax, title) in enumerate(zip(axes.flat, titles)):
    y = np.r_[0, counts[j].cumsum()] / audit["trials"]
    ax.plot(x, y, color="#177c90", lw=1.7)
    n = audit["groups"][str(data["groups"][j])]["trials_with_event"]
    ax.set_title(f"{title}\n{n}/512 trials with an event", fontsize=10)
    ax.set_xlabel("Completed updates (millions)")
    ax.set_ylabel("Cumulative events per trial")
    ax.set_xlim(0,3);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
fig.suptitle("V100 x 8 / 512 trials / p = 0.5 / 3,000,000 updates", fontsize=12)
fig.savefig(HERE / "module-events.png", dpi=180)
plt.close(fig)

ratios = json.loads((HERE / "amp-ratios.json").read_text())["rows"]
fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
ax.bar(np.arange(1,7)-.18, [r["aggregate_output_input_ratio"] for r in ratios], width=.36,
       color="#177c90", label="Observed total output / input")
ax.bar(np.arange(1,7)+.18, [r["si_instance_2_coefficient"] for r in ratios], width=.36,
       color="#bbc2c8", label="SI Instance 2 coefficient (reference)")
ax.set_xticks(range(1,7), [f"x{i}" for i in range(1,7)])
ax.set_ylabel("Output / input count ratio")
ax.set_title("Amp aggregate ratios over the full run\nFinite-run diagnostic, not calibrated per-pulse gain", fontsize=11)
ax.legend(frameon=False,fontsize=9);ax.grid(axis="y",alpha=.15)
fig.savefig(HERE / "amp-ratios.png", dpi=180)
plt.close(fig)
