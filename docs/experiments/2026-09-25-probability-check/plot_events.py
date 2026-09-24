"""Plot the archived paired diagnostic; event steps are zero based in JSON."""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "comparison.json").read_text())
assert not data["errors"]
probe = data["global_prob_0_5"]["rows"]
control = data["global_prob_1_0_control"]["rows"]
categories = [
    ("FSM to Amp", lambda name: re.fullmatch(r"[AB]_x[1-6]output", name)),
    ("Amp output", lambda name: "_Amp_x" in name),
    ("Unit output", lambda name: name.startswith("F_value_")),
]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
fig = plt.figure(figsize=(12, 8), layout="constrained")
grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.35])
for column, (title, match) in enumerate(categories):
    ax = fig.add_subplot(grid[0, column * 2:column * 2 + 2])
    for records, label, color, style in [
        (probe, "global_prob = 0.5", "#176fa6", "-"),
        (control, "global_prob = 1.0", "#cf6b22", "--"),
    ]:
        selected = sorted((r for r in records if match(r["name"])),
                          key=lambda row: row["step"])
        times = [0] + [r["step"] + 1 for r in selected] + [100000]
        values = [0] + np.cumsum([r["count"] for r in selected]).tolist()
        values.append(values[-1])
        ax.step(np.array(times) / 1000, values, where="post", label=label,
                color=color, linestyle=style, linewidth=1.7)
        if records is probe:
            ax.annotate(f"{values[-1]} events", (100, values[-1]),
                        xytext=(-6, 7), textcoords="offset points", ha="right")
    total = sum(r["count"] for r in probe if match(r["name"]))
    ax.set(title=title, xlabel="Completed updates (thousands)",
           xlim=(0, 100), ylim=(-3, total * 1.2 + 2))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    if column == 0:
        ax.set_ylabel("Cumulative events (2 trials combined)")
        ax.legend(loc="upper left", fontsize=9)

for trial in [0, 1]:
    ax = fig.add_subplot(grid[1, trial * 3:trial * 3 + 3])
    labels = [f"{unit} x{i}" for unit in "AB" for i in range(1, 7)]
    for row, label in enumerate(labels):
        unit, variable = label.split()
        times = [(r["step"] + 1) / 1000 for r in probe
                 if r["trial"] == trial and r["name"] == f"{unit}_{variable}output"]
        ax.vlines(times, row - 0.30, row + 0.30,
                  color="#176fa6" if unit == "A" else "#a54669", linewidth=1.2)
    ax.axhline(5.5, color="0.75", linewidth=0.8)
    ax.set_yticks(range(12), labels=labels)
    ax.set(title=f"Trial {trial}: FSM to Amp detections, global_prob = 0.5",
           xlabel="Completed updates (thousands)", xlim=(0, 100), ylim=(11.7, -0.7))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", alpha=0.2)

fig.suptitle("BCA-IP probability diagnostic: same seed and trial IDs\n"
             "2 trials, 100,000 updates each; output detections do not certify solution bits",
             fontsize=13)
fig.savefig(ROOT / "probe-events.png", dpi=180)
plt.close(fig)
print(ROOT / "probe-events.png")
