"""Plot the two CJoin inputs from the archived physical monitor diagnostic."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "independent.json").read_text())
fixture = next(f for f in data["fixtures"] if f["name"] == "fsm_x1")
fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), layout="constrained")
for ax, case, title, result in zip(
    axes, fixture["cases"][:2],
    ["global_prob = 1.0: the two inputs never coincide",
     "Only Token bin rules changed to 0.5: the inputs can coincide"],
    ["After 2,000 updates: counter = 0, downstream = 0",
     "After 2,000 updates: counter = 1, downstream = 1"],
):
    # Start-of-step codes for trial 1: one signal injected before update 0.
    codes = np.array([frame[0][1] for frame in case["first_120_neighborhoods"]])
    present = np.array([(codes & 1) != 0, (codes & 2) != 0, (codes & 3) == 3])
    colored = present * np.array([1, 2, 3])[:, None]
    ax.imshow(colored[:, 20:52], interpolation="nearest", aspect="auto",
              cmap=ListedColormap(["#eef1f4", "#2875a5", "#cb7634", "#298457"]),
              vmin=0, vmax=3, extent=(19.5, 51.5, 2.5, -0.5))
    ax.set_yticks([0, 1, 2], labels=["Bin token (180, 7)", "Signal token (179, 8)", "Both inputs present"])
    ax.set_xticks(range(20, 52, 2))
    ax.set_xticks(np.arange(19.5, 52, 1), minor=True)
    ax.grid(which="minor", axis="x", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False)
    ax.set_title(title + "\n" + result, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("CA update index (state before rule matching)")
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle("BCA-IP: phase blocking at the actual FSM-to-Amp monitor\n"
             "Same input pulse and geometry; colored cells mean a token is present",
             fontsize=14)
fig.savefig(ROOT / "phase-blocking.png", dpi=180)
plt.close(fig)
print(ROOT / "phase-blocking.png")
