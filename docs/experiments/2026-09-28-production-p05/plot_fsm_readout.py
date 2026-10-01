"""Figures for terminal FSM output readouts; no simulated/reconstructed events."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "fsm-readout-summary.json").read_text())
trials = [json.loads(line) for line in (HERE / "fsm-readout-trials.jsonl").read_text().splitlines()]
data = np.load(HERE / "fsm-output-counts.npz")
counts, resets = data["counts"], data["resets"]
width = int(data["bin_width"])
palette = ["#13795b", "#56677b", "#d58a22"]
categories = ["optimal", "stable_nonoptimal", "unresolved"]
labels = ["Stable optimal output", "Both stable, nonoptimal", "Insufficient stable evidence"]
amounts = [summary[k] for k in ("optimal_trials", "stable_nonoptimal_trials", "unresolved_trials")]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), width_ratios=[1, 1.3], constrained_layout=True)
ax = axes[0]
ax.barh(np.arange(3), amounts, color=palette, height=.58)
ax.set_yticks(np.arange(3), ["Optimal", "Nonoptimal", "Unresolved"])
ax.invert_yaxis(); ax.set_xlim(0, 535)
for i, n in enumerate(amounts):
    ax.text(n+7, i, f"{n} / 512\n{n/512:.1%}", va="center", fontsize=11)
ax.set_xlabel("Independent trials (A/B together)")
ax.set_title("Final stable FSM output at 3,000,000 updates")
ax.grid(axis="x", alpha=.15)
states = np.array([categories.index(t["status"]) for t in trials]).reshape(32, 16)
ax = axes[1]
ax.imshow(states, cmap=ListedColormap(palette), vmin=0, vmax=2, aspect="auto",
          extent=(-.5, 15.5, 31.5, -.5), interpolation="nearest")
ax.set_xticks([0, 4, 8, 12, 15]); ax.set_yticks(range(0, 32, 4), [str(i*16) for i in range(0,32,4)])
ax.set_xlabel("Column offset (trial ID = row label + column)")
ax.set_ylabel("First trial ID in row")
ax.set_title("All 512 trials")
fig.legend(handles=[Patch(facecolor=c, label=l) for c,l in zip(palette,labels)],
           loc="outside lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("BCA-IP / V100 x 8 / global_prob = 0.5", fontsize=13)
fig.savefig(HERE / "fsm-readout-outcomes.png", dpi=180)
plt.close(fig)

colors = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#333333"]
samples = [0, 13, 8, 248]
fig, axes = plt.subplots(len(samples), 2, figsize=(12, 12), constrained_layout=True)
start_bin = 240
x = np.arange(start_bin, counts.shape[1]+1)*width/1e6
for row, trial in enumerate(samples):
    for u, unit in enumerate("AB"):
        ax, record = axes[row, u], trials[trial][unit]
        y = np.concatenate([np.zeros((1,6)), counts[trial*2+u, start_bin:].cumsum(0)])
        for v in range(6):
            ax.step(x, y[:,v], where="post", color=colors[v], lw=1.3)
        ax.axvspan(record["readout_start"]/1e6, 3, color="#dcefe5", alpha=.5, zorder=-2)
        ax.axvline(record["last_rate_change"]/1e6, color="#222222", linestyle=":", lw=1.2)
        for _, _, step in resets[(resets[:,0] == trial) & (resets[:,1] == u)]:
            ax.axvline(step/1e6, color="#b03b3b", linestyle="--", alpha=.65, lw=1)
        vector = "".join("?" if v is None else str(v) for v in record["vector"])
        status = "optimal" if record["optimal"] else "nonoptimal" if record["stable_readout"] else "unresolved"
        ax.set_title(f"Trial {trial}, Unit {unit}: {vector} ({status})", fontsize=10)
        ax.set_xlim(2.4, 3); ax.set_ylim(bottom=0); ax.grid(alpha=.18)
        ax.set_ylabel("FSM output pulses since 2.4 M")
        ax.set_xlabel("Completed updates (millions)")
legend = [Line2D([],[],color=c,label=f"x{i+1}") for i,c in enumerate(colors)]
legend += [Line2D([],[],color="#222222",linestyle=":",label="Last joint rate change"),
           Line2D([],[],color="#b03b3b",linestyle="--",label="Reset of this unit"),
           Patch(facecolor="#dcefe5",label="Readout interval")]
fig.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
fig.suptitle("Terminal cumulative FSM outputs (10,000-update plotting bins)\nReadout quarter counts use exact event times", fontsize=12)
fig.savefig(HERE / "fsm-readout-examples.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for u,unit in enumerate("AB"):
    ax, record = axes[u], trials[0][unit]
    y = np.concatenate([np.zeros((1,6)), counts[u].cumsum(0)])
    x = np.arange(len(y))*width/1e6
    for v in range(6):
        ax.step(x,y[:,v],where="post",color=colors[v],label=f"x{v+1}",lw=1.3)
    for _,_,step in resets[(resets[:,0]==0)&(resets[:,1]==u)]:
        ax.axvline(step/1e6,color="#b03b3b",linestyle="--",alpha=.6)
    ax.axvspan(record["readout_start"]/1e6,3,color="#dcefe5",alpha=.5,zorder=-2)
    ax.set_title(f"Trial 0, Unit {unit}: {record['vector']}",fontsize=10)
    ax.set_xlim(0,3);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
    ax.set_xlabel("Completed updates (millions)");ax.set_ylabel("Cumulative FSM output pulses")
axes[0].legend(ncol=3,frameon=False,fontsize=8)
fig.suptitle("Example: Unit A retains an optimal terminal output; Unit B is reset",fontsize=12)
fig.savefig(HERE / "fsm-readout-trial-0.png",dpi=180)
plt.close(fig)
