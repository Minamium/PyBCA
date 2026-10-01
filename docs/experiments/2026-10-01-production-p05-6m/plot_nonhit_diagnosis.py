"""Recorded output dynamics and observed escape rates; no reconstructed events."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "stall-diagnosis-summary.json").read_text())
rows = [json.loads(s) for s in (HERE / "nonhit-trials.jsonl").read_text().splitlines()]
data = np.load(HERE / "fsm-output-counts.npz")
counts, resets = data["counts"], data["resets"]
width = int(data["bin_width"])
stages = np.load(HERE / "stage-counts.npz")
names = stages["names"].tolist()
colors = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#333333"]
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})

fig, ax = plt.subplots(figsize=(8,4.5),constrained_layout=True)
values = ["9","10","12"]
total = np.array([summary["arrival_by_3m_best_value"][x]["at_3m"] for x in values])
hits = np.array([summary["arrival_by_3m_best_value"][x]["reached_by_6m"] for x in values])
ax.bar(values,hits,color="#13795b",label="Reached optimum during 3M to 6M")
ax.bar(values,total-hits,bottom=hits,color="#bcc4cc",label="Still unconfirmed at 6M")
for i,(n,k) in enumerate(zip(total,hits)):
    ax.text(i,n+1,f"{k}/{n} newly reached ({k/n:.1%})",ha="center",fontsize=10)
ax.set_ylim(0,56);ax.set_ylabel("Trials unconfirmed at 3M")
ax.set_xlabel("Best readable objective at 3M (optimum = 14)")
ax.set_title("Extra 3M steps mostly resolve additive completion, not value-10 plateaus")
ax.legend(loc="upper right",frameon=False,fontsize=9)
fig.savefig(HERE / "escape-by-prior-value.png",dpi=180)
plt.close(fig)

trial = 119
record = next(r for r in rows if r["trial_id"]==trial)
fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
x=np.arange(counts.shape[1]+1)*width/1e6
for j,u in enumerate("AB"):
    ax=axes[j]
    y=np.vstack([np.zeros((1,6)),counts[trial*2+j].cumsum(axis=0)])
    for i in range(6):ax.step(x,y[:,i],where="post",color=colors[i],label=f"x{i+1}",lw=1.3)
    for t in record[u]["reset_steps"]:ax.axvline(t/1e6,color="#bb3333",ls="--",lw=.8,alpha=.7)
    ax.axvline(3,color="#555555",ls=":",lw=1)
    ax.set_title(f"Trial {trial}, Unit {u}: final {''.join(map(str,record[u]['terminal']['vector']))}; resets = {len(record[u]['reset_steps'])}")
    ax.set_ylabel("Cumulative FSM output pulses");ax.grid(alpha=.15)
axes[0].legend(ncol=6,frameon=False,loc="upper left")
axes[-1].set_xlabel("Completed CA updates (millions); red dashed lines = reset of this unit")
axes[-1].set_xlim(0,6)
fig.suptitle("A holds a value-10 output for 4.81M updates while B receives 10 resets",fontsize=12)
fig.savefig(HERE / "nonhit-trial-119-6m.png",dpi=180)
plt.close(fig)

trial,start,end=28,2180000,2480000
b0,b1=start//width,end//width
x=np.arange(b0,b1+1)*width/1e6
fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
for name,label,color in [("F_value_A","Unit A (decoded objective 9)","#0072b2"),("F_value_B","Unit B (decoded objective 7)","#d55e00")]:
    y=np.r_[0,stages["counts"][trial,b0:b1,names.index(name)].cumsum()]
    axes[0].step(x,y,where="post",label=label,color=color)
for name,label,color in [("B_x5output","FSM x5 output", "#0072b2"),("B_Amp_x5output","Amp x5 output", "#d55e00")]:
    y=np.r_[0,stages["counts"][trial,b0:b1,names.index(name)].cumsum()]
    axes[1].step(x,y,where="post",label=label,color=color)
for ax in axes:
    ax.set_xlim(start/1e6,end/1e6);ax.set_xlabel("Completed CA updates (millions)")
    ax.set_ylabel("Pulses since 2.18M");ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
axes[0].set_title("Unit flow reverses the decoded objective order")
axes[1].set_title("B x5 has no new FSM pulses, but 177 Amp pulses")
fig.suptitle("Trial 28: recorded 300k window with no reset in either unit",fontsize=12)
fig.savefig(HERE / "ranking-memory-example.png",dpi=180)
plt.close(fig)
