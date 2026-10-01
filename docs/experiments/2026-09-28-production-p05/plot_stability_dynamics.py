"""Plot shortened-persistence readouts and nonoptimal output trajectories."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
sweep = json.loads((HERE/"short-stability-summary.json").read_text())
dynamics = json.loads((HERE/"nonhit-summary.json").read_text())
cases = {r["trial_id"]:r for r in json.loads((HERE/"short-stability-cases.json").read_text())}
z = np.load(HERE/"fsm-output-counts.npz")
counts, resets, width = z["counts"], z["resets"], int(z["bin_width"])
colors = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#333333"]
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})

table = sorted(sweep["tables"]["original"],key=lambda r:r["minimum_duration"])
x = np.array([r["minimum_duration"] for r in table])/1000
fig, ax = plt.subplots(figsize=(8.5,4.6),constrained_layout=True)
for key,label,color in [("terminal_optimal_trials","Optimal at the end","#13795b"),
                        ("ever_supported_optimum_trials","Optimum supported in any regime","#4765ad")]:
    values = [r[key] for r in table]
    ax.plot(x,values,"o-",color=color,label=label,lw=1.8)
    for xx,yy in zip(x,values):ax.annotate(str(yy),(xx,yy),xytext=(0,7),textcoords="offset points",ha="center",fontsize=9)
ax.set_xticks(x);ax.set_yticks(range(422,432,2));ax.set_ylim(421,432)
ax.set_xlabel("Minimum regime duration (thousand CA updates)")
ax.set_ylabel("Trials out of 512")
ax.set_title("Shorter persistence threshold: 423 -> 425 terminal optima\nSignal test unchanged; original and finer fits give the same success IDs",fontsize=11)
ax.legend(loc="lower left",frameon=False);ax.grid(alpha=.18)
fig.savefig(HERE/"short-stability.png",dpi=180);plt.close(fig)

g = dynamics["groups"]["never_supported"]
fig, axes = plt.subplots(1,2,figsize=(11.5,4.5),constrained_layout=True)
vals=[9,10,12]; n=[g["best_readable_terminal_values"][str(v)] for v in vals]
axes[0].bar([str(v) for v in vals],n,color=["#56677b","#c88024","#338776"])
for i,k in enumerate(n):axes[0].text(i,k+1,str(k),ha="center")
axes[0].set_ylim(0,55);axes[0].set_xlabel("Best readable terminal objective (optimum = 14)")
axes[0].set_ylabel("Trials");axes[0].set_title("82 trials with no supported optimum")
for name,label,color in [("never_supported","Never supported (82)","#c88024"),("retained","Retained optimum (425)","#13795b")]:
    a=np.sort(dynamics["groups"][name]["best_terminal_same_durations"])/1e6
    axes[1].step(np.r_[0,a],np.r_[0,np.arange(1,len(a)+1)/len(a)],where="post",label=label,color=color,lw=1.8)
axes[1].axvline(1,color="#777777",ls=":",lw=1)
axes[1].set_xlabel("Unchanged terminal-best vector duration (million updates)")
axes[1].set_ylabel("Fraction of trials");axes[1].set_xlim(0,3);axes[1].set_ylim(0,1.02)
axes[1].set_title("Long plateaus also occur for nonoptimal outputs")
axes[1].legend(frameon=False,fontsize=9)
for a in axes:a.grid(axis="y",alpha=.18)
fig.savefig(HERE/"nonhit-characteristics.png",dpi=180);plt.close(fig)


def trace(ax,trial,unit,start,end,change=None):
    u="AB".index(unit); lo=start//width;hi=end//width
    y=np.vstack([np.zeros((1,6)),counts[trial*2+u,lo:hi].cumsum(axis=0)])
    t=np.arange(lo,hi+1)*width/1e6
    for v in range(6):ax.step(t,y[:,v],where="post",color=colors[v],lw=1.25)
    for _,_,at in resets[(resets[:,0]==trial)&(resets[:,1]==u)]:
        if start<=at<=end:ax.axvline(at/1e6,color="#b03b3b",ls="--",alpha=.65,lw=1)
    if change is not None:ax.axvline(change/1e6,color="#333333",ls=":",lw=1)
    r=cases[trial]["terminal"][unit]
    vector="".join("?" if b is None else str(b) for b in r["vector"])
    value=r["objective_value"] if r["objective_value"] is not None else "?"
    ax.set_title(f"Trial {trial}, Unit {unit}; final {vector}, value {value}",fontsize=10)
    ax.set_xlim(start/1e6,end/1e6);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
    ax.set_xlabel("Completed updates (millions)");ax.set_ylabel("FSM output pulses in shown interval")


legend=[Line2D([],[],color=c,label=f"x{i+1}") for i,c in enumerate(colors)]
legend += [Line2D([],[],color="#b03b3b",ls="--",label="Reset of this unit")]
fig, axes=plt.subplots(4,2,figsize=(12,11.5),constrained_layout=True)
for row,trial in enumerate([13,19,119,32]):
    for col,unit in enumerate("AB"):trace(axes[row,col],trial,unit,0,3_000_000)
fig.legend(handles=legend,loc="outside lower center",ncol=7,frameon=False,fontsize=9)
fig.suptitle("Nonoptimal output patterns: a long-lived candidate and repeated exploration\nCurves use 10,000-update plotting bins; binary readouts use exact event times",fontsize=12)
fig.savefig(HERE/"nonhit-output-examples.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),constrained_layout=True)
for col,unit in enumerate("AB"):trace(axes[col],119,unit,0,3_000_000)
fig.legend(handles=legend,loc="outside lower center",ncol=7,frameon=False,fontsize=9)
fig.suptitle("Trial 119: Unit A retains [1,1,1,1,0,0] (value 10); Unit B is reset",fontsize=12)
fig.savefig(HERE/"nonhit-trial-119.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(5,2,figsize=(12,14),constrained_layout=True)
for row,(trial,change) in enumerate([(339,2_020_000),(412,930_000),(248,2_780_000),(295,2_850_000),(458,2_860_000)]):
    for col,unit in enumerate("AB"):
        trace(axes[row,col],trial,unit,change-300_000,min(3_000_000,change+300_000),change)
        axes[row,col].set_title(f"Trial {trial}, Unit {unit}; near B's rate change at {change/1e6:.2f} M",fontsize=10)
fig.legend(handles=legend+[Line2D([],[],color="#333333",ls=":",label="B rate change")],
           loc="outside lower center",ncol=4,frameon=False,fontsize=9)
fig.suptitle("Five earlier optimum readouts not confirmed at the endpoint\nTop two: reset into B; bottom three: no recorded reset into B",fontsize=12)
fig.savefig(HERE/"earlier-optimum-output-changes.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),constrained_layout=True)
for ax,(trial,unit,change) in zip(axes,[(8,"B",2_960_000),(37,"A",2_930_000)]):
    trace(ax,trial,unit,2_800_000,3_000_000,change)
    ax.axvspan(change/1e6,3,color="#dcefe5",alpha=.4,zorder=-2)
fig.legend(handles=legend,loc="outside lower center",ncol=7,frameon=False,fontsize=9)
fig.suptitle("Two extra terminal optima admitted by shorter persistence requirements",fontsize=12)
fig.savefig(HERE/"short-stability-additions.png",dpi=180);plt.close(fig)
