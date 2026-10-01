"""Regenerate the A100 comparison from the archived measurements."""
from pathlib import Path
import gzip
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
raw_search = gzip.decompress((ROOT / "capacity-search.json.gz").read_bytes())
plan = json.loads(gzip.decompress((ROOT / "capacity-profile.json.gz").read_bytes()))
assert plan["completed"]
assert hashlib.sha256(raw_search).hexdigest() == plan["continuation"]["source_plan_sha256"]
control = json.loads((ROOT / "trials-64.json").read_text())
profiles = [plan["profile_512"], plan["profile_near_full_memory"]]
summary = {"gpu": plan["gpu"], "global_prob": plan["global_prob"],
           "target_steps": plan["target_steps"], "completed_performance_benchmark": True,
           "completed_target_simulation": False, "profiles": []}
for row in profiles:
    result = {key: row[key] for key in ["trials", "workers_on_same_gpu", "warmup_steps",
        "steps_per_repeat", "repeats", "seconds_per_step_samples", "median_ms_per_step",
        "trial_steps_per_sec", "nominal_hours_for_target", "conservative_hours_for_target",
        "cohort_flush_checkpoint_sec", "sampled_device_peak_used_bytes"]}
    result["sampled_device_peak_used_gib"] = row["sampled_device_peak_used_bytes"] / 2**30
    result["device_memory_fraction"] = row["sampled_device_peak_used_bytes"] / plan["gpu_total_memory_bytes"]
    summary["profiles"].append(result)
summary["control_64"] = {"trial_steps_per_sec": control["trial_steps_per_sec"],
    "median_ms_per_step": control["median_ms_per_step"],
    "update_only_hours_for_target": 3_000_000 * control["median_ms_per_step"] / 1000 / 3600,
    "checkpoint_filesystem": "Temporary directory; not the primary output filesystem"}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (memory, throughput) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
fig.suptitle("BCA-IP · one A100-SXM4-80GB · global probability 0.5", fontsize=14)
probes = [r for r in plan["measurements"] if r["label"].startswith("capacity-") and r["status"] == "ok"]
memory.scatter([r["trials"] for r in probes],
               [r["sampled_device_peak_used_bytes"] / 2**30 for r in probes],
               color="#8b949e", s=32, label="Short capacity probe")
throughput.scatter([r["trials"] for r in probes],
                   [r["trial_steps_per_sec"] for r in probes],
                   color="#8b949e", s=32, label="Short capacity probe")
for axis in [memory, throughput]:
    axis.set_xlabel("Simultaneous trials on one GPU")
    axis.set_xlim(0, 4300)
    axis.grid(alpha=0.15)
for r in profiles:
    memory.scatter(r["trials"], r["sampled_device_peak_used_bytes"] / 2**30,
                   color="#1762aa", marker="s", s=55, zorder=3)
    throughput.scatter(r["trials"], r["trial_steps_per_sec"],
                       color="#1762aa", marker="s", s=55, zorder=3)
    throughput.annotate(f'{r["trials"]:,}: {r["trial_steps_per_sec"]:,.0f}/s',
        (r["trials"], r["trial_steps_per_sec"]), xytext=(0, 12), textcoords="offset points",
        ha="center", color="#1762aa")
memory.scatter([], [], color="#1762aa", marker="s", s=55, label="Three-repeat profile")
throughput.scatter([], [], color="#1762aa", marker="s", s=55, label="Three-repeat profile")
throughput.scatter(64, control["trial_steps_per_sec"], color="#ce7d20", marker="*", s=120,
                   label="64-trial single-process control", zorder=4)
throughput.annotate(f'64: {control["trial_steps_per_sec"]:,.0f}/s',
    (64, control["trial_steps_per_sec"]), xytext=(14, 0), textcoords="offset points", va="center")
memory.axhline(plan["memory_limit_bytes"] / 2**30, color="#bc4646", ls="--", lw=1.2,
               label="96% of CUDA usable memory")
memory.axhline(plan["gpu_total_memory_bytes"] / 2**30, color="#555555", ls=":", lw=1)
for n in [3840, 4096]:
    memory.text(n, 82, "OOM", ha="center", fontsize=9, color="#bc4646", rotation=45)
memory.set_ylim(0, 89)
memory.set_ylabel("Sampled whole-device memory use (GiB)")
throughput.set_ylim(1400, 4150)
throughput.set_ylabel("Trial updates per second (higher is better)")
memory.legend(loc="lower right", fontsize=8, frameon=False)
throughput.legend(loc="upper right", bbox_to_anchor=(1.02, 0.87), fontsize=8, frameon=False)
fig.savefig(ROOT / "throughput-capacity.png", dpi=180)
fig.savefig(ROOT / "throughput-capacity.pdf")
print(json.dumps(summary, indent=2))
