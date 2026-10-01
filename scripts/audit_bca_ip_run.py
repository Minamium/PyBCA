"""Audit a completed BCA-IP run and summarize module events, not optimum hits.

Checks every checkpoint and immutable history chunk. Reset set/clear rows are
validated as pairs; a logical reset is counted once using its set event.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from PyBCA.core.io import extract_cellspace_and_offset, load_cell_space_yaml_to_numpy
from PyBCA.api.streaming import sha256

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("td_to_fsm", "fsm_to_amp", "amp_output", "unit_output", "comparator", "reset_signal")


def category(name):
    if "_core_input_" in name:
        return 0
    if re.fullmatch("[AB]_x[1-6]output", name):
        return 1
    if "_Amp_x" in name:
        return 2
    if name.startswith("F_value_"):
        return 3
    if name.startswith("Comparate_"):
        return 4
    if name.endswith("_set"):
        return 5
    return None


def matching_probe_rows(directory):
    m = json.loads((directory / "manifest.json").read_text())
    rows = []
    for chunk in m["chunks"]:
        f = directory / chunk["path"]
        assert sha256(f) == chunk["sha256"]
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            rows.append((r["trial"], r["step"], r["name"], r["count"]))
    return sorted(rows), m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--remote-hashes", type=Path, required=True)
    ap.add_argument("--reference-probe", type=Path)
    ap.add_argument("--trials", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--global-prob", type=float, default=.5)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.run.resolve()
    remote_hashes = {}
    for line in args.remote_hashes.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.*/rank_\d{4}/checkpoint\.pt)", line)
        if match:
            remote_hashes[Path(match[2]).parent.name] = match[1]
    grid, ox, oy = extract_cellspace_and_offset(load_cell_space_yaml_to_numpy(str(ROOT / "Sample/Cellspace/BCA-IP.yaml")))
    inputs = {"cellspace": sha256(ROOT / "Sample/Cellspace/BCA-IP.yaml"),
              "rules": [sha256(ROOT / "Sample/rule/base-rule.yaml")],
              "events": sha256(ROOT / "Sample/Specialevent/BCA-IP_event.py")}
    binsize = 100_000
    bin_count = (args.steps+binsize-1)//binsize
    totals = np.zeros((args.trials,len(GROUPS)), dtype=np.int64)
    binned = np.zeros((args.trials,len(GROUPS),bin_count), dtype=np.int64)
    first = np.full((args.trials,len(GROUPS)), -1, dtype=np.int64)
    last = first.copy()
    names_total = Counter()
    all_ids, audits, reset_rows, comparison_rows, td_rows, probe_rows = [], [], [], [], [], []
    cells_sha, checkpoints, envs = set(), [], []
    probe = probe_manifest = None
    if args.reference_probe:
        probe, probe_manifest = matching_probe_rows(args.reference_probe)
    for directory in sorted(source.glob("rank_????")):
        path = directory / "checkpoint.pt"
        digest = sha256(path)
        assert digest == remote_hashes[directory.name], path
        state = torch.load(path, map_location="cpu", weights_only=True)
        m = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        assert state["manifest"] == m and state["next_step"] == m["next_step"] == args.steps
        assert summary["current_step"] == summary["target_step"] == args.steps and not summary["stopped"]
        assert m["identity"]["inputs"] == inputs and state["offset"] == [ox,oy]
        identity = m["identity"]
        for name,digest_source in identity["implementation"].items():
            assert sha256(ROOT / "src/PyBCA" / name) == digest_source, name
        assert identity["global_prob"] == args.global_prob and identity["seed"] == args.seed
        assert identity["rng_mode"] == "independent" and identity["rng_version"] == "philox4x32-10-v1"
        assert not identity["state_gate_enable"] and identity["trial_constant_sweep"] is None
        if probe_manifest:
            for k in ("inputs", "implementation", "global_prob", "seed", "rng_mode", "rng_version"):
                assert identity[k] == probe_manifest["identity"][k]
        ids = m["trial_ids"]
        assert ids == identity["trial_ids"] == summary["trial_ids"]
        rank = summary["rank"]
        assert ids == list(range(rank*64,(rank+1)*64))
        assert all(0<=i<args.trials for i in ids)
        all_ids.extend(ids)
        cells = state["cells"][:,0].numpy()
        assert list(state["cells"].shape) == [len(ids),1,*grid.shape]
        assert np.isin(cells,[-1,0,1,2]).all()
        assert np.all((cells==0)==(grid==0)) and np.all((cells==-1)==(grid==-1))
        assert bool(torch.all(state["rule_probs"] == 1))
        for cell in cells:
            cells_sha.add(hashlib.sha256(cell.tobytes()).hexdigest())
        previous = records = 0
        memory = set()
        for chunk in m["chunks"]:
            f = directory / chunk["path"]
            raw = f.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == chunk["sha256"], f
            lines = raw.splitlines()
            assert len(lines)-1 == chunk["records"]
            assert chunk["start_step"] == previous and previous < chunk["next_step"] <= args.steps
            header = json.loads(lines[0])["__chunk__"]
            assert header["start_step"] == previous and header["next_step"] == chunk["next_step"]
            seen, reset_set, reset_clear = set(), set(), set()
            for line in lines[1:]:
                r = json.loads(line)
                t, step, name, count = r["trial"],r["step"],r["name"],r["count"]
                assert r["kind"] == "event" and t in ids and name in m["events"] and count == 1
                assert previous <= step < chunk["next_step"]
                key = (t,step,name)
                assert key not in seen
                seen.add(key)
                names_total[name] += count
                g = category(name)
                if g is not None:
                    totals[t,g] += count
                    binned[t,g,step//binsize] += count
                    first[t,g] = step+1 if first[t,g] == -1 else min(first[t,g],step+1)
                    last[t,g] = max(last[t,g],step+1)
                if name.startswith("Reset_"):
                    unit = name.split("_")[3]
                    (reset_set if name.endswith("_set") else reset_clear).add((t,step,unit))
                    if name.endswith("_set"):
                        reset_rows.append({"trial":t,"completed_updates":step+1,"unit":unit})
                elif name.startswith("Comparate_"):
                    comparison_rows.append({"trial":t,"completed_updates":step+1,"event":name})
                elif "_core_input_" in name:
                    td_rows.append((t,step+1,name[0]))
                if probe_manifest and t in probe_manifest["trial_ids"] and step < probe_manifest["next_step"]:
                    probe_rows.append((t,step,name,count))
                records += 1
            assert reset_set == reset_clear, (directory,previous,"reset pair mismatch")
            previous = chunk["next_step"]
            memory.add(chunk.get("memory",{}).get("cuda_allocated_bytes"))
        assert previous == args.steps
        audit = {"rank":rank,"checkpoint_sha256":digest,"steps":previous,"trials":len(ids),
                 "records":records,"chunks":len(m["chunks"]),"elapsed_sec":summary["elapsed_sec"],
                 "cuda_allocated_bytes":sorted(v for v in memory if v is not None),
                 "token_count_range":[int((cells==2).sum((1,2)).min()),int((cells==2).sum((1,2)).max())]}
        audits.append(audit)
        envs.append(m["environment"])
        checkpoints.append(str(path))
        print(json.dumps({"rank":rank,"checked_records":records,"group_events":totals[ids].sum(axis=0).tolist()}),flush=True)
    assert sorted(all_ids) == list(range(args.trials))
    if probe is not None:
        assert sorted(probe_rows) == probe
    group_summary = {}
    for j,name in enumerate(GROUPS):
        times = first[first[:,j]>=0,j]
        group_summary[name] = {"events":int(totals[:,j].sum()),"trials_with_event":int((totals[:,j]>0).sum()),
                               "count_per_trial_min_median_max":[int(totals[:,j].min()),float(np.median(totals[:,j])),int(totals[:,j].max())],
                               "first_completed_update_min_median_max":([int(times.min()),float(np.median(times)),int(times.max())] if len(times) else None),
                               "last_event_completed_update":int(last[:,j].max()) if len(times) else None}
    after_reset = set()
    reset_times = {}
    for r in reset_rows:
        key=(r["trial"],r["unit"])
        reset_times[key]=min(reset_times.get(key,args.steps+1),r["completed_updates"])
    for t,step,unit in td_rows:
        if step > reset_times.get((t,unit),args.steps+1):
            after_reset.add(t)
    result = {"checked_at":datetime.now(timezone.utc).isoformat(),"run":str(source),
              "trials":args.trials,"steps":args.steps,"global_prob":args.global_prob,"seed":args.seed,
              "groups":group_summary,"event_name_totals":dict(sorted(names_total.items())),
              "records":sum(x["records"] for x in audits),"checked_chunks":sum(x["chunks"] for x in audits),
              "unique_final_states":len(cells_sha),"audits":audits,"environments":envs,
              "trials_with_all_six_event_groups":int(np.all(totals>0,axis=1).sum()),
              "trials_with_td_input_after_first_reset_of_same_unit":len(after_reset),
              "prior_probe_prefix_exact_match":True if probe is not None else None,
              "prior_probe_event_records":len(probe) if probe is not None else None,
              "reset_set_clear_pairs_match":True,"errors":[],
              "optimum_hit_statistics":None,
              "optimum_limitation":"No certified state decoder or per-step first-hit observer was part of this run. Module outputs do not establish an optimum."}
    (args.output / "audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.output / "reset-events.json").write_text(json.dumps(reset_rows,indent=2)+"\n")
    (args.output / "comparator-events.json").write_text(json.dumps(comparison_rows,indent=2)+"\n")
    np.savez_compressed(args.output / "event-counts.npz",totals=totals,first=first,last=last,
                        bins=binned,bin_width=binsize,groups=np.array(GROUPS))
    print(json.dumps({"groups":group_summary,"records":result["records"],
                      "probe_prefix_match":result["prior_probe_prefix_exact_match"],
                      "reset_then_td_trials":len(after_reset),"errors":[]}),flush=True)


if __name__ == "__main__":
    main()
