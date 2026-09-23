"""Low-memory matching and candidate-based conflict resolution.

All matches are computed from the start-of-step state. Inner conflicts reject
every center contributing to a shared target; outer conflicts reject a center
if any of its targets was written by an earlier rule. Boundaries are zero.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import random as counter_rng


class CandidatePlan:
    def __init__(self, sim, mode="torch_sparse", rng_mode="legacy", trial_ids=None,
                 candidate_capacity=4096):
        self.sim = sim
        self.mode = mode
        self.rng_mode = rng_mode
        self.trial_ids = list(range(sim.parallel_trial)) if trial_ids is None else list(trial_ids)
        self.seed = None
        rules = sim.rule_arrays_tensor.detach().cpu()
        pre = rules[:, 0]
        required = (pre != 0) | sim.rule_mask.cpu()[None]
        self.match_plan = [(y, x, pre[:, y, x].to(sim.device)[None, :, None, None],
                            required[:, y, x].to(sim.device)[None, :, None, None])
                           for y in range(3) for x in range(3) if required[:, y, x].any()]
        self.write_plan = []
        for pre_r, post_r in rules:
            loc = ((pre_r != post_r) & (post_r != -1)).nonzero()
            self.write_plan.append(((loc - 1).to(sim.device), post_r[loc[:, 0], loc[:, 1]].to(sim.device)))
        self.cuda = None
        if mode == "cuda":
            from .cuda_backend import CudaPlan
            self.cuda = CudaPlan(sim, candidate_capacity)
        self._prepare_events()

    def set_seed(self, seed):
        if self.seed != seed:
            self.keys = counter_rng.trial_keys(int(seed), self.trial_ids)
            if self.cuda is not None:
                self.cuda.set_keys(self.keys)
            self.seed = int(seed)

    def match(self):
        s = self.sim
        t, _, h, w = s.TCHW.shape
        padded = F.pad(s.TCHW[:, 0], (1, 1, 1, 1), value=0)
        result = torch.ones((t, len(s.rule_ids), h, w), dtype=torch.bool, device=s.device)
        for y, x, value, required in self.match_plan:
            result.logical_and_((padded[:, None, y:y+h, x:x+w] == value) | ~required)
        return result

    def _independent_gate(self, mask, global_prob):
        s = self.sim
        t, n, h, w = mask.shape
        if not np.isscalar(global_prob):
            raise ValueError("Independent RNG requires a scalar global_prob")
        probs = s.rule_probs_tensor.detach().cpu().numpy()
        if probs.shape == (n,):
            probs = np.broadcast_to(probs, (t, n))
        if probs.shape != (t, n):
            raise ValueError("Independent rule probabilities must be [N] or [T,N]")
        hits = mask.nonzero()
        a = hits.cpu().numpy()
        if len(a):
            ti, ri, y, x = a.T
            ids = y * w + x
            keep = counter_rng.uniform(self.keys[ti], ids, s._current_step, ri, counter_rng.GLOBAL) < global_prob
            keep &= counter_rng.uniform(self.keys[ti], ids, s._current_step, ri, counter_rng.RULE) < probs[ti, ri]
            rejected = hits[torch.as_tensor(~keep, device=s.device)]
            mask[tuple(rejected.T)] = False
        return mask

    def _resolve(self, r, trials=None):
        s = self.sim
        mask = s.TNHW_boolMask[:, r]
        if trials is None:
            centers = mask.nonzero()
        else:
            centers = mask[trials:trials+1].nonzero()
            centers[:, 0] += trials
        offsets, values = self.write_plan[r]
        k, m = len(centers), len(offsets)
        if k and m:
            _, _, h, w = s.TCHW.shape
            ys = centers[:, 1, None] + offsets[None, :, 0]
            xs = centers[:, 2, None] + offsets[None, :, 1]
            valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            dest = ((centers[:, 0, None] * h + ys) * w + xs)[valid]
            owners = torch.arange(k, device=s.device)[:, None].expand(k, m)[valid]
            vals = values[None].expand(k, m)[valid]
            rejected = torch.zeros(k, dtype=torch.bool, device=s.device)
            _, inverse, counts = torch.unique(dest, return_inverse=True, return_counts=True)
            rejected[owners[counts[inverse] >= 2]] = True
            rejected[owners[s.TCHW_applied.view(-1)[dest]]] = True
            live_targets = ~rejected[owners]
            dead = centers[rejected]
            mask[tuple(dead.T)] = False
            s.TCHW.view(-1)[dest[live_targets]] = vals[live_targets]
            s.TCHW_applied.view(-1)[dest[live_targets]] = True

    def update(self, global_prob):
        s = self.sim
        s.TCHW_applied.zero_()
        if self.cuda is not None:
            self.cuda.update(global_prob, self.rng_mode)
        else:
            s.TNHW_boolMask = self.match()
            if self.rng_mode == "legacy":
                if global_prob is not None and global_prob != 1:
                    s.TNHW_boolMask = s._global_prob_gate(global_prob)
                # Check the compact probabilities, not their expanded grid.
                if not bool(torch.all(s.rule_probs_tensor == 1)):
                    s.TNHW_boolMask = s._rule_prob_gate()
                order = torch.randperm(len(s.rule_ids), generator=s.rng, device=s.device).cpu().tolist()
                for r in order:
                    self._resolve(r)
            else:
                self._independent_gate(s.TNHW_boolMask, global_prob)
                order = counter_rng.permutations(self.keys, len(s.rule_ids), s._current_step)
                for trial, row in enumerate(order):
                    for r in row:
                        self._resolve(int(r), trial)
        # Final centers suffice for counts, independent of rule visitation order.
        if s.record_rule_history:
            if self.cuda is not None:
                counts = self.cuda.rule_counts
                if s.history_recorder is not None:
                    s.history_recorder.record_rule_counts(counts)
                else:
                    s._append_rule_counts(counts)
            else:
                for r in range(len(s.rule_ids)):
                    s._record_rule_history(r)
        return s.TCHW

    def _prepare_events(self):
        s = self.sim
        events = s.spatial_event_arrays
        t, _, h, w = s.TCHW.shape
        self.event_info = []
        if events is not None:
            for i, ev in enumerate(events):
                x, y, value, wx, wy, new_value = map(int, ev[:6])
                x, wx, y, wy = x-s.offset_x, wx-s.offset_x, y-s.offset_y, wy-s.offset_y
                if 0 <= x < w and 0 <= y < h and 0 <= wx < w and 0 <= wy < h:
                    self.event_info.append((i, y*w+x, value, wy*w+wx, new_value,
                                            float(ev[6]) if len(ev) >= 7 else 1.,
                                            int(ev[7]) if len(ev) >= 9 else -1,
                                            int(ev[8]) if len(ev) >= 9 else -1))
        if self.cuda is not None:
            self.cuda.prepare_events(self.event_info)

    def events(self):
        s = self.sim
        if self.cuda is not None:
            hits = self.cuda.events()
        else:
            e = len(s.spatial_event_names or [])
            hits = torch.zeros((s.parallel_trial, e), dtype=torch.bool, device=s.device)
            flat = s.TCHW[:, 0].flatten(1)
            # Evaluate all conditions before any write.
            for i, src, value, dst, new_value, prob, start, end in self.event_info:
                if (start < 0 or s._current_step >= start) and (end < 0 or s._current_step <= end):
                    rnd = counter_rng.uniform(self.keys, i, s._current_step, domain=counter_rng.EVENT)
                    hits[:, i] = (flat[:, src] == value) & torch.as_tensor(rnd < prob, device=s.device)
            for i, src, value, dst, new_value, prob, start, end in self.event_info:
                flat[:, dst] = torch.where(hits[:, i], new_value, flat[:, dst])
                s.TCHW_applied.view(s.parallel_trial, -1)[:, dst] |= hits[:, i]
        if s.history_recorder is not None:
            s.history_recorder.record_events(hits)
        elif s.event_history is not None:
            for trial, event in hits.nonzero().cpu().tolist():
                s.event_history[trial][s.spatial_event_names[event]].append(s._current_step)

    def state_gates(self):
        if self.cuda is not None:
            self.cuda.state_gates()
        else:
            self.sim.apply_state_gates()
