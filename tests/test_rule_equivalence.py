"""All supplied rules against a coordinate/set oracle, legacy and new cores.

Run explicitly (the full audit is deliberately separate from quick unittests):
  PYTHONPATH=src python tests/test_rule_equivalence.py --device cpu --output results/rules-cpu.json
  PYTHONPATH=src python tests/test_rule_equivalence.py --device cuda --output results/rules-cuda.json

Candidate injection isolates application semantics from matching/RNG. Separate
real update tests cover their composition. This is finite local exhaustiveness,
not a claim to enumerate every possible complete cell space or random history.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from PyBCA._legacy.cli_simClass import BCA_Simulator as Legacy
from PyBCA._legacy import lib as legacy_io
from PyBCA.core import io
from PyBCA.core import random as counter_rng
from PyBCA.core.optimized import CandidatePlan
from PyBCA.core.simulator import BCA_Simulator as Reference

ROOT = Path(__file__).resolve().parents[1]


def write_offsets(rule):
    # Deliberately derive the definition directly, without simulator helpers.
    return [(y-1, x-1, int(rule[1, y, x])) for y in range(3) for x in range(3)
            if rule[1, y, x] != rule[0, y, x] and rule[1, y, x] != -1]


def oracle_match(cells, rules):
    t, h, w = cells.shape
    mask = np.ones((t, len(rules), h, w), dtype=bool)
    for r, (pre, _) in enumerate(rules):
        for y in range(h):
            for x in range(w):
                for py, px in itertools.product(range(3), repeat=2):
                    if py != 1 and px != 1 and pre[py, px] == 0:
                        continue
                    yy, xx = y+py-1, x+px-1
                    value = cells[:, yy, xx] if 0 <= yy < h and 0 <= xx < w else 0
                    mask[:, r, y, x] &= value == pre[py, px]
    return mask


def oracle_apply(cells, rules, mask, order, written=None):
    """Enumerate clipped target sets; no convolution or candidate-plan calls.

    Vectorize ONLY over independent test cases. Each center and each target is
    explicitly visited. Count all original candidates before rejecting any.
    """
    t, h, w = cells.shape
    out, live = cells.copy(), mask.copy()
    used = np.zeros_like(cells, dtype=bool) if written is None else written.copy()
    for r in order:
        targets = {}
        multiplicity = np.zeros_like(cells, dtype=np.int16)
        for y, x in itertools.product(range(h), range(w)):
            targets[y, x] = [(y+dy, x+dx, value) for dy, dx, value in write_offsets(rules[r])
                             if 0 <= y+dy < h and 0 <= x+dx < w]
            for yy, xx, _ in targets[y, x]:
                multiplicity[:, yy, xx] += mask[:, r, y, x]
        for (y, x), destinations in targets.items():
            for yy, xx, _ in destinations:
                live[:, r, y, x] &= (multiplicity[:, yy, xx] == 1) & ~used[:, yy, xx]
        # No writes occur until all decisions for this rule have been made.
        for (y, x), destinations in targets.items():
            keep = live[:, r, y, x]
            for yy, xx, value in destinations:
                out[keep, yy, xx] = value
                used[keep, yy, xx] = True
    return out, live, used


def make_sim(cls, cells, rules, device, probs=None):
    """Use the production allocator, but accept small in-memory fixtures."""
    s = cls.__new__(cls)
    s.device, s._current_step = device, 0
    s.execution_mode, s.rng_mode, s.candidate_plan = "reference", "legacy", None
    s.trial_ids, s.trial_offset, s.candidate_capacity = None, 0, 4096
    s.quiet, s.use_tqdm, s.gui_mode = True, True, False
    s.history_recorder, s.trial_constant_sweep = None, None
    s.record_rule_history, s.rule_history_rule_ids = True, None
    s.rule_ids = list(range(len(rules)))
    s.rule_arrays_tensor = torch.as_tensor(rules.copy(), dtype=torch.int8, device=device)
    s.rule_probs_tensor = torch.as_tensor(np.ones(len(rules)) if probs is None else probs,
                                          dtype=torch.float32, device=device).clone()
    s.rule_probs_base_tensor = s.rule_probs_tensor.clone()
    s.cellspace_tensor = torch.as_tensor(cells[0].copy(), dtype=torch.int8, device=device)
    s.rng = torch.Generator(device=device)
    s.offset_x = s.offset_y = 0
    s.spatial_event_arrays = s.spatial_event_arrays_tensor = s.spatial_event_names = None
    s.state_conversions = s.state_conversions_tensor = None
    s.set_ParallelTrial(len(cells))
    s.TCHW[:, 0].copy_(torch.as_tensor(cells, device=device))
    return s


def equal(actual, expected, label):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if not np.array_equal(actual, expected):
        pos = tuple(np.argwhere(actual != expected)[0])
        raise AssertionError(f"{label}: first mismatch {pos}: {actual[pos]} != {expected[pos]}")


class Audit:
    def __init__(self, device, output):
        self.device, self.output = device, output
        self.mode = "cuda" if device.startswith("cuda") else "torch_sparse"
        self.report = {"schema": "pybca-rule-audit-v1", "device": device,
                       "torch": torch.__version__, "rules": [], "sections": {}, "passed": False}
        if device.startswith("cuda"):
            self.report["gpu"] = torch.cuda.get_device_name(device)
        self.entries = []
        for path in sorted((ROOT/"Sample/rule").glob("*.yaml")):
            loaded = io.load_transition_rules_yaml(str(path))
            old = legacy_io.load_transition_rules_yaml(str(path))
            assert len(loaded) == len(old)
            for ordinal, (r, legacy_rule) in enumerate(zip(loaded, old)):
                a = np.stack([r.prev_pattern, r.next_pattern])
                equal(a, np.stack([legacy_rule.prev_pattern, legacy_rule.next_pattern]), "YAML legacy load")
                assert r.rule_id == legacy_rule.rule_id and r.probability == legacy_rule.probability
                row = {"file": str(path.relative_to(ROOT)), "id": int(r.rule_id), "ordinal": ordinal,
                       "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "probability": r.probability, "pattern_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                self.entries.append((a, row))
                self.report["rules"].append(row)
        sources = list((ROOT/"src/PyBCA/core").rglob("*.py")) + list((ROOT/"src/PyBCA/core").glob("*.cu"))
        sources += [ROOT/"src/PyBCA/_legacy/cli_simClass.py", ROOT/"src/PyBCA/_legacy/lib.py", Path(__file__)]
        self.report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(sources)}

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2)+"\n")

    def application(self, label, cells, rules, mask, order, written=None, capacities=(4096,)):
        expected = oracle_apply(cells, rules, mask, order, written)
        for cls in (Legacy, Reference):
            sim = make_sim(cls, cells, rules, self.device)
            sim.TNHW_boolMask.copy_(torch.as_tensor(mask, device=self.device))
            if written is not None:
                sim.TCHW_applied[:, 0].copy_(torch.as_tensor(written, device=self.device))
            for index, r in enumerate(order):
                sim.TNHW_boolMask[:, r] = sim._rule_inner_conflict_resolution(r)
                sim.TNHW_boolMask[:, r] = sim._rule_outer_conflict_resolution(r)
                if cls is Reference:
                    sim._record_rule_history(r)
                sim._write_back(r)
                # Compare after each applied rule, as well as the final result.
                prefix = oracle_apply(cells, rules, mask, order[:index+1], written)
                self.check(sim, prefix, f"{label}/{cls.__module__}/prefix{index+1}")
            if cls is Reference:
                for t, history in enumerate(sim.rule_history):
                    for r in order:
                        assert len(history[f"rule_{r}"]) == int(expected[1][t, r].sum()), label
        for capacity in capacities:
            sim = make_sim(Reference, cells, rules, self.device)
            plan = CandidatePlan(sim, self.mode, "legacy", candidate_capacity=capacity)
            sim.TNHW_boolMask.copy_(torch.as_tensor(mask, device=self.device))
            if written is not None:
                sim.TCHW_applied[:, 0].copy_(torch.as_tensor(written, device=self.device))
            if plan.cuda is None:
                for index, r in enumerate(order):
                    plan._resolve(r)
                    self.check(sim, oracle_apply(cells, rules, mask, order[:index+1], written),
                               f"{label}/sparse/prefix{index+1}")
            else:
                plan.cuda.order.copy_(torch.as_tensor(order, device=self.device)[None])
                plan.cuda.launch(1)
                plan.cuda.launch(2, independent=False)
                equal(plan.cuda.rule_counts, expected[1].sum(axis=(2, 3)), label+"/CUDA counts")
            self.check(sim, expected, f"{label}/{self.mode}/capacity{capacity}")
        return expected

    @staticmethod
    def check(sim, expected, label):
        for actual, value, name in zip((sim.TCHW[:, 0], sim.TNHW_boolMask, sim.TCHW_applied[:, 0]),
                                       expected, ("cells", "accepted", "written")):
            equal(actual, value, label+"/"+name)

    def matching_and_firing(self):
        for rule, row in self.entries:
            # All 256 int8 values at each of the 9 pattern positions. This tests
            # required equality and wildcard corners without assuming state range.
            cells = np.full((9*256+49, 7, 7), -128, np.int8)
            cells[:, 2:5, 2:5] = rule[0]
            required = (rule[0] != 0)
            required[1, :] = required[:, 1] = True
            expected_center = []
            for q in range(9):
                cells[q*256:(q+1)*256, 2+q//3, 2+q%3] = np.arange(-128, 128, dtype=np.int16)
                expected_center.extend((np.arange(-128, 128) == rule[0].flat[q]) if required.flat[q]
                                       else np.ones(256, bool))
            for k, (y, x) in enumerate(itertools.product(range(7), repeat=2)):
                grid = cells[9*256+k]
                grid.fill(-128)
                for py, px in itertools.product(range(3), repeat=2):
                    if 0 <= y+py-1 < 7 and 0 <= x+px-1 < 7:
                        grid[y+py-1, x+px-1] = rule[0, py, px]
            rules = rule[None]
            expected = oracle_match(cells, rules)
            equal(expected[:9*256, 0, 3, 3], np.array(expected_center), "oracle equality/wildcards")
            for cls in (Legacy, Reference):
                sim = make_sim(cls, cells, rules, self.device)
                equal(sim._match_centers_all_rules(), expected, str(row)+"/match/"+cls.__module__)
            plan = CandidatePlan(sim, self.mode)
            if plan.cuda is None:
                actual = plan.match()
            else:
                plan.cuda.launch(0, independent=False)
                actual = sim.TNHW_boolMask
            equal(actual, expected, str(row)+"/match/"+self.mode)
            row["matching_cases"] = len(cells)
            row["positive_matching_centers"] = int(expected.sum())

            # A real update, with a single isolated actual rule guaranteed to fire.
            positive = cells[9*256+24:9*256+25].copy()
            original = oracle_match(positive, rules)
            assert original[0, 0, 3, 3]
            out = oracle_apply(positive, rules, original, [0])
            assert out[1][0, 0, 3, 3], row
            for cls, mode in ((Legacy, None), (Reference, None), (Reference, self.mode)):
                s = make_sim(cls, positive, rules, self.device)
                if mode:
                    s.candidate_plan = CandidatePlan(s, mode)
                s.rng.manual_seed(17)
                s.update_cellspace(1.)
                self.check(s, out, str(row)+"/actual fired update")
            row["forced_firing_verified"] = True

            # Exercise the original probability (including .001) plus 0/.37/1,
            # global gates, RNG fast paths, final generator state and counts.
            for prob, global_prob in ((row["probability"], .63), (0., 1.), (.37, 1.), (1., 0.), (1., 1.)):
                batch = np.repeat(positive, 128, axis=0)
                sims = [make_sim(cls, batch, rules, self.device, [prob]) for cls in (Legacy, Reference, Reference)]
                sims[-1].candidate_plan = CandidatePlan(sims[-1], self.mode)
                for s in sims:
                    s.rng.manual_seed(781)
                    s.update_cellspace(global_prob)
                for s in sims[1:]:
                    self.check(s, tuple(x.cpu().numpy() for x in
                               (sims[0].TCHW[:, 0], sims[0].TNHW_boolMask, sims[0].TCHW_applied[:, 0])), str(row)+"/probability")
                    equal(s.rng.get_state(), sims[0].rng.get_state().cpu().numpy(), "legacy RNG consumption")
                assert sims[1].rule_history == sims[2].rule_history
            row["legacy_probability_cases"] = 5*128
            row["status"] = "matching-and-real-update-passed"
        self.report["sections"]["matching_and_firing"] = "passed"

    def local_conflicts(self):
        # All 512 subsets of centers, in a 3x3 domain and the same neighborhood
        # translated to interior/each boundary orientation of a larger domain.
        bits = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(bool).reshape(512, 3, 3)
        anchors = [(y, x) for y, x in itertools.product((0, 2, 4), repeat=2)]
        for rule, row in self.entries:
            total = 0
            for h, placements in ((3, [(0, 0)]), (7, anchors)):
                mask = np.zeros((512*len(placements), 1, h, h), bool)
                for i, (y, x) in enumerate(placements):
                    mask[i*512:(i+1)*512, 0, y:y+3, x:x+3] = bits
                cells = np.full((len(mask), h, h), 37, np.int8)
                self.application(str(row)+"/all-local-candidates", cells, rule[None], mask, [0], capacities=(1, 4096))
                total += len(mask)
            row["candidate_subset_cases"] = total
            # For every possible center in a 3x3 domain enumerate ALL subsets of
            # its clipped targets as previously written. Includes no target left
            # in the domain and every combination of outer conflicts.
            masks, written = [], []
            for y, x in itertools.product(range(3), repeat=2):
                targets = [(y+dy, x+dx) for dy, dx, _ in write_offsets(rule)
                           if 0 <= y+dy < 3 and 0 <= x+dx < 3]
                for bits_ in range(2**len(targets)):
                    m, a = np.zeros((1, 3, 3), bool), np.zeros((3, 3), bool)
                    m[0, y, x] = True
                    for j, (yy, xx) in enumerate(targets):
                        a[yy, xx] = bool(bits_ & (1 << j))
                    masks.append(m)
                    written.append(a)
            mask = np.stack(masks)
            self.application(str(row)+"/all-prior-target-subsets", np.full((len(mask), 3, 3), 37, np.int8),
                             rule[None], mask, [0], np.stack(written))
            row["previously_written_subset_cases"] = len(mask)
            # Cross inner conflicts with every single previously written cell.
            # Outer rejection is an OR over these cells; the isolated-center
            # test above separately checks every combination of target bits.
            mask = np.tile(bits[:, None], (10, 1, 1, 1))
            written = np.zeros((len(mask), 3, 3), bool)
            for cell in range(9):
                written[(cell+1)*512:(cell+2)*512, cell//3, cell%3] = True
            self.application(str(row)+"/inner-outer-product", np.full((len(mask), 3, 3), 37, np.int8),
                             rule[None], mask, [0], written, capacities=(1, 4096))
            row["inner_outer_combined_cases"] = len(mask)
        self.report["sections"]["local_conflicts"] = "passed"

    def independent_pipeline(self):
        groups = {}
        for rule, row in self.entries:
            groups.setdefault(row["file"], []).append((rule, row))
        total = 0
        for filename, entries in groups.items():
            rules = np.stack([rule for rule, _ in entries])
            # Each actual rule has a positive patch in a distinct trial. T=N
            # deliberately also tests canonical independent [T,N] probabilities.
            n = len(rules)
            cells = np.full((n, 7, 7), -128, np.int8)
            cells[:, 2:5, 2:5] = rules[:, 0]
            matched = oracle_match(cells, rules)
            ids = [2**63+31+i*19 for i in range(n)]
            keys = counter_rng.trial_keys(2**64-1, ids)
            for case, global_prob in enumerate((1., .53, 0.)):
                probs = (np.ones((n, n), np.float32) if case == 0 else
                         np.tile(np.array([row["probability"] for _, row in entries], np.float32), (n, 1)))
                if case == 1:
                    probs[::3] = .37
                    probs[1::3, ::2] = 0.
                step = 2**32+case
                gated = matched.copy()
                for t, r, y, x in np.argwhere(matched):
                    gated[t, r, y, x] &= (
                        counter_rng.uniform(keys[t], y*7+x, step, r, counter_rng.GLOBAL) < global_prob and
                        counter_rng.uniform(keys[t], y*7+x, step, r, counter_rng.RULE) < probs[t, r])
                orders = counter_rng.permutations(keys, n, step)
                expected = [np.empty_like(cells), np.empty_like(gated), np.empty_like(cells, dtype=bool)]
                for t, order in enumerate(orders):
                    out = oracle_apply(cells[t:t+1], rules, gated[t:t+1], list(order))
                    for a, b in zip(expected, out):
                        a[t:t+1] = b
                # Compare fused CUDA matching/gates and GPU permutations before
                # comparing the complete independent update and recorded counts.
                s = make_sim(Reference, cells, rules, self.device, probs)
                s._current_step = step
                s.candidate_plan = plan = CandidatePlan(s, self.mode, "independent", ids)
                plan.set_seed(2**64-1)
                if plan.cuda is None:
                    equal(plan._independent_gate(plan.match(), global_prob), gated, filename+"/independent gates")
                else:
                    plan.cuda.launch(0, global_prob, independent=True)
                    equal(s.TNHW_boolMask, gated, filename+"/independent fused gates")
                s.update_cellspace(global_prob)
                self.check(s, expected, filename+"/independent pipeline")
                if plan.cuda is not None:
                    equal(plan.cuda.order, orders, filename+"/trial orders")
                    equal(plan.cuda.rule_counts, expected[1].sum(axis=(2, 3)), filename+"/independent counts")
                for t, history in enumerate(s.rule_history):
                    for r in range(n):
                        assert len(history[f"rule_{r}"]) == int(expected[1][t, r].sum())
                total += n
            for _, row in entries:
                row["independent_pipeline_cases"] = 3
        self.report["sections"]["independent_pipeline"] = {"trials": total, "passed": True}

    def rule_pairs(self):
        # Every ordered file+ordinal pair, including both application orders and
        # pairs from alternative YAML files. All displacements that can overlap
        # a 3x3 write footprint (-2..2) plus four definitely disjoint controls.
        offsets = list(itertools.product(range(-2, 3), repeat=2)) + [(-3, 0), (3, 0), (0, -3), (0, 3)]
        masks = []
        for y, x in itertools.product((0, 3, 6), repeat=2):
            for dy, dx in offsets:
                if 0 <= y+dy < 7 and 0 <= x+dx < 7:
                    m = np.zeros((2, 7, 7), bool)
                    m[0, y, x] = m[1, y+dy, x+dx] = True
                    masks.append(m)
        mask = np.stack(masks)
        cells = np.full((len(mask), 7, 7), 37, np.int8)
        pairs = cases = conflicts = 0
        for i, (first, row) in enumerate(self.entries):
            for j, (second, _) in enumerate(self.entries):
                expected = self.application(f"ordered-pair {i},{j}", cells, np.stack([first, second]), mask, [0, 1])
                conflicts += int((~expected[1][:, 1].any(axis=(1, 2))).sum())
                pairs += 1
                cases += len(mask)
            row["ordered_pair_partners"] = len(self.entries)
            if i % 8 == 0:
                print(f"ordered rule pairs: {pairs}/{len(self.entries)**2}", flush=True)
                self.save()
        self.report["sections"]["rule_pairs"] = {"ordered_pairs": pairs, "cases": cases,
                                                  "outer_rejections": conflicts, "passed": True}

    def witnesses_and_conversions(self):
        # Snapshot semantics: A creates (or removes) B's precondition while their
        # write targets differ. B must use the pre-step match in both directions.
        witness_count = 0
        for before, after in ((0, 2), (2, 0)):
            rules = np.zeros((2, 2, 3, 3), np.int8)
            rules[:, 0, 1, 1] = 1
            rules[0, 0, 1, 2] = before
            rules[1, 0, 1, 2] = 2
            rules[:, 1] = rules[:, 0]
            rules[0, 1, 1, 2] = after
            rules[1, 1, 1, 1] = 3
            cells = np.zeros((1, 5, 5), np.int8)
            cells[0, 2, 2], cells[0, 2, 3] = 1, before
            matches = oracle_match(cells, rules)
            expected = oracle_apply(cells, rules, matches, [0, 1])
            assert expected[0][0, 2, 2] == (3 if before == 2 else 1)
            for cls, mode in ((Legacy, None), (Reference, None), (Reference, self.mode)):
                s = make_sim(cls, cells, rules, self.device)
                if mode:
                    s.candidate_plan = CandidatePlan(s, mode)
                with patch("torch.randperm", return_value=torch.tensor([0, 1], device=self.device)):
                    s.update_cellspace(1.)
                self.check(s, expected, "snapshot witness")
            witness_count += 1
        # Inner rejection must include centers also destined for outer rejection;
        # resolving outer first would wrongly rescue a colliding center.
        rules = np.zeros((1, 2, 3, 3), np.int8)
        rules[0, 1, 1, 1:3] = [1, 2]
        mask = np.zeros((1, 1, 3, 4), bool)
        mask[0, 0, 1, 1:3] = True
        written = np.zeros((1, 3, 4), bool)
        written[0, 1, 1] = True
        result = self.application("inner-before-outer", np.zeros((1, 3, 4), np.int8), rules, mask, [0], written)
        assert not result[1].any()
        # Explicit no-write, zero writes, post=-1 and negative/int8 extremes.
        for value in (None, -1, 0, -128, 127):
            rule = np.ones((1, 2, 3, 3), np.int8)
            if value is not None:
                rule[0, 1] = value
            mask = np.ones((1, 1, 3, 3), bool)
            self.application(f"write-special-{value}", np.full((1, 3, 3), 23, np.int8), rule, mask, [0], capacities=(1, 4096))
        conversions = []
        for path in sorted((ROOT/"Sample/rule").glob("*.yaml")):
            rows = io.load_multiple_state_conversions([str(path)])
            if not len(rows):
                continue
            values = np.arange(-128, 128, dtype=np.int16).astype(np.int8).reshape(1, 16, 16)
            expected = values.copy()
            for old, new in rows:
                expected[values == old] = new
            for cls, mode in ((Legacy, None), (Reference, None), (Reference, self.mode)):
                s = make_sim(cls, values, np.zeros((1, 2, 3, 3), np.int8), self.device)
                s.state_conversions, s.state_conversions_tensor = rows, torch.as_tensor(rows, device=self.device)
                if mode:
                    CandidatePlan(s, mode).state_gates()
                else:
                    s.apply_state_gates()
                equal(s.TCHW[:, 0], expected, "state conversion all int8")
                equal(s.TCHW_applied[:, 0], expected != values, "state conversion written")
            conversions.append({"file": str(path.relative_to(ROOT)), "rows": rows.tolist(), "values_tested": 256})
        self.report["sections"]["witnesses_and_conversions"] = {"snapshot_cases": witness_count,
            "inner_before_outer": True, "special_write_cases": 5, "state_conversions": conversions, "passed": True}

    def run(self):
        start = time.monotonic()
        try:
            for method in (self.matching_and_firing, self.local_conflicts, self.rule_pairs,
                           self.independent_pipeline, self.witnesses_and_conversions):
                print(f"Starting {method.__name__} ({len(self.entries)} rule entries, {self.device})", flush=True)
                method()
                self.save()
            for _, row in self.entries:
                row["status"] = "passed"
            self.report["passed"] = True
        except Exception as exc:
            self.report["error"] = repr(exc)
            raise
        finally:
            self.report["elapsed_seconds"] = time.monotonic()-start
            self.save()
        print(json.dumps({"passed": True, "rules": len(self.entries), "elapsed_seconds": self.report["elapsed_seconds"]}), flush=True)


class OracleSanityTests(unittest.TestCase):
    def test_collision_rejects_both_centers_without_reconsidering_loser(self):
        rules = np.zeros((1, 2, 3, 3), np.int8)
        rules[0, 1, 1, 1:3] = [1, 2]
        mask = np.zeros((1, 1, 3, 4), bool)
        mask[0, 0, 1, 1:3] = True
        cells = np.zeros((1, 3, 4), np.int8)
        out, live, written = oracle_apply(cells, rules, mask, [0])
        self.assertFalse(live.any())
        self.assertFalse(written.any())
        self.assertTrue(np.array_equal(out, cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    Audit(args.device, args.output).run()
