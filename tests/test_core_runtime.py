"""Semantic tests, runnable with pytest or stdlib unittest (including on rokko)."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from PyBCA.api import Config, Engine
from PyBCA.api.streaming import HistoryWriter, iter_history
from PyBCA.core.random import philox_u32, trial_keys, uniform, permutations
from PyBCA.core.simulator import BCA_Simulator

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("PYBCA_TEST_DEVICE", "cpu")
MODE = "cuda" if DEVICE.startswith("cuda") else "torch_sparse"
torch.set_num_threads(1)


def config(**kw):
    return replace(Config(cellspace_path=str(ROOT/"Sample/Cellspace/test.yaml"),
        rule_paths=[str(ROOT/"Sample/rule/base-rule.yaml")],
        spatial_event_file_path=str(ROOT/"Sample/Specialevent/test_event.py"),
        device=DEVICE, execution_mode=MODE, rng_mode="independent", seed=31,
        trials=3, steps=12, quiet=True, use_tqdm="false", record_rule_history=True), **kw)


def assert_sim(test, a, b, histories=True):
    for name in ("TCHW", "TNHW_boolMask", "TCHW_applied"):
        test.assertTrue(torch.equal(getattr(a, name).cpu(), getattr(b, name).cpu()), name)
    test.assertEqual(a._current_step, b._current_step)
    if histories:
        test.assertEqual(a.event_history, b.event_history)
        test.assertEqual(a.rule_history, b.rule_history)


class RandomTests(unittest.TestCase):
    def test_published_philox_vector(self):
        self.assertEqual(int(philox_u32([0, 0], 0, 0, 0, 0)), 0x6627E8D5)

    def test_domains_and_distribution(self):
        keys = trial_keys(1234, [71])
        values = uniform(keys, np.arange(100000), 991)
        self.assertLess(abs(float((values < .3).mean())-.3), .006)
        self.assertFalse(np.array_equal(values, uniform(keys, np.arange(100000), 991, domain=2)))
        rows = permutations(trial_keys(42, range(200)), 28, 3)
        self.assertEqual(np.unique(rows, axis=0).shape[0], 200)
        for row in rows:
            self.assertEqual(sorted(row), list(range(28)))

    def test_partition_invariance(self):
        all_run = Engine(config(trial_ids=[18, 99, 700])).run().simulator
        for index, trial in enumerate([18, 99, 700]):
            one = Engine(config(trials=1, trial_ids=[trial])).run().simulator
            self.assertTrue(torch.equal(all_run.TCHW[index:index+1], one.TCHW))
            self.assertEqual(all_run.event_history[index], one.event_history[0])
            self.assertEqual(all_run.rule_history[index], one.rule_history[0])

    def test_sweep_partition_invariance(self):
        c = config(rule_paths=[str(ROOT/"Sample/rule/base-rule.yaml"), str(ROOT/"Sample/rule/Join_fork.yaml")],
                   trial_constant_sweep={"join_err_0_input": {"base": .13, "delta": .17}}, trials=3)
        batch = Engine(c).state.simulator
        for index in range(3):
            one = Engine(replace(c, trials=1, trial_ids=[index], trial_offset=index)).state.simulator
            self.assertTrue(torch.equal(batch.rule_probs_tensor[index], one.rule_probs_tensor[0]))

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA required")
    def test_cpu_gpu_counter_decisions_and_trajectory(self):
        cpu = Engine(config(device="cpu", execution_mode="torch_sparse")).run().simulator
        gpu = Engine(config()).run().simulator
        assert_sim(self, cpu, gpu)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA required")
    def test_high_counter_and_key_words_on_gpu(self):
        ids = [2**63+1, 2**64-1, 89]
        c = config(seed=2**64-1, trial_ids=ids)
        s = Engine(c).state.simulator
        s._current_step = 2**32+17
        s.candidate_plan.set_seed(c.seed)
        s.candidate_plan.cuda.launch(2)
        expected = permutations(trial_keys(c.seed, ids), len(s.rule_ids), s._current_step)
        np.testing.assert_array_equal(s.candidate_plan.cuda.order.cpu().numpy(), expected)


class CandidateTests(unittest.TestCase):
    def test_bca_ip_all_event_coordinates(self):
        c = config(cellspace_path=str(ROOT/"Sample/Cellspace/BCA-IP.yaml"),
                   spatial_event_file_path=str(ROOT/"Sample/Specialevent/BCA-IP_event.py"),
                   trials=1, execution_mode="reference", rng_mode="legacy")
        ref = Engine(c).state.simulator
        for row in ref.spatial_event_arrays:
            x, y, value = map(int, row[:3])
            ref.TCHW[:, 0, y-ref.offset_y, x-ref.offset_x] = value
        new = copy.deepcopy(ref)
        from PyBCA.core.optimized import CandidatePlan
        new.candidate_plan = CandidatePlan(new, MODE, "independent")
        new.candidate_plan.set_seed(c.seed)
        ref.apply_spatial_events()
        new.candidate_plan.events()
        assert_sim(self, ref, new)
        fired = sum(len(v) for v in ref.event_history[0].values())
        self.assertEqual(fired, len(ref.spatial_event_names))

    def test_square_trial_rule_probability_matrix(self):
        s = Engine(config(execution_mode="reference", rng_mode="legacy")).state.simulator
        s.cellspace_tensor = torch.zeros((3, 4), dtype=torch.int8, device=DEVICE)
        s.rule_arrays_tensor = torch.zeros((3, 2, 3, 3), dtype=torch.int8, device=DEVICE)
        s.rule_ids = [0, 1, 2]
        s.rule_probs_tensor = torch.tensor([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]], device=DEVICE)
        s.spatial_event_arrays = s.spatial_event_arrays_tensor = None
        s.spatial_event_names = None
        s.rng_mode = "independent"
        s.execution_mode = MODE
        s.set_ParallelTrial(3)
        s.step(1., seed=19)
        self.assertFalse(s.TNHW_boolMask[0].any().item())
        self.assertTrue(s.TNHW_boolMask[1].all().item())
        self.assertFalse(s.TNHW_boolMask[2].any().item())

    def test_legacy_exact(self):
        for probability in (0., .37, 1.):
            c = config(rng_mode="legacy", global_prob=probability, steps=20)
            ref = Engine(replace(c, execution_mode="reference")).run().simulator
            new = Engine(c).run().simulator
            assert_sim(self, ref, new)

    def test_forced_overflow_has_no_truncation(self):
        c = config(candidate_capacity=1, steps=30)
        small = Engine(c).run().simulator
        large = Engine(replace(c, candidate_capacity=4096)).run().simulator
        assert_sim(self, small, large)

    def test_random_conflicts_and_boundaries(self):
        # Inject candidates directly to exercise dense collisions, negative
        # states, zero/unchanged writes and clipping, rather than rare matches.
        base = Engine(config(execution_mode="reference", rng_mode="legacy", trials=2)).state.simulator
        for seed in range(40):
            gen = torch.Generator().manual_seed(701+seed)
            ref = copy.deepcopy(base)
            ref.cellspace_tensor = torch.randint(-1, 3, (7, 9), generator=gen, dtype=torch.int8).to(DEVICE)
            ref.rule_arrays_tensor = torch.randint(-1, 3, (3, 2, 3, 3), generator=gen, dtype=torch.int8).to(DEVICE)
            ref.rule_ids = [0, 1, 2]
            ref.rule_probs_base_tensor = torch.ones(3, device=DEVICE)
            ref.rule_probs_tensor = torch.ones(3, device=DEVICE)
            ref.spatial_event_arrays = ref.spatial_event_arrays_tensor = None
            ref.spatial_event_names = None
            ref.set_ParallelTrial(2)
            new = copy.deepcopy(ref)
            from PyBCA.core.optimized import CandidatePlan
            new.candidate_plan = CandidatePlan(new, MODE, "legacy", candidate_capacity=1 if seed % 2 else 4096)
            mask = (torch.rand((2, 3, 7, 9), generator=gen) < .25).to(DEVICE)
            ref._match_centers_all_rules = lambda: mask.clone()
            ref.rng.manual_seed(seed)
            ref.update_cellspace(1.)
            new.TNHW_boolMask.copy_(mask)
            new.rng.manual_seed(seed)
            order = torch.randperm(3, generator=new.rng, device=DEVICE)
            if MODE == "cuda":
                plan = new.candidate_plan.cuda
                plan.order.copy_(order[None])
                plan.launch(1)
                plan.launch(2, independent=False)
                new._append_rule_counts(plan.rule_counts)
            else:
                for r in order.tolist():
                    new.candidate_plan._resolve(r)
                for r in range(3):
                    new._record_rule_history(r)
            assert_sim(self, ref, new)

    def test_random_matching_corners_and_negative_states(self):
        base = Engine(config(execution_mode="reference", rng_mode="legacy", trials=2)).state.simulator
        from PyBCA.core.optimized import CandidatePlan
        for seed in range(20):
            gen = torch.Generator().manual_seed(seed)
            base.cellspace_tensor = torch.randint(-1, 2, (8, 9), dtype=torch.int8, generator=gen).to(DEVICE)
            base.rule_arrays_tensor = torch.randint(-1, 2, (5, 2, 3, 3), dtype=torch.int8, generator=gen).to(DEVICE)
            base.rule_ids = list(range(5))
            base.set_ParallelTrial(2)
            plan = CandidatePlan(base, MODE)
            expected = base._match_centers_all_rules()
            if plan.cuda is None:
                actual = plan.match()
            else:
                plan.cuda.launch(0, independent=False)
                actual = base.TNHW_boolMask
            self.assertTrue(torch.equal(expected, actual))


class StreamingTests(unittest.TestCase):
    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA required")
    def test_independent_resume_between_cpu_and_gpu(self):
        for initial_device, initial_mode, resume_device, resume_mode in (
                (DEVICE, MODE, "cpu", "torch_sparse"), ("cpu", "torch_sparse", DEVICE, MODE)):
            with tempfile.TemporaryDirectory() as td:
                c = config(device=initial_device, execution_mode=initial_mode, stream_dir=td,
                           steps=7, flush_interval=3, checkpoint_interval=6, global_prob=.51)
                Engine(c).run()
                continued = replace(c, device=resume_device, execution_mode=resume_mode, steps=15)
                resumed = Engine(replace(continued, resume_from=str(Path(td)/"checkpoint.pt"))).run().simulator
                full = Engine(replace(continued, stream_dir=None)).run().simulator
                self.assertTrue(torch.equal(resumed.TCHW, full.TCHW))
                self.assertEqual(self.actual_records(td), self.expected_records(full, [0, 1, 2]))

    def test_crash_before_first_periodic_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as td:
            c = config(stream_dir=td, steps=5, flush_interval=2, checkpoint_interval=100)
            engine = Engine(c)
            original = engine.stepper
            def fail(step):
                if step == 3:
                    raise RuntimeError("simulated interruption")
                original(step)
            engine.stepper = fail
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                engine.run()
            checkpoint = Path(td)/"checkpoint.pt"
            self.assertEqual(torch.load(checkpoint, weights_only=True)["next_step"], 0)
            resumed = Engine(replace(c, resume_from=str(checkpoint))).run().simulator
            full = Engine(replace(c, stream_dir=None)).run().simulator
            self.assertTrue(torch.equal(resumed.TCHW, full.TCHW))
            self.assertEqual(self.actual_records(td), self.expected_records(full, [0, 1, 2]))

    def test_event_snapshot_window_and_negative_origin_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cells = root/"cells.yaml"
            cells.write_text("- coord: {x: -2, y: -3}\n  value: 1\n- coord: {x: -1, y: -3}\n  value: 1\n")
            events = root/"events.py"
            events.write_text("events = [\n"
                "('emit', (-2,-3), 1, (-1,-3), 2, 1.0, 1, 1),\n"
                "('same_step_observe', (-1,-3), 2, (-2,-3), 3, 1.0, 1, 1),\n"
                "('later_observe', (-1,-3), 2, (-2,-3), 4, 1.0, 2, 2)]\n")
            c = config(cellspace_path=str(cells), spatial_event_file_path=str(events),
                       global_prob=0., steps=4, record_rule_history=False)
            full = Engine(c).run().simulator
            self.assertEqual(full.event_history[0], {"emit": [1], "same_step_observe": [], "later_observe": [2]})
            run_dir = str(root/"run")
            Engine(replace(c, stream_dir=run_dir, steps=2, flush_interval=1, checkpoint_interval=1)).run()
            resumed = Engine(replace(c, stream_dir=run_dir, resume_from=str(root/"run/checkpoint.pt"),
                                     flush_interval=1, checkpoint_interval=1)).run().simulator
            self.assertEqual((resumed.offset_x, resumed.offset_y), (-2, -3))
            self.assertTrue(torch.equal(full.TCHW, resumed.TCHW))
            self.assertEqual(self.expected_records(full, [0,1,2]), self.actual_records(run_dir))

    def test_writer_lock_and_stop_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            c = config(stream_dir=td, steps=10, flush_interval=2, checkpoint_interval=4)
            engine = Engine(c)
            writer = HistoryWriter(c, engine.state.simulator)
            try:
                with self.assertRaisesRegex(RuntimeError, "Another process"):
                    HistoryWriter(c, engine.state.simulator)
            finally:
                writer.close()
            # A fresh output directory can checkpoint a graceful stop at step 0.
            c = replace(c, stream_dir=str(Path(td)/"stopped"))
            engine = Engine(c)
            engine.stop_requested = True
            result = engine.run()
            self.assertEqual(result.current_step, 0)
            saved = torch.load(Path(c.stream_dir)/"checkpoint.pt", weights_only=True)
            self.assertEqual(saved["next_step"], 0)

    def expected_records(self, sim, ids):
        rows = Counter()
        for kind, history in (("event", sim.event_history), ("rule", sim.rule_history)):
            for trial, events in enumerate(history or []):
                for name, steps in events.items():
                    for step in steps:
                        rows[kind, ids[trial], step, name] += 1
        return rows

    def actual_records(self, path):
        rows = Counter()
        for r in iter_history(path, verify=True):
            rows[r["kind"], r["trial"], r["step"], r["name"]] += r["count"]
        return rows

    def test_stream_resume_matches_uninterrupted_with_crash_tail(self):
        for rng in ("legacy", "independent"):
            c = config(rng_mode=rng, steps=23, trial_ids=[5, 8, 99], global_prob=.71)
            full = Engine(c).run().simulator
            with tempfile.TemporaryDirectory() as td:
                first_config = replace(c, steps=10, stream_dir=td, flush_interval=3, checkpoint_interval=6)
                first = Engine(first_config).run().simulator
                checkpoint = Path(td)/"checkpoint.pt"
                saved = checkpoint.read_bytes()
                # Commit a tail beyond the checkpoint, then simulate a crash by
                # restoring the older checkpoint. Resume must not duplicate it.
                Engine(replace(first_config, steps=16, resume_from=str(checkpoint))).run()
                checkpoint.write_bytes(saved)
                resumed = Engine(replace(first_config, steps=23, resume_from=str(checkpoint))).run().simulator
                self.assertTrue(torch.equal(full.TCHW, resumed.TCHW))
                self.assertEqual((full.offset_x, full.offset_y), (resumed.offset_x, resumed.offset_y))
                self.assertEqual(self.expected_records(full, [5, 8, 99]), self.actual_records(td))
                self.assertIsNone(first.event_history)
                self.assertIsNone(first.rule_history)

    def test_reject_changed_inputs_and_existing_run(self):
        with tempfile.TemporaryDirectory() as td:
            c = config(stream_dir=td, steps=3, flush_interval=2, checkpoint_interval=4)
            Engine(c).run()
            with self.assertRaises(FileExistsError):
                Engine(c).run()
            with self.assertRaisesRegex(ValueError, "do not match"):
                Engine(replace(c, steps=6, seed=999, resume_from=str(Path(td)/"checkpoint.pt"))).run()

    def test_corrupt_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            c = config(stream_dir=td, steps=3, flush_interval=2, checkpoint_interval=4)
            Engine(c).run()
            manifest = json.loads((Path(td)/"manifest.json").read_text())
            (Path(td)/manifest["chunks"][0]["path"]).write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                Engine(replace(c, steps=6, resume_from=str(Path(td)/"checkpoint.pt"))).run()


if __name__ == "__main__":
    unittest.main()
