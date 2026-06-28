"""Detailed parity checks across many Sample configurations.

Run:
    PYTHONPATH=src python tests/simulator_parity.py
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import tempfile
import time

import torch

from PyBCA._legacy.cli_simClass import BCA_Simulator as LegacyBCA
from PyBCA.api import Config, Engine
from PyBCA.core.simulator import BCA_Simulator as CoreBCA

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Case:
    name: str
    cellspace_path: str
    rule_paths: tuple[str, ...]
    steps: int
    global_prob: float
    seed: int
    trials: int
    event_path: str | None = None
    state_gate_enable: bool = False
    state_gate_interval: int = 500
    trial_constant_sweep: dict[str, dict[str, float]] | None = None
    run_engine_check: bool = False
    run_steps_check: bool = True
    check_event_export: bool = False


def _resolve(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((ROOT / p).resolve())


def _assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    if not torch.equal(a, b):
        diff = (a != b).sum().item()
        raise AssertionError(f"{name} mismatch (different elements: {diff}).")


def _assert_dict_equal(a: dict, b: dict, name: str) -> None:
    if a != b:
        raise AssertionError(f"{name} mismatch.")


def _assert_event_history_equal(a, b, name: str) -> None:
    if a is None and b is None:
        return
    if (a is None) != (b is None):
        raise AssertionError(f"{name} mismatch (one side is None).")
    if a != b:
        raise AssertionError(f"{name} mismatch.")


def _assert_frame_records_equal(df_a, df_b, name: str) -> None:
    cols_a = list(df_a.columns)
    cols_b = list(df_b.columns)
    if cols_a != cols_b:
        raise AssertionError(f"{name} columns mismatch: {cols_a} != {cols_b}")
    rec_a = df_a.to_dict(orient="records")
    rec_b = df_b.to_dict(orient="records")
    if rec_a != rec_b:
        raise AssertionError(f"{name} records mismatch.")


def _load_jsonl_without_meta(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if isinstance(obj, dict) and "__meta__" in obj:
                continue
            rows.append(obj)
    return rows


def _new_sim(case: Case):
    kwargs = dict(
        cellspace_path=_resolve(case.cellspace_path),
        rule_paths=[_resolve(p) for p in case.rule_paths],
        device="cpu",
        spatial_event_filePath=_resolve(case.event_path),
        gui_mode=False,
        use_tqdm=False,
        trial_constant_sweep=copy.deepcopy(case.trial_constant_sweep),
    )
    legacy = LegacyBCA(**kwargs)
    core = CoreBCA(**kwargs)
    return legacy, core


def _check_initial_parity(case: Case, legacy, core) -> None:
    _assert_tensor_equal(legacy.cellspace_tensor, core.cellspace_tensor, f"{case.name}: cellspace")
    _assert_tensor_equal(
        legacy.rule_arrays_tensor, core.rule_arrays_tensor, f"{case.name}: rule_arrays"
    )
    _assert_tensor_equal(
        legacy.rule_probs_tensor, core.rule_probs_tensor, f"{case.name}: rule_probs_tensor"
    )
    _assert_tensor_equal(legacy.TCHW, core.TCHW, f"{case.name}: initial TCHW")

    _assert_dict_equal(
        legacy.rule_id_to_index,
        core.rule_id_to_index,
        f"{case.name}: rule_id_to_index",
    )
    _assert_dict_equal(
        legacy.const_rule_indices,
        core.const_rule_indices,
        f"{case.name}: const_rule_indices",
    )

    legacy_event = getattr(legacy, "spatial_event_arrays_tensor", None)
    core_event = getattr(core, "spatial_event_arrays_tensor", None)
    if legacy_event is None and core_event is None:
        pass
    elif (legacy_event is None) != (core_event is None):
        raise AssertionError(f"{case.name}: spatial_event_arrays_tensor None mismatch")
    else:
        _assert_tensor_equal(legacy_event, core_event, f"{case.name}: spatial_event_arrays_tensor")

    legacy_conv = getattr(legacy, "state_conversions_tensor", None)
    core_conv = getattr(core, "state_conversions_tensor", None)
    if legacy_conv is None and core_conv is None:
        pass
    elif (legacy_conv is None) != (core_conv is None):
        raise AssertionError(f"{case.name}: state_conversions_tensor None mismatch")
    else:
        _assert_tensor_equal(legacy_conv, core_conv, f"{case.name}: state_conversions_tensor")


def _check_step_parity(case: Case, legacy, core, step_idx: int) -> None:
    _assert_tensor_equal(legacy.TCHW, core.TCHW, f"{case.name}: step={step_idx} TCHW")
    _assert_tensor_equal(
        legacy.TCHW_applied, core.TCHW_applied, f"{case.name}: step={step_idx} TCHW_applied"
    )
    _assert_tensor_equal(
        legacy.TNHW_boolMask, core.TNHW_boolMask, f"{case.name}: step={step_idx} TNHW_boolMask"
    )
    _assert_event_history_equal(
        legacy.event_history,
        core.event_history,
        f"{case.name}: step={step_idx} event_history",
    )
    if legacy._current_step != core._current_step:
        raise AssertionError(f"{case.name}: step={step_idx} current step mismatch.")


def _check_export_parity(case: Case, legacy, core) -> None:
    if legacy.event_history is None and core.event_history is None:
        return

    with tempfile.TemporaryDirectory(prefix=f"bca_parity_{case.name}_") as td:
        for fmt in ("jsonl_trials", "jsonl_trials_dict", "jsonl"):
            legacy_out = str(Path(td) / f"{case.name}_legacy_{fmt}.out")
            core_out = str(Path(td) / f"{case.name}_core_{fmt}.out")

            legacy_df = legacy.save_event_histry_for_dataframe(
                path=legacy_out,
                format=fmt,
                deduplicate=True,
                return_df=True,
                save_meta=True,
            )
            core_df = core.save_event_histry_for_dataframe(
                path=core_out,
                format=fmt,
                deduplicate=True,
                return_df=True,
                save_meta=True,
            )

            _assert_frame_records_equal(legacy_df, core_df, f"{case.name}: export df ({fmt})")

            legacy_rows = _load_jsonl_without_meta(legacy_out)
            core_rows = _load_jsonl_without_meta(core_out)
            if legacy_rows != core_rows:
                raise AssertionError(f"{case.name}: export file payload mismatch ({fmt}).")


def _run_stepwise_parity(case: Case) -> tuple[float, object]:
    start = time.perf_counter()

    legacy, core = _new_sim(case)
    legacy.Allocate_torch_Tensors_on_Device()
    core.Allocate_torch_Tensors_on_Device()
    legacy.set_ParallelTrial(case.trials)
    core.set_ParallelTrial(case.trials)

    _check_initial_parity(case, legacy, core)

    for i in range(case.steps):
        legacy.step(
            global_prob=case.global_prob,
            seed=case.seed,
            debug=False,
            debug_per_trial=False,
            state_gate_enable=case.state_gate_enable,
            state_gate_interval=case.state_gate_interval,
        )
        core.step(
            global_prob=case.global_prob,
            seed=case.seed,
            debug=False,
            debug_per_trial=False,
            state_gate_enable=case.state_gate_enable,
            state_gate_interval=case.state_gate_interval,
        )
        _check_step_parity(case, legacy, core, i)

    if case.check_event_export:
        _check_export_parity(case, legacy, core)

    elapsed = time.perf_counter() - start
    return elapsed, core


def _run_steps_api_parity(case: Case) -> float:
    start = time.perf_counter()

    legacy, core = _new_sim(case)
    legacy.Allocate_torch_Tensors_on_Device()
    core.Allocate_torch_Tensors_on_Device()
    legacy.set_ParallelTrial(case.trials)
    core.set_ParallelTrial(case.trials)
    _check_initial_parity(case, legacy, core)

    legacy.run_steps(
        steps=case.steps,
        global_prob=case.global_prob,
        seed=case.seed,
        debug=False,
        debug_per_trial=False,
        state_gate_enable=case.state_gate_enable,
        state_gate_interval=case.state_gate_interval,
    )
    core.run_steps(
        steps=case.steps,
        global_prob=case.global_prob,
        seed=case.seed,
        debug=False,
        debug_per_trial=False,
        state_gate_enable=case.state_gate_enable,
        state_gate_interval=case.state_gate_interval,
    )

    _assert_tensor_equal(legacy.TCHW, core.TCHW, f"{case.name}: run_steps TCHW")
    _assert_tensor_equal(
        legacy.TCHW_applied, core.TCHW_applied, f"{case.name}: run_steps TCHW_applied"
    )
    _assert_tensor_equal(
        legacy.TNHW_boolMask, core.TNHW_boolMask, f"{case.name}: run_steps TNHW_boolMask"
    )
    _assert_event_history_equal(
        legacy.event_history,
        core.event_history,
        f"{case.name}: run_steps event_history",
    )
    if legacy._current_step != core._current_step:
        raise AssertionError(f"{case.name}: run_steps current step mismatch.")

    return time.perf_counter() - start


def _run_engine_check(case: Case, expected_core) -> float:
    start = time.perf_counter()

    cfg = Config(
        cellspace_path=_resolve(case.cellspace_path),
        rule_paths=tuple(_resolve(p) for p in case.rule_paths),
        device="cpu",
        trials=case.trials,
        steps=case.steps,
        global_prob=case.global_prob,
        seed=case.seed,
        spatial_event_file_path=_resolve(case.event_path),
        use_tqdm="false",
        trial_constant_sweep=copy.deepcopy(case.trial_constant_sweep),
        state_gate_enable=case.state_gate_enable,
        state_gate_interval=case.state_gate_interval,
    )

    result = Engine(cfg).run()
    eng_sim = result.simulator
    _assert_tensor_equal(expected_core.TCHW, eng_sim.TCHW, f"{case.name}: engine TCHW")
    _assert_event_history_equal(
        expected_core.event_history,
        eng_sim.event_history,
        f"{case.name}: engine event_history",
    )

    return time.perf_counter() - start


def _run_case(case: Case) -> None:
    step_elapsed, core = _run_stepwise_parity(case)

    run_steps_elapsed = 0.0
    if case.run_steps_check:
        run_steps_elapsed = _run_steps_api_parity(case)

    engine_elapsed = 0.0
    if case.run_engine_check:
        engine_elapsed = _run_engine_check(case, core)

    print(
        f"[OK] {case.name} "
        f"(step={step_elapsed:.3f}s run_steps={run_steps_elapsed:.3f}s engine={engine_elapsed:.3f}s)"
    )


def _cases() -> list[Case]:
    return [
        Case(
            name="cjoin_base_prob05",
            cellspace_path="Sample/Cellspace/C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml",),
            steps=20,
            global_prob=0.5,
            seed=7,
            trials=3,
            run_engine_check=True,
        ),
        Case(
            name="cjoin_base_prob1",
            cellspace_path="Sample/Cellspace/C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml",),
            steps=10,
            global_prob=1.0,
            seed=7,
            trials=2,
        ),
        Case(
            name="test_base_prob0",
            cellspace_path="Sample/Cellspace/test.yaml",
            rule_paths=("Sample/rule/base-rule.yaml",),
            steps=8,
            global_prob=0.0,
            seed=13,
            trials=2,
        ),
        Case(
            name="test_with_event_window",
            cellspace_path="Sample/Cellspace/test.yaml",
            rule_paths=("Sample/rule/base-rule.yaml",),
            event_path="Sample/Specialevent/test_event.py",
            steps=110,
            global_prob=0.5,
            seed=13,
            trials=2,
            run_engine_check=True,
            check_event_export=True,
        ),
        Case(
            name="jf_cjoin_base_joinfork",
            cellspace_path="Sample/Cellspace/JF-C-join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            steps=12,
            global_prob=0.5,
            seed=5,
            trials=2,
        ),
        Case(
            name="two_way_cjoin_base_joinfork",
            cellspace_path="Sample/Cellspace/2-way_C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            steps=8,
            global_prob=0.5,
            seed=3,
            trials=1,
        ),
        Case(
            name="bnn_event1_stategate_sweep",
            cellspace_path="Sample/Cellspace/BNN.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            event_path="Sample/Specialevent/BNN_event_1.py",
            steps=15,
            global_prob=0.5,
            seed=11,
            trials=4,
            state_gate_enable=True,
            state_gate_interval=5,
            trial_constant_sweep={
                "join_err_0_input": {"base": 0.0, "delta": 0.001},
                "join_err_1_input": {"base": 0.0, "delta": 0.0005},
                "fork_err_0_input": {"base": 0.0, "delta": 0.001},
            },
            run_engine_check=True,
            check_event_export=True,
        ),
        Case(
            name="bnn_event2_stategate_sweep",
            cellspace_path="Sample/Cellspace/BNN.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            event_path="Sample/Specialevent/BNN_event_2.py",
            steps=10,
            global_prob=0.5,
            seed=19,
            trials=4,
            state_gate_enable=True,
            state_gate_interval=3,
            trial_constant_sweep={
                "join_err_0_input": {"base": 0.0, "delta": 0.001},
                "join_err_1_input": {"base": 0.0, "delta": 0.0005},
                "fork_err_0_input": {"base": 0.0, "delta": 0.001},
            },
        ),
        Case(
            name="join_err_p0",
            cellspace_path="Sample/Cellspace/Join_err/P0_join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            event_path="Sample/Specialevent/Join_detect.py",
            steps=20,
            global_prob=0.5,
            seed=17,
            trials=5,
            state_gate_enable=True,
            state_gate_interval=4,
            trial_constant_sweep={
                "join_err_0_input": {"base": 1e-6, "delta": 0.0},
                "join_err_1_input": {"base": 1e-5, "delta": 0.0},
            },
            check_event_export=True,
        ),
        Case(
            name="join_err_p1",
            cellspace_path="Sample/Cellspace/Join_err/P1_join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            event_path="Sample/Specialevent/Join_detect.py",
            steps=20,
            global_prob=0.5,
            seed=17,
            trials=5,
            state_gate_enable=True,
            state_gate_interval=4,
            trial_constant_sweep={
                "join_err_0_input": {"base": 1e-6, "delta": 0.0},
                "join_err_1_input": {"base": 1e-5, "delta": 0.0},
            },
        ),
        Case(
            name="join_err_p2",
            cellspace_path="Sample/Cellspace/Join_err/P2_join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            event_path="Sample/Specialevent/Join_detect.py",
            steps=20,
            global_prob=0.5,
            seed=17,
            trials=5,
            state_gate_enable=True,
            state_gate_interval=4,
            trial_constant_sweep={
                "join_err_0_input": {"base": 1e-6, "delta": 0.0},
                "join_err_1_input": {"base": 1e-5, "delta": 0.0},
            },
        ),
        Case(
            name="fork_err_p0",
            cellspace_path="Sample/Cellspace/Fork_err/P0_fork.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            event_path="Sample/Specialevent/Fork_detect.py",
            steps=20,
            global_prob=0.5,
            seed=23,
            trials=5,
            state_gate_enable=True,
            state_gate_interval=4,
            trial_constant_sweep={
                "fork_err_0_input": {"base": 1e-5, "delta": 0.0},
            },
        ),
        Case(
            name="fork_err_p1",
            cellspace_path="Sample/Cellspace/Fork_err/P1_fork.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
            event_path="Sample/Specialevent/Fork_detect.py",
            steps=20,
            global_prob=0.5,
            seed=23,
            trials=5,
            state_gate_enable=True,
            state_gate_interval=4,
            trial_constant_sweep={
                "fork_err_0_input": {"base": 1e-5, "delta": 0.0},
            },
        ),
        Case(
            name="cjoin_err_rule3_p0",
            cellspace_path="Sample/Cellspace/C-join_err/P0_C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/C-Join_err-rule_3.yaml"),
            event_path="Sample/Specialevent/C-Joinerrdetect.py",
            steps=12,
            global_prob=0.5,
            seed=29,
            trials=4,
            trial_constant_sweep={
                "join_err_0_input": {"base": 0.001, "delta": 0.0},
                "join_err_1_input": {"base": 0.002, "delta": 0.0},
            },
        ),
        Case(
            name="cjoin_err_rule3_p1",
            cellspace_path="Sample/Cellspace/C-join_err/P1_C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/C-Join_err-rule_3.yaml"),
            event_path="Sample/Specialevent/C-Joinerrdetect.py",
            steps=12,
            global_prob=0.5,
            seed=29,
            trials=4,
            trial_constant_sweep={
                "join_err_0_input": {"base": 0.001, "delta": 0.0},
                "join_err_1_input": {"base": 0.002, "delta": 0.0},
            },
        ),
        Case(
            name="cjoin_err_rule3_p2",
            cellspace_path="Sample/Cellspace/C-join_err/P2_C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/C-Join_err-rule_3.yaml"),
            event_path="Sample/Specialevent/C-Joinerrdetect.py",
            steps=12,
            global_prob=0.5,
            seed=29,
            trials=4,
            trial_constant_sweep={
                "join_err_0_input": {"base": 0.001, "delta": 0.0},
                "join_err_1_input": {"base": 0.002, "delta": 0.0},
            },
        ),
        Case(
            name="cjoin_err_plain_rule_p0",
            cellspace_path="Sample/Cellspace/C-join_err/P0_C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/C-Join_err-rule.yaml"),
            event_path="Sample/Specialevent/C-Joinerrdetect.py",
            steps=8,
            global_prob=0.5,
            seed=41,
            trials=3,
        ),
        Case(
            name="cjoin_err_plain_rule_p1",
            cellspace_path="Sample/Cellspace/C-join_err/P1_C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/C-Join_err-rule.yaml"),
            event_path="Sample/Specialevent/C-Joinerrdetect.py",
            steps=8,
            global_prob=0.5,
            seed=41,
            trials=3,
        ),
        Case(
            name="cjoin_err_plain_rule_p2",
            cellspace_path="Sample/Cellspace/C-join_err/P2_C-Join.yaml",
            rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/C-Join_err-rule.yaml"),
            event_path="Sample/Specialevent/C-Joinerrdetect.py",
            steps=8,
            global_prob=0.5,
            seed=41,
            trials=3,
        ),
        Case(
            name="bca_ip_large_grid",
            cellspace_path="Sample/Cellspace/BCA-IP.yaml",
            rule_paths=("Sample/rule/base-rule.yaml",),
            event_path="Sample/Specialevent/BCA-IP_event.py",
            steps=1,
            global_prob=1.0,
            seed=31,
            trials=1,
            run_engine_check=True,
            run_steps_check=False,
        ),
    ]


def _run_rule_history_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="bca_rule_history_") as td:
        out = Path(td) / "rule_history.jsonl"
        cfg = Config(
            cellspace_path=_resolve("Sample/Cellspace/Join_err/P2_join.yaml"),
            rule_paths=(
                _resolve("Sample/rule/base-rule.yaml"),
                _resolve("Sample/rule/Join_fork.yaml"),
            ),
            device="cpu",
            trials=3,
            steps=5,
            global_prob=1.0,
            seed=1,
            use_tqdm="false",
            rule_history_rule_ids=(200, 201, 202, 203),
            rule_history_path=str(out),
            rule_history_format="jsonl_trials",
            rule_history_return_df=True,
            state_gate_enable=True,
        )

        result = Engine(cfg).run()
        if not out.exists():
            raise AssertionError(f"rule history output was not created: {out}")
        if result.rule_history is None or result.rule_history.empty:
            raise AssertionError("rule history dataframe is empty.")

        event_names = set(result.rule_history["event"])
        allowed = {"rule_200", "rule_201", "rule_202", "rule_203"}
        if not event_names.issubset(allowed):
            raise AssertionError(f"unexpected rule history events: {sorted(event_names)}")
        if "rule_200" not in event_names:
            raise AssertionError("expected rule_200 to fire in P2 join smoke case.")

        lines = out.read_text(encoding="utf-8").splitlines()
        if len(lines) != 4:
            raise AssertionError(f"unexpected jsonl line count: {len(lines)}")

        meta = json.loads(lines[0])["__meta__"]
        if meta["history_kind"] != "rule_history":
            raise AssertionError("rule history metadata kind mismatch.")
        if meta["record_rule_history"] is not True:
            raise AssertionError("rule_history_path should enable record_rule_history.")
        if meta["rule_history_rule_ids"] != [200, 201, 202, 203]:
            raise AssertionError("tracked rule ids were not persisted correctly.")

        first_trial = json.loads(lines[1])
        events = {name: steps for name, steps in first_trial["events"]}
        if len(events.get("rule_200", [])) == 0:
            raise AssertionError("rule_200 steps were not written for trial 0.")

    print("[OK] rule_history_smoke")


def main() -> None:
    torch.manual_seed(0)
    start_all = time.perf_counter()

    for case in _cases():
        _run_case(case)

    _run_rule_history_smoke()

    elapsed = time.perf_counter() - start_all
    print(f"All parity checks passed. total={elapsed:.3f}s")


if __name__ == "__main__":
    main()
