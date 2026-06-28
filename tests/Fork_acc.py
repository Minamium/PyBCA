import argparse
from pathlib import Path

from PyBCA.cli_simClass import BCA_Simulator

ROOT = Path(__file__).resolve().parents[1]
FORK_RULE_HISTORY_IDS = (
    204, 205,
    210, 211,
    216, 217,
    222, 223,
)


def sample_path(*parts: str) -> str:
    return str((ROOT / "Sample" / Path(*parts)).resolve())


parser = argparse.ArgumentParser(description="Fork accuracy test")
parser.add_argument("--fork_ok_1", "--fork-ok-1", type=float, default=1.0)
parser.add_argument("--fork_err_0", type=float, default=1e-5)
parser.add_argument("--trials", type=int, default=100_000)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--global_prob", type=float, default=0.5)
parser.add_argument("--state_gate_interval", type=int, default=500)
parser.add_argument("--output_prefix", type=str, default="fork_acc")
parser.add_argument("--rule_history_prefix", type=str, default=None)
parser.add_argument(
    "--record_rule_history",
    "--record-rule-history",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--record_all_rules", "--record-all-rules", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

cellspace_paths = [
    ("P0", sample_path("Cellspace", "Fork_err", "P0_fork.yaml")),
    ("P1", sample_path("Cellspace", "Fork_err", "P1_fork.yaml")),
]

rule_paths = [
    sample_path("rule", "base-rule.yaml"),
    sample_path("rule", "Join_fork.yaml"),
]
event_path = sample_path("Specialevent", "Fork_detect.py")

for name, cellspace_path in cellspace_paths:
    simulator = BCA_Simulator(
        cellspace_path,
        rule_paths,
        device=args.device,
        spatial_event_filePath=event_path,
        use_tqdm=True,
        record_rule_history=args.record_rule_history,
        rule_history_rule_ids=None if args.record_all_rules else FORK_RULE_HISTORY_IDS,
        trial_constant_sweep={
            "fork_ok_1_input": {"base": args.fork_ok_1, "delta": 0.0},
            "fork_err_0_input": {"base": args.fork_err_0, "delta": 0.0},
        },
    )
    simulator.Allocate_torch_Tensors_on_Device()
    simulator.set_ParallelTrial(args.trials)
    simulator.run_steps(
        steps=args.steps,
        global_prob=args.global_prob,
        seed=1,
        debug=False,
        debug_per_trial=False,
        state_gate_enable=True,
        state_gate_interval=args.state_gate_interval,
    )

    output_file = f"{args.output_prefix}_{name}.jsonl"
    simulator.save_event_histry_for_dataframe(
        output_file,
        format="jsonl_trials",
        deduplicate=True,
        return_df=False,
    )

    if args.record_rule_history:
        rule_history_prefix = args.rule_history_prefix or f"{args.output_prefix}_rule"
        rule_output_file = f"{rule_history_prefix}_{name}.jsonl"
        simulator.save_rule_history_for_dataframe(
            rule_output_file,
            format="jsonl_trials",
            deduplicate=False,
            return_df=False,
        )

print("done")
