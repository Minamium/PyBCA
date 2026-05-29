import argparse
from pathlib import Path

from PyBCA.cli_simClass import BCA_Simulator

ROOT = Path(__file__).resolve().parents[1]


def sample_path(*parts: str) -> str:
    return str((ROOT / "Sample" / Path(*parts)).resolve())


EVENT_PRESETS = {
    1: "Hebb learning input schedule: pre input at 0-20k steps, post input at 0-40k steps.",
    2: "Noise baseline: no external pre/post neuron input.",
}

parser = argparse.ArgumentParser(description="BNN test")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--trials", type=int, default=2000)
parser.add_argument("--steps", type=int, default=80_000)
parser.add_argument("--global_prob", type=float, default=0.5)
parser.add_argument("--state_gate_interval", type=int, default=500)
parser.add_argument("--output_prefix", type=str, default="bnn")
parser.add_argument(
    "--event_num",
    type=int,
    choices=EVENT_PRESETS,
    default=1,
    help=(
        "Select Sample/Specialevent/BNN_event_N.py. "
        "1: Hebb learning schedule, 2: no-input noise baseline."
    ),
)
args = parser.parse_args()

cellspace_path = sample_path("Cellspace", "BNN.yaml")
rule_paths = [
    sample_path("rule", "base-rule.yaml"),
    sample_path("rule", "Join_fork.yaml"),
]

event_path = sample_path("Specialevent", f"BNN_event_{args.event_num}.py")

simulator = BCA_Simulator(
    cellspace_path,
    rule_paths,
    device=args.device,
    spatial_event_filePath=event_path,
    use_tqdm=True,
    trial_constant_sweep={
        "join_err_0_input": {"base": 0.0, "delta": 0.000005},
        "join_err_1_input": {"base": 0.0, "delta": 0.000001},
        "fork_err_0_input": {"base": 0.0, "delta": 0.000005},
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

simulator.save_event_histry_for_dataframe(
    f"{args.output_prefix}.jsonl",
    format="jsonl_trials",
    deduplicate=True,
    return_df=False,
)
