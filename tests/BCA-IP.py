import argparse
from pathlib import Path

from PyBCA.cli_simClass import BCA_Simulator

ROOT = Path(__file__).resolve().parents[1]


def sample_path(*parts: str) -> str:
    return str((ROOT / "Sample" / Path(*parts)).resolve())


parser = argparse.ArgumentParser(description="BCA-IP test")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--steps", type=int, default=2_000_000)
parser.add_argument("--global_prob", type=float, default=1.0)
parser.add_argument("--output_prefix", type=str, default="BCA-IP")
args = parser.parse_args()

cellspace_path = sample_path("Cellspace", "BCA-IP.yaml")
rule_paths = [sample_path("rule", "base-rule.yaml")]
event_path = sample_path("Specialevent", "BCA-IP_event.py")

simulator = BCA_Simulator(
    cellspace_path,
    rule_paths,
    device=args.device,
    spatial_event_filePath=event_path,
    use_tqdm=True,
)

simulator.Allocate_torch_Tensors_on_Device()
simulator.set_ParallelTrial(args.trials)

simulator.run_steps(
    steps=args.steps,
    global_prob=args.global_prob,
    seed=1,
    debug=False,
    debug_per_trial=False,
    state_gate_enable=False,
)

simulator.save_event_histry_for_dataframe(
    f"{args.output_prefix}.jsonl",
    format="jsonl_trials",
    deduplicate=True,
    return_df=False,
)
