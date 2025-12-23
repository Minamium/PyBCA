import argparse
from PyBCA.cli_simClass import BCA_Simulator
from PyBCA import lib, load_state_conversions_from_yaml

parser = argparse.ArgumentParser(description="Join accuracy test")
parser.add_argument("--join_err_0", type=float, default=1e-6)
parser.add_argument("--join_err_1", type=float, default=1e-5)
parser.add_argument("--trials", type=int, default=100_000)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--global_prob", type=float, default=0.5)
parser.add_argument("--state_gate_interval", type=int, default=500)
parser.add_argument("--output_prefix", type=str, default="join_acc")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

cellspace_paths = [
    ("P0", "PyBCA/Sample/Cellspace/Join_err/P0_join.yaml"),
    ("P1", "PyBCA/Sample/Cellspace/Join_err/P1_join.yaml"),
    ("P2", "PyBCA/Sample/Cellspace/Join_err/P2_join.yaml"),
]

rule_paths = [
    "PyBCA/Sample/rule/base-rule.yaml",
    "PyBCA/Sample/rule/Join_fork.yaml"
]

for name, cellspace_path in cellspace_paths:
    simulator = BCA_Simulator(
        cellspace_path, rule_paths, device=f"{args.device}",
        spatial_event_filePath="PyBCA/Sample/Specialevent/Join_detect.py",
        use_tqdm=True,
        trial_constant_sweep={
            "join_err_0_input": {"base": args.join_err_0, "delta": 0.0},
            "join_err_1_input": {"base": args.join_err_1, "delta": 0.0},
        }
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
        state_gate_interval=args.state_gate_interval
    )
    
    output_file = f"{args.output_prefix}_{name}.jsonl"
    simulator.save_event_histry_for_dataframe(
        output_file,
        format="jsonl_trials",
        deduplicate=True,
        return_df=False
    )

print("done")