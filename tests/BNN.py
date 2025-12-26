import argparse
from PyBCA.cli_simClass import BCA_Simulator
from PyBCA import lib, load_state_conversions_from_yaml

parser = argparse.ArgumentParser(description="BNN test")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--trials", type=int, default=2000)
parser.add_argument("--steps", type=int, default=80_000)
parser.add_argument("--global_prob", type=float, default=0.5)
parser.add_argument("--state_gate_interval", type=int, default=500)
parser.add_argument("--output_prefix", type=str, default="bnn")
args = parser.parse_args()

cellspace_path = "PyBCA/Sample/Cellspace/BNN.yaml"
rule_paths = [
        "PyBCA/Sample/rule/base-rule.yaml",
        "PyBCA/Sample/rule/Join_fork.yaml"
        ]
    
simulator = BCA_Simulator(cellspace_path, rule_paths, device=args.device,
                              spatial_event_filePath="PyBCA/Sample/Specialevent/BNN_event.py",
                              use_tqdm=True,
                              trial_constant_sweep = {
                                  "join_err_0_input": {"base": 0.0,    "delta": 0.000005},
                                  "join_err_1_input": {"base": 0.0,    "delta": 0.000001},
                                  "fork_err_0_input": {"base": 0.0,    "delta": 0.000005},
                                 })

simulator.Allocate_torch_Tensors_on_Device()
simulator.set_ParallelTrial(args.trials)

steps = args.steps

simulator.run_steps(steps=steps, 
                    global_prob=args.global_prob, 
                    seed=1, 
                    debug=False, 
                    debug_per_trial=False, 
                    state_gate_enable=True, 
                    state_gate_interval=args.state_gate_interval)

simulator.save_event_histry_for_dataframe(f"{args.output_prefix}.jsonl", 
                                           format="jsonl_trials", 
                                           deduplicate=True, 
                                           return_df=False)