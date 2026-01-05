import argparse
from PyBCA.cli_simClass import BCA_Simulator
from PyBCA import lib, load_state_conversions_from_yaml

parser = argparse.ArgumentParser(description="BCA-IP test")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--steps", type=int, default=2_000_000)
parser.add_argument("--global_prob", type=float, default=1.0)
parser.add_argument("--output_prefix", type=str, default="BCA-IP")
args = parser.parse_args()

cellspace_path = "PyBCA/Sample/Cellspace/BCA-IP.yaml"
rule_paths = ["PyBCA/Sample/rule/base-rule.yaml"]
    
simulator = BCA_Simulator(cellspace_path, rule_paths, device=args.device,
                              spatial_event_filePath=f"PyBCA/Sample/Specialevent/BCA-IP_event.py",
                              use_tqdm=True)

simulator.Allocate_torch_Tensors_on_Device()
simulator.set_ParallelTrial(args.trials)

steps = args.steps

simulator.run_steps(steps=steps, 
                    global_prob=args.global_prob, 
                    seed=1, 
                    debug=False, 
                    debug_per_trial=False, 
                    state_gate_enable=False)

simulator.save_event_histry_for_dataframe(f"{args.output_prefix}.jsonl", 
                                           format="jsonl_trials", 
                                           deduplicate=True, 
                                           return_df=False)