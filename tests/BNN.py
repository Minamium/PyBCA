from PyBCA.cli_simClass import BCA_Simulator
from PyBCA import lib, load_state_conversions_from_yaml

cellspace_path = "PyBCA/Sample/Cellspace/BNN.yaml"
rule_paths = [
        "PyBCA/Sample/rule/base-rule.yaml",
        "PyBCA/Sample/rule/Join_fork.yaml"
        ]
    
simulator = BCA_Simulator(cellspace_path, rule_paths, device="cuda",
                              spatial_event_filePath="PyBCA/Sample/Specialevent/BNN_event.py",
                              use_tqdm=True,
                              trial_constant_sweep = {
                                  "join_err_0_input": {"base": 0.0,    "delta": 0.000005},
                                  "join_err_1_input": {"base": 0.0,    "delta": 0.000001},
                                  "fork_err_0_input": {"base": 0.0,    "delta": 0.000005},
                                 })

simulator.Allocate_torch_Tensors_on_Device()
simulator.set_ParallelTrial(2000)

steps = 80_000

simulator.run_steps(steps=steps, 
                    global_prob=0.5, 
                    seed=1, 
                    debug=False, 
                    debug_per_trial=False, 
                    state_gate_enable=True, 
                    state_gate_interval=500)

simulator.save_event_histry_for_dataframe("event_history.jsonl", 
                                           format="jsonl_trials", 
                                           deduplicate=True, 
                                           return_df=False)