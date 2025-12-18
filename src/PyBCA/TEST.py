# PyBCAはeditable installされているので絶対インポートを使用
from PyBCA.cli_simClass import BCA_Simulator
from PyBCA import lib, load_state_conversions_from_yaml

import torch, sys, time
torch.set_printoptions(
    threshold=sys.maxsize,   # 省略せず全部出す
    linewidth=200,           # 1行の最大幅
    sci_mode=False,          # 指数表記をやめる（必要なら）
    precision=10             # 小数表示桁数（浮動小数なら）
)

if __name__ == "__main__":
    print("PyBCA CUDA Simulator Debug Mode")
    print("="*60)
    
    # === BCA_Simulator初期化テスト ===
    print("[TEST] BCA_Simulator initialization")
    print("-"*60)

    cellspace_path = "Sample/Cellspace/BNN.yaml"
    rule_paths = [
        "Sample/rule/base-rule.yaml",
        "Sample/rule/Join_fork.yaml"
    ]
    
    simulator = BCA_Simulator(cellspace_path, rule_paths, device="cpu",
                              spatial_event_filePath="Sample/Specialevent/BNN_event.py",
                              use_tqdm=True,
                              trial_constant_sweep = {
                                  "join_err_0_input": {"base": 0.0,    "delta": 0.000001},
                                  "join_err_1_input": {"base": 0.0,    "delta": 0.00001},
                                  "fork_err_0_input": {"base": 0.0,    "delta": 0.000001},
                              }
                              )

    import numpy as np
    # np.set_printoptions(threshold=np.inf, linewidth=10**9)  # 全要素表示

    # セル空間の表示
    # print(simulator.cellspace)

    # オフセット情報の表示
    # print(simulator.offset_x, simulator.offset_y)

    # 遷移規則の表示
    # print(simulator.rule_ids)
    # print(simulator.rule_arrays)
    # print(simulator.rule_probs)
    
    simulator.Allocate_torch_Tensors_on_Device()

    #simulator.rule_probs_tensor[0] = 0.1

    # PyTorchテンソルの表示
    #print(simulator.cellspace_tensor)
    print(simulator.rule_arrays_tensor)
    print(simulator.const_rule_ids)
    #print(simulator.rule_probs_tensor)
    #print(simulator.spatial_event_arrays_tensor)
    #print(simulator.state_conversions_tensor)

    simulator.set_ParallelTrial(1)
    #print(simulator.TCHW)
    print(simulator.rule_probs_tensor)
    timer = time.time()
    simulator.run_steps(steps=80_000, global_prob=0.5, seed=1, debug=False, debug_per_trial=False, state_gate_enable=True, state_gate_interval=300)
    print(f"Time: {time.time() - timer}")

    #print("After Apllied run_steps, TCHW")
    #print(simulator.TCHW)

    #print("After Apllied run_steps, TNHW_boolMask")
    #print(simulator.TNHW_boolMask[0,:,:,:])
    #print(simulator.TNHW_boolMask[1,:,:,:])

    #print("After Apllied run_steps, TCHW_applied")
    #print(simulator.TCHW_applied)

    #simulator.debug()
    #print(simulator.TCHW)
    simulator.save_final_state(0, "tested1.yaml")
    #simulator.save_final_state(1, "tested2.yaml")
    #simulator.save_final_state(2, "tested3.yaml")

    simulator.save_event_histry_for_dataframe("event_history.jsonl", format="jsonl_trials", deduplicate=True, return_df=False)
    
    