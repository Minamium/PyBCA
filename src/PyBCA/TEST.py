from .cli_simClass import BCA_Simulator

try:
    # パッケージとして使用される場合
    from . import lib
except ImportError:
    # 直接実行される場合
    import lib

import torch, sys
torch.set_printoptions(
    threshold=sys.maxsize,   # 省略せず全部出す
    linewidth=200,           # 1行の最大幅
    sci_mode=False,          # 指数表記をやめる（必要なら）
    precision=10             # 小数表示桁数（浮動小数なら）
)

if __name__ == "__main__":
    print("PyBCA CUDA Simulator Debug Mode")

    cellspace_path = "SampleCP/test.yaml"
    rule_paths = [
        "src/PyBCA/rule/base-rule.yaml",
        "src/PyBCA/rule/C-Join_err-rule.yaml"
    ]
    
    simulator = BCA_Simulator(cellspace_path, rule_paths, device="cpu",
                              spatial_event_filePath="SampleCP/test_event.py")

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
    print(simulator.cellspace_tensor)
    #print(simulator.rule_arrays_tensor)
    #print(simulator.rule_probs_tensor)
    print(simulator.spatial_event_arrays_tensor)

    simulator.set_ParallelTrial(3)
    #print(simulator.TCHW)
    simulator.run_steps(100, global_prob=0.5, seed=1, debug=False, debug_per_trial=False)

    #print("After Apllied run_steps, TCHW")
    #print(simulator.TCHW)

    #print("After Apllied run_steps, TNHW_boolMask")
    #print(simulator.TNHW_boolMask[0,:,:,:])
    #print(simulator.TNHW_boolMask[1,:,:,:])

    #print("After Apllied run_steps, TCHW_applied")
    #print(simulator.TCHW_applied)

    #simulator.debug()
    #print(simulator.TCHW)
    #simulator.save_final_state(0, "tested1.yaml")
    #simulator.save_final_state(1, "tested2.yaml")
    #simulator.save_final_state(2, "tested3.yaml")

    simulator.save_event_histry_for_dataframe("event_history.jsonl", format="jsonl_trials", deduplicate=True, return_df=False)
    
    