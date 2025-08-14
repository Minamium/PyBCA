from cli_simClass import BCA_Simulator

if __name__ == "__main__":
    print("PyBCA CUDA Simulator Debug Mode")

    cellspace_path = "../SampleCP/test.yaml"
    rule_paths = [
        "rule/base-rule.yaml",
        "rule/C-Join_err-rule.yaml"
    ]
    
    simulator = BCA_Simulator(cellspace_path, rule_paths, device="cpu",
                              spatial_event_filePath="../SampleCP/BCA-IP_event.py")

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

    # PyTorchテンソルの表示
    print(simulator.cellspace_tensor)
    #print(simulator.rule_ids_tensor)
    print(simulator.rule_arrays_tensor)
    print(simulator.rule_probs_tensor)
    print(simulator.spatial_event_arrays_tensor)
    