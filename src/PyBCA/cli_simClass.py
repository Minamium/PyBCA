try:
    # パッケージとして使用される場合
    from . import lib
except ImportError:
    # 直接実行される場合
    import lib

import torch
from typing import List

# クラス定義
class BCA_Simulator:
    def __init__(self,
                 cellspace_path: str,
                 rule_paths: List[str],
                 device: str = "cuda",
                 spatial_event_filePath: str | None = None
                 ):
        # セル空間読み込み
        self.cellspace_with_offset = lib.load_cell_space_yaml_to_numpy(cellspace_path)
        self.cellspace, self.offset_x, self.offset_y = lib.extract_cellspace_and_offset(self.cellspace_with_offset)

        # 遷移規則読み込み
        self.rule_ids = lib.get_rule_ids_from_files(rule_paths)
        self.rule_arrays, self.rule_probs = lib.load_multiple_transition_rules_with_probability(rule_paths)

        # 特殊イベント読み込み
        if spatial_event_filePath is not None:
            self.spatial_event_arrays = lib.load_special_events_from_file(spatial_event_filePath)
        else:
            self.spatial_event_arrays = None    

        # 計算デバイス設定
        self.device = device

    def Allocate_torch_Tensors_on_Device(self):
        # PyTorchテンソルにnumpy配列を転送
        self.cellspace_tensor = torch.from_numpy(self.cellspace).to(self.device)
        #self.rule_ids_tensor = torch.from_numpy(self.rule_ids).to(self.device)
        self.rule_arrays_tensor = torch.from_numpy(self.rule_arrays).to(self.device)
        self.rule_probs_tensor = torch.from_numpy(self.rule_probs).to(self.device)
        
        if self.spatial_event_arrays is not None:
            self.spatial_event_arrays_tensor = torch.from_numpy(self.spatial_event_arrays).to(self.device)
        else:
            self.spatial_event_arrays_tensor = None

        # デバイス情報を表示
        device_info = f"Device: {self.cellspace_tensor.device}"
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(self.cellspace_tensor.device)
            device_info += f" ({gpu_name})"
        
        print(f"Allocated torch tensors on {device_info}")
        print(f"Cellspace tensor shape: {self.cellspace_tensor.shape}")
        print(f"Rule arrays tensor shape: {self.rule_arrays_tensor.shape}")
        print(f"Rule probabilities tensor shape: {self.rule_probs_tensor.shape}")
        if self.spatial_event_arrays_tensor is not None:
            print(f"Spatial events tensor shape: {self.spatial_event_arrays_tensor.shape}")

    # 平行試行数より、三次元テンソルを作成する
    def set_ParallelTrial(self, parallel_trial: int):
        self.parallel_trial = parallel_trial

        # Trial x Height x Widthの三次元テンソルを作成
        self.THW_cellspace_tensor = self.cellspace_tensor.repeat(parallel_trial, 1, 1).contiguous()

    # 任意ステップ数だけセル空間を更新する
    def run_steps(self, steps: int, global_prob: float):
        print(f"Run steps: {steps}")
        for i in range(steps):
            print(f"Step {i}")
            # デバッグのために受け取る配列を増やす
            self.THW_cellspace_tensor, self.THW_boolMask, self.THW_applied = lib.update_cellspace(
                self.THW_cellspace_tensor,
                self.rule_arrays_tensor,
                self.rule_probs_tensor,
                global_prob=global_prob,
                seed=None,
                device=self.device
            )
            