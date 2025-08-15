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

    # 平行試行数より、4次元テンソルを作成する(T, 1, H, W)
    def set_ParallelTrial(self, parallel_trial: int):
        self.parallel_trial = parallel_trial

        # Trial x 1 x Height x Widthの4次元テンソルを作成
        self.TCHW = self.cellspace_tensor.unsqueeze(0).unsqueeze(1).expand(parallel_trial, 1, *self.cellspace_tensor.shape).clone()
        self.Pickup_rule = torch.zeros((2,3,3), dtype=torch.int8, device=self.device)
        self.TCHW_boolMask = torch.zeros((T,1,H,W), dtype=torch.bool, device=self.device)
        self.tmp_mask = torch.zeros((T,1,H,W), dtype=torch.int8, device=self.device)
        self.TCHW_applied = torch.zeros((T,1,H,W), dtype=torch.bool, device=self.device)

    # 任意ステップ数だけセル空間を更新する
    def run_steps(self, steps: int, global_prob: float):
        print(f"Run steps: {steps}")
        for i in range(steps):
            print(f"Step {i}")
            # デバッグのために受け取る配列を増やす
            self.TCHW, self.TCHW_boolMask, self.TCHW_applied = self.update_cellspace(
                global_prob=global_prob,
                seed=None,
            )

    ###################################
    # セル空間を更新する関数を定義する
    # 引数は全てtorchテンソルにして、cudaデバイス上で実行できるようにする
    ###################################
    def update_cellspace(self,
                         global_prob: float | None,            # グローバル確率
                         seed: int | None = None,
                        ) -> torch.Tensor:

        ####################################
        # 更新に使うテンソルの定義と整合性チェック  #
        ####################################

        # TCHW: [Trial, 1, H, W] dtype=int8, Trial別セル空間配列(引数, 戻り値)
        # 要求するデータ型と合致するかの整合性チェック
        assert self.TCHW.ndim == 4, "TCHW must be (T,1,H,W)"
        T, C, H, W = self.TCHW.shape

        # rule_arrays: [N,2,3,3] dtype=int8, N種類の遷移規則を記録した配列(引数)
        # 要求するデータ型と合致するかの整合性チェック
        assert self.rule_arrays.ndim == 4 and self.rule_arrays.shape[1:] == (2,3,3), "rule_arrays must be (N,2,3,3)"

        # rule_mask: [[0, 1, 0], [1, 1, 1], [0, 1, 0]] dtype=bool, 四近傍遷移規則マッチング用のマスク配列
        rule_mask = torch.tensor(
            [[0,1,0],
             [1,1,1],
             [0,1,0]], dtype=torch.bool, device=device
        )

        # rule_probs: [N] dtype=float32, N種類の遷移規則の確率配列(引数)
        # 要求するデータ型と合致するかの整合性チェック
        assert self.rule_probs.ndim == 1 and self.rule_probs.shape[0] == self.rule_arrays.shape[0], "rule_probs must be (N,)"

        # global_prob: float32, グローバル確率(引数)
        # 要求するデータ型と合致するかの整合性チェック
        assert global_prob is None or isinstance(global_prob, float), "global_prob must be float or None"

        # Pickup_rule: [2, 3, 3] dtype=int8, ループ内でシャッフル遷移規則から取り出す遷移規則
        # 要求するデータ型と合致するかの整合性チェック

        # THW_boolMask: [Trial, 1, H, W] dtype=bool, 取り出した遷移規則をマッチして適用できたセルの中心座標を1とするboolマスク
        # 要求するデータ型と合致するかの整合性チェック

        # tmp_mask: [Trial, 1, H, W] dtype=int8, 取り出した遷移規則により書き換えられる差分セルだけを検査するk_writeカーネルを元に書き換え予定のセルに1を足していくためのテンソル
        # 要求するデータ型と合致するかの整合性チェック

        # THW_applied: [Trial, 1, H, W] dtype=bool, 今までの遷移規則の適用により書き換えが行われたセルに1を立てておくboolマスク
        # 要求するデータ型と合致するかの整合性チェック

        ###########################
        # 更新関数内テンソルの初期化  #
        ###########################
    
        # Pickup_rule, THW_boolMask, tmp_mask, THW_appliedを初期化
    
        ###################
        # 乱数生成器の定義  #
        ###################
    

        ################################
        # 遷移規則のシャッフル(トライアル共有) #
        ################################
    
        # 遷移規則のシャッフル

        ########################################
        # シャッフル遷移規則配列の要素順にループを回す  #
        ########################################

        # loop begin
        # for ..




    
        # loop end
        
        # デバッグのために受け取る
        return self.TCHW

    # 特殊イベントを適用する
    def apply_spatial_events(self, 
                             state: torch.Tensor,
                             event_array: torch.Tensor
                            ) -> torch.Tensor:
        """
        """
        return state, do_event_list
###################################