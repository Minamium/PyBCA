try:
    # パッケージとして使用される場合
    from . import lib
except ImportError:
    # 直接実行される場合
    import lib

import torch
import torch.nn.functional as F 
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

        # 乱数生成器
        self.rng = torch.Generator(device=self.device)

        # 畳み込み有効化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        

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
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
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
        self.TCHW = self.cellspace_tensor.repeat(parallel_trial, 1, 1).unsqueeze(1).contiguous()
        T, C, H, W = self.TCHW.shape

        # 更新関数でのバッファテンソル群の作成
        self.Pickup_rule = torch.zeros((2,3,3), dtype=torch.int8, device=self.device)
        self.TNHW_boolMask = torch.zeros((parallel_trial,self.rule_arrays_tensor.shape[0],H,W), dtype=torch.bool, device=self.device)
        self.tmp_mask = torch.zeros((parallel_trial,1,H,W), dtype=torch.int8, device=self.device)
        self.TCHW_applied = torch.zeros((parallel_trial,1,H,W), dtype=torch.bool, device=self.device)
        self.shuffle_rule = self.rule_arrays_tensor.clone()
        self.rule_mask = torch.tensor(
            [[0,1,0],
             [1,1,1],
             [0,1,0]], dtype=torch.bool, device=self.device
        )
    

    # 任意ステップ数だけセル空間を更新する
    def run_steps(self, steps: int, global_prob: float, seed: int = 0):
        print(f"Run steps: {steps}")
        for i in range(steps):
            print(f"Step {i}")
            ###################
            # 乱数生成器の定義  #
            ###################
            self.rng.manual_seed(i + 65536 + seed)
            self.update_cellspace(
                global_prob=global_prob
            )

    ###################################
    # セル空間を更新する関数を定義する
    # 引数は全てtorchテンソルにして、cudaデバイス上で実行できるようにする
    ###################################
    def update_cellspace(self,
                         global_prob: float | None,            # グローバル確率
                        ) -> torch.Tensor:

        ####################################
        # 更新に使うテンソルの定義と整合性チェック  #
        ####################################

        # TCHW: [Trial, 1, H, W] dtype=int8, Trial別セル空間配列(引数, 戻り値)
        # ループ前処理は特に必要ない

        # rule_arrays: [N,2,3,3] dtype=int8, N種類の遷移規則を記録した配列(引数)
        # ループ前処理は特に必要ない
        
        # rule_mask: [[0, 1, 0], [1, 1, 1], [0, 1, 0]] dtype=bool, 四近傍遷移規則マッチング用のマスク配列
        # ループ前処理は特に必要ない

        # rule_probs: [N] dtype=float32, N種類の遷移規則の確率配列(引数)
        # ループ前処理は特に必要ない

        # Pickup_rule: [2, 3, 3] dtype=int8, ループ内でシャッフル遷移規則から取り出す遷移規則
        # ループ前処理は特に必要ない, シャッフルルールテンソルから取り出す時上書きするため
        
        # TNHW_boolMask: [Trial, N, H, W] dtype=bool, 取り出した遷移規則をマッチして適用できたセルの中心座標を1とするboolマスク
        # 初期化
        self.TNHW_boolMask.fill_(False)

        # tmp_mask: [Trial, 1, H, W] dtype=int8, 取り出した遷移規則により書き換えられる差分セルだけを検査するk_writeカーネルを元に書き換え予定のセルに1を足していくためのテンソル
        # 初期化
        self.tmp_mask.fill_(0)

        # TCHW_applied: [Trial, 1, H, W] dtype=bool, 今までの遷移規則の適用により書き換えが行われたセルに1を立てておくboolマスク
        # 初期化
        self.TCHW_applied.fill_(False)

        ####################################
        # 遷移規則マッチングとグローバル確率ゲート #
        ####################################

        # TCHW, ruleテンソルによりTNHW_boolMaskにマッチした3*3領域の中心座標を1へ(畳み込みによるセル空間とtrialと全遷移規則に並列な処理)
        self.TNHW_boolMask = self._match_centers_all_rules()

        # グローバル確率ゲート(セル空間とtrialと全遷移規則に並列な処理)

        ################################
        # 遷移規則のシャッフル(トライアル共有) #
        ################################
        # 遷移規則のシャッフル
        N = self.rule_arrays_tensor.shape[0]
        perm = torch.randperm(N, generator=self.rng, device=self.device)
        self.shuffle_rule  = self.rule_arrays_tensor.index_select(0, perm)
        self.shuffle_probs = self.rule_probs_tensor.index_select(0, perm)

        ########################################
        # シャッフル遷移規則配列の要素順にループを回す  #
        ########################################
        for i in range(N):
            # 遷移規則の取り出し
            self.Pickup_rule = self.shuffle_rule[i]

            # 遷移規則の確率の取り出し
            self.Pickup_rule_prob = self.shuffle_probs[i]

            # 遷移規則確率ゲート(セル空間とtrialに並列な処理)
            rule_prob = self.shuffle_probs[i]

            # 規則内競合解決(セル空間とtrialに並列な処理)

            # 他規則競合解決(セル空間とtrialに並列な処理)

            # 書き換え実行(セル空間とtrialに並列な処理)
            
        # loop end
        return self.TCHW

    # 特殊イベントを適用する
    def apply_spatial_events(self):
        pass
###################################

    def _match_centers_all_rules(self) -> torch.Tensor:
        """
        Returns:
            TNHW bool: 各 trial × 各 rule × 各セル中心が「preの3x3（四近傍）と一致」したか
        振る舞い:
            - 境界外は0として照合
            - rule_mask が False の位置は「不問（常に一致扱い）」
            - -1 は通常の状態として厳密一致（ワイルドカードではない）
        """
        assert hasattr(self, "TCHW"), "call set_ParallelTrial() first."
        T, _, H, W = self.TCHW.shape

        # 使うルール集合（シャッフル済みがあればそれ、無ければ元のテンソル）
        rules = getattr(self, "shuffle_rule", self.rule_arrays_tensor)   # [N,2,3,3] int8
        N = rules.shape[0]

        # 3x3 pre パターン（float32で比較。-1等もそのまま厳密比較）
        pre = rules[:, 0].float()                                        # [N,3,3]

        # 境界外=0 でパディング→unfoldして全3×3近傍を一括取得
        x = self.TCHW.float()                                            # [T,1,H,W] float32に変換
        x_pad = F.pad(x, pad=(1,1,1,1), mode="constant", value=0)        # [T,1,H+2,W+2]
        # unfold: [T, 1*9, H*W] → 形を戻す
        nbh = F.unfold(x_pad, kernel_size=3, padding=0, stride=1)        # [T,9,H*W]
        nbh = nbh.view(T, 1, 3, 3, H, W)                                 # [T,1,3,3,H,W]

        # ブロードキャスト比較（全位置）
        # nbh: [T,1,3,3,H,W] vs pre: [N,3,3] → [T,N,3,3,H,W]
        eq = (nbh == pre.view(1, N, 3, 3, 1, 1))                         # float32で比較

        # 四近傍マスク適用：False の位置は「不問」= 常に一致扱い
        mask = self.rule_mask.bool()                                     # [3,3]
        eq_masked = eq | (~mask.view(1, 1, 3, 3, 1, 1))

        # 3×3 の全てが一致 → 中心だけ True のマスク（中心も mask=True なので厳密に比較される）
        # [T,N,3,3,H,W] → [T,N,9,H,W] → all → [T,N,H,W]
        match_tnhw = eq_masked.view(T, N, 9, H, W).all(dim=2).contiguous()

        # 必要なら「何かの規則に一致した」T×1×H×W も更新しておく（従来のTCHW_boolMask互換）
        if hasattr(self, "TCHW_boolMask"):
            self.TCHW_boolMask.copy_(match_tnhw.any(dim=1, keepdim=True))

        return match_tnhw                                       # [T,N,H,W]