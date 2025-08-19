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
            self.spatial_event_names  = lib.get_event_names_from_file(spatial_event_filePath)
        else:
            self.spatial_event_arrays = None    
            self.spatial_event_names  = None
        
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

        # trial ごとの {event_name: [steps...]} 辞書を作る
        if getattr(self, "spatial_event_names", None) is not None:
            names = list(self.spatial_event_names)
            self.event_history = [{name: [] for name in names} for _ in range(parallel_trial)]
        else:
            self.event_history = None
    

    # 任意ステップ数だけセル空間を更新する
    def run_steps(self, steps: int, global_prob: float, seed: int = 0, debug: bool = False, debug_per_trial: bool = False):
        print(f"Run steps: {steps}")
        for i in range(steps):
            print(f"Step {i}")
            self._current_step = i
            ###################
            # 乱数生成器の定義  #
            ###################
            self.rng.manual_seed(i + 65536 + seed)
            self.update_cellspace(
                global_prob=global_prob,
                debug=debug,
                debug_per_trial=debug_per_trial
            )

            # 特殊イベントの適用
            self.apply_spatial_events()

    ###################################
    # セル空間を更新する関数を定義する
    # 引数は全てtorchテンソルにして、cudaデバイス上で実行できるようにする
    ###################################
    def update_cellspace(self,
                         global_prob: float | None,            # グローバル確率
                         debug: bool = False,                  # デバッグモード
                         debug_per_trial: bool = False         # トライアル別詳細統計
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
        
        if debug:
            print("After Match Centers All Rules")
            self.debug(show_per_trial=debug_per_trial)

        # グローバル確率ゲート(セル空間とtrialと全遷移規則に並列な処理)
        self.TNHW_boolMask = self._global_prob_gate(global_prob)

        if debug:
            print("After Global Prob Gate")
            self.debug(show_per_trial=debug_per_trial)

        # 遷移規則確率ゲート(セル空間とtrialと全遷移規則に並列な処理)
        self.TNHW_boolMask = self._rule_prob_gate()

        if debug:
            print("After Rule Prob Gate")
            self.debug(show_per_trial=debug_per_trial)

        ################################
        # 遷移規則のシャッフル(トライアル共有) #
        ################################
        # 遷移規則のシャッフル
        N = self.rule_arrays_tensor.shape[0]
        perm = torch.randperm(N, generator=self.rng, device=self.device)
        index_shuffle = perm

        ########################################
        # シャッフル遷移規則配列の要素順にループを回す  #
        ########################################
        for i in range(N):
            # 遷移規則の取り出し
            idx = index_shuffle[i]
            self.Pickup_rule = self.rule_arrays_tensor[idx]

            # 規則内競合解決(セル空間とtrialに並列な処理)
            self.TNHW_boolMask[:, idx, :, :] = self._rule_inner_conflict_resolution(idx)
            if debug:
                print(f"After Rule Inner Conflict Resolution {idx}")
                #self.debug(show_per_trial=debug_per_trial)

            # 他規則競合解決(セル空間とtrialに並列な処理)
            self.TNHW_boolMask[:, idx, :, :] = self._rule_outer_conflict_resolution(idx)
            if debug:
                print(f"After Rule Outer Conflict Resolution {idx}")
                #self.debug(show_per_trial=debug_per_trial)

            # 書き換え実行(セル空間とtrialに並列な処理)
            self._write_back(idx)
            
        # loop end
        if debug:
            print("After Write Back")
            self.debug(show_per_trial=debug_per_trial)
        return self.TCHW

    # 特殊イベントを適用する
    def apply_spatial_events(self) -> None:
        """
        lib.load_special_events_from_file() の形式に準拠:
          1行 = [ref_x, ref_y, ref_state, write_x, write_y, write_state]（global座標, X→Y）
        (ref_x,ref_y) が ref_state のとき (write_x,write_y) に write_state を確定書き込み。
        全 trial に同一内容を適用。完全ベクトル化。競合なし前提。
        """
        ev = getattr(self, "spatial_event_arrays_tensor", None)
        if ev is None or ev.numel() == 0:
            return

        # T, H, W が無いと何もできない
        assert hasattr(self, "TCHW"), "call set_ParallelTrial() first."
        T, _, H, W = self.TCHW.shape

        device = self.device
        ev = ev.to(device=device, dtype=torch.int64)

        # !!! ここがポイント: X→Y の順で読む !!!
        pre_x_g, pre_y_g = ev[:, 0], ev[:, 1]
        pre_v            = ev[:, 2].to(torch.int8)
        pst_x_g, pst_y_g = ev[:, 3], ev[:, 4]
        pst_v            = ev[:, 5].to(torch.int8)

        # global -> local（オフセットは X と Y をそれぞれ引く）
        pre_x = pre_x_g - self.offset_x
        pre_y = pre_y_g - self.offset_y
        pst_x = pst_x_g - self.offset_x
        pst_y = pst_y_g - self.offset_y

        # 画面内だけ採用
        keep = (pre_x >= 0) & (pre_x < W) & (pre_y >= 0) & (pre_y < H) & \
               (pst_x >= 0) & (pst_x < W) & (pst_y >= 0) & (pst_y < H)
        if not keep.any():
            return

        pre_x = pre_x[keep].to(torch.int64)
        pre_y = pre_y[keep].to(torch.int64)
        pre_v = pre_v[keep]
        pst_x = pst_x[keep].to(torch.int64)
        pst_y = pst_y[keep].to(torch.int64)
        pst_v = pst_v[keep]

        # 線形Index化（行優先: y*W + x）
        pre_lin = pre_y * W + pre_x       # [E]
        pst_lin = pst_y * W + pst_x       # [E]

        # (T×E) で条件判定
        cs_flat    = self.TCHW[:, 0].reshape(T, H * W)                         # [T, H*W] int8 (ビュー)
        cur_pre_TE = cs_flat.gather(1, pre_lin.view(1, -1).expand(T, -1))      # [T, E]
        cond_TE    = (cur_pre_TE == pre_v.view(1, -1).expand(T, -1))           # [T, E] bool
        if not cond_TE.any():
            if getattr(self, "_debug_spatial_events", False):
                print("[events] no match: kept", keep.sum().item())
            return

        # 条件を満たす (t,e) を抽出し、(t*HW + pst_lin[e]) へ書き込み
        hit   = cond_TE.nonzero(as_tuple=False)    # [K,2] (t,e)
        t_idx = hit[:, 0]
        e_idx = hit[:, 1]
        dst   = (t_idx.to(torch.int64) * (H * W) + pst_lin[e_idx]).to(torch.int64)
        wvals = pst_v[e_idx]                       # int8

        # self.TCHW をインプレース更新（ビューに対する index_copy_）
        cs_all = self.TCHW[:, 0].reshape(T * H * W)   # int8, ビュー
        cs_all.index_copy_(0, dst, wvals)

        # 適用フラグ（任意）
        if hasattr(self, "TCHW_applied"):
            applied_all = self.TCHW_applied[:, 0].reshape(T * H * W)
            applied_all.index_fill_(0, dst, True)

        if getattr(self, "_debug_spatial_events", False):
            print(f"[events] total={ev.shape[0]}, kept={keep.sum().item()}, hits={hit.shape[0]}")
            print("  sample(local):",
                  list(zip(pre_x.tolist()[:3], pre_y.tolist()[:3], pre_v.tolist()[:3],
                           pst_x.tolist()[:3], pst_y.tolist()[:3], pst_v.tolist()[:3])))

        # 既存：hit, t_idx, e_idx を算出済み
        # names を keep マスクに合わせて整列
        if getattr(self, "event_history", None) is not None:
            # GPUの keep をCPU boolsへ
            keep_list = keep.detach().to('cpu').tolist()
            names_kept = [n for n, k in zip(self.spatial_event_names, keep_list) if k]

        step = getattr(self, "_current_step", None)
        if step is not None and hit.shape[0] > 0:
            # (t,e) ごとに発火ステップを記録
            for t, e in zip(t_idx.tolist(), e_idx.tolist()):
                self.event_history[t][names_kept[e]].append(step)

###################################

    # 遷移規則マッチング
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

        # --- 動的マスク（四隅に非ゼロがあるルールだけ8近傍を使う） ---
        cross_mask = self.rule_mask.bool()  # [3,3] 十字
        corner_mask = torch.tensor(
            [[1,0,1],
             [0,0,0],
             [1,0,1]], dtype=torch.bool, device=self.device
        )  # 角だけTrue

        # ルールごとに「見る位置」= 十字 + （角で pre!=0 の場所）
        # ※ pre==0 は“未指定”として扱う前提。0を条件にしたい設計なら別途「指定マスク」が必要。
        use_corners = (pre != 0) & corner_mask                # [N,3,3]
        mask_N = cross_mask.view(1,3,3).expand(N,-1,-1) | use_corners  # [N,3,3]

        # マスクFalseは“不問”＝常に一致扱い
        eq_masked = eq | (~mask_N.view(1, N, 3, 3, 1, 1))

        # 3×3 の全てが一致 → 中心だけ True のマスク（中心も mask=True なので厳密に比較される）
        # [T,N,3,3,H,W] → [T,N,9,H,W] → all → [T,N,H,W]
        match_tnhw = eq_masked.view(T, N, 9, H, W).all(dim=2).contiguous()

        # 必要なら「何かの規則に一致した」T×1×H×W も更新しておく（従来のTCHW_boolMask互換）
        if hasattr(self, "TCHW_boolMask"):
            self.TCHW_boolMask.copy_(match_tnhw.any(dim=1, keepdim=True))

        return match_tnhw                                       # [T,N,H,W]

    # グローバル確率ゲート
    def _global_prob_gate(self, global_prob: float | torch.Tensor | None) -> torch.Tensor:
        """
        TNHW 粒度でグローバル確率ゲートを適用。
        global_prob: スカラー / [N] / [T,N] / [T,N,H,W] / None に対応
        返り値: [T,N,H,W] bool
        """
        assert hasattr(self, "TNHW_boolMask"), "先に _match_centers_all_rules() を呼んでください。"
        base = self.TNHW_boolMask
        T, N, H, W = base.shape
        device = base.device

        # p を作成（None は 1.0 として扱う）
        if global_prob is None:
            return base
        p = torch.as_tensor(global_prob, dtype=torch.float32, device=device).clamp_(0, 1)

        # 形状正規化 → (T,N,H,W) へブロードキャスト
        if p.ndim == 0:                              # スカラー
            p = p.view(1,1,1,1).expand(T,N,H,W)
        elif p.ndim == 1:                            # [N]
            if p.shape[0] != N:
                raise ValueError(f"global_prob length {p.shape[0]} != N={N}")
            p = p.view(1,N,1,1).expand(T,N,H,W)
        elif p.ndim == 2:                            # [T,N] or [N,T]
            if p.shape == (N,T):
                p = p.t().contiguous()
            if p.shape != (T,N):
                raise ValueError(f"global_prob shape {tuple(p.shape)} must be [T,N]")
            p = p.view(T,N,1,1).expand(T,N,H,W)
        elif p.ndim == 4:                            # [T,N,H,W]
            if p.shape != (T,N,H,W):
                raise ValueError(f"global_prob shape {tuple(p.shape)} must be [T,N,H,W]")
        else:
            raise ValueError("global_prob はスカラー / [N] / [T,N] / [T,N,H,W] / None に対応します。")

        if torch.all(p == 1):
            return base.clone()
        if torch.all(p == 0):
            return torch.zeros_like(base, dtype=torch.bool)

        rnd  = torch.rand((T,N,H,W), device=device, generator=self.rng)
        gate = (rnd < p)
        return (base & gate).contiguous()
        
    # 遷移規則確率ゲート
    def _rule_prob_gate(self) -> torch.Tensor:
        """
        TNHW 粒度で遷移規則適用確率ゲートを適用。
        self.rule_probs_tensor の想定形状は [N]（ルールごと確率）
        ただし、[T,N] / [T,N,H,W] に差し替えても動くように対応。
        返り値: [T,N,H,W] bool
        """
        assert hasattr(self, "TNHW_boolMask"), "先に _match_centers_all_rules() を呼んでください。"
        base = self.TNHW_boolMask
        T, N, H, W = base.shape
        device = base.device

        p = torch.as_tensor(self.rule_probs_tensor, dtype=torch.float32, device=device).clamp_(0, 1)

        # 形状正規化 → (T,N,H,W) へブロードキャスト
        if p.ndim == 0:                               # 全ルール同一確率にしたい場合も一応サポート
            p = p.view(1,1,1,1).expand(T,N,H,W)
        elif p.ndim == 1:                             # [N]
            if p.shape[0] != N:
                raise ValueError(f"rule_probs length {p.shape[0]} != N={N}")
            p = p.view(1,N,1,1).expand(T,N,H,W)
        elif p.ndim == 2:                             # [T,N] or [N,T]
            if p.shape == (N,T):
                p = p.t().contiguous()
            if p.shape != (T,N):
                raise ValueError(f"rule_probs shape {tuple(p.shape)} must be [T,N]")
            p = p.view(T,N,1,1).expand(T,N,H,W)
        elif p.ndim == 4:                             # [T,N,H,W]
            if p.shape != (T,N,H,W):
                raise ValueError(f"rule_probs shape {tuple(p.shape)} must be [T,N,H,W]")
        else:
            raise ValueError("rule_probs は [N] を基本とし、[T,N] / [T,N,H,W] / スカラーにも対応します。")

        if torch.all(p == 1):
            return base.clone()
        if torch.all(p == 0):
            return torch.zeros_like(base, dtype=torch.bool)

        rnd  = torch.rand((T,N,H,W), device=device, generator=self.rng)
        gate = (rnd < p)
        return (base & gate).contiguous()

    def _rule_outer_conflict_resolution(self, rule_idx: int) -> torch.Tensor:
        """
        既に確定（書込み済み）セルと衝突する書込みを除外。
        返り値: 競合除外後の中心マスク [T,H,W] bool
        """
        center = self.TNHW_boolMask[:, rule_idx, :, :]           # [T,H,W] bool
        wf, wb, _ = self._build_rule_kernels(rule_idx)
        if wf is None:
            return center

        # このルールが書こうとしている任意ターゲット
        tgt_any = (F.conv2d(center.unsqueeze(1).float(), wf, padding=1) > 0.5).any(dim=1)  # [T,H,W]
        # 既に確定済みセルと衝突
        hit = tgt_any & self.TCHW_applied[:, 0]

        if not hit.any():
            return center

        # 衝突ターゲットに寄与した中心を逆射影で特定し除外
        bad_center = (F.conv2d(hit.unsqueeze(1).float(), wb, padding=1) > 0.5) \
                     .any(dim=1) & center
        return (center & ~bad_center)

    def _build_rule_kernels(self, rule_idx: int):
        pre  = self.rule_arrays_tensor[rule_idx, 0]   # [3,3] int8
        post = self.rule_arrays_tensor[rule_idx, 1]   # [3,3] int8

        # 書き込み対象は pre↔post の差分セルのみ（post==-1 は変更なし）
        write_pos = (post != pre) & (post != -1)

        idx = torch.nonzero(write_pos, as_tuple=False)  # [M,2] (y,x)
        if idx.numel() == 0:
            return None, None, None

        y, x = idx[:, 0], idx[:, 1]
        M = idx.shape[0]

        # 中心→ターゲット: conv2d の仕様上 (2-y, 2-x) に 1 を置く
        w_fwd = torch.zeros((M, 1, 3, 3), device=self.device, dtype=torch.float32)
        w_fwd[torch.arange(M, device=self.device), 0, 2 - y, 2 - x] = 1.0

        # ターゲット→中心: (y, x) に 1 を置く（逆射影）
        w_back = torch.zeros_like(w_fwd)
        w_back[torch.arange(M, device=self.device), 0, y, x] = 1.0

        # 各ターゲットへ書く値
        v_post = post[y, x].to(torch.int8)
        return w_fwd, w_back, v_post

    def _rule_inner_conflict_resolution(self, rule_idx: int) -> torch.Tensor:
        """
        同一ルール内で同一ターゲットに複数中心から書こうとする競合を除外。
        返り値: 競合除外後の中心マスク [T,H,W] bool
        """
        center = self.TNHW_boolMask[:, rule_idx, :, :]           # [T,H,W] bool
        wf, wb, _ = self._build_rule_kernels(rule_idx)
        if wf is None:
            return center

        c = center.unsqueeze(1).float()                          # [T,1,H,W]
        tgt = (F.conv2d(c, wf, padding=1) > 0.5)                 # [T,M,H,W] bool

        # 同一ルール内のターゲット重複
        conflict = (tgt.sum(dim=1) >= 2)                         # [T,H,W] bool
        if not conflict.any():
            return center

        # 競合ターゲットに寄与した中心を逆射影で除外
        bad_center = (F.conv2d(conflict.unsqueeze(1).float(), wb, padding=1) > 0.5) \
                     .any(dim=1) & center
        return (center & ~bad_center)

    def _write_back(self, rule_idx: int) -> None:
        """
        競合解決後の中心マスクに基づいて差分セルのみ書き込み。
        """
        center = self.TNHW_boolMask[:, rule_idx, :, :]           # [T,H,W] bool
        wf, _, v_post = self._build_rule_kernels(rule_idx)
        if wf is None:
            return

        # 中心→ターゲット（M 本）
        tgt = (F.conv2d(center.unsqueeze(1).float(), wf, padding=1) > 0.5)  # [T,M,H,W] bool
        if not tgt.any():
            return

        # 競合は事前解決済みなので合算でOK
        cs = self.TCHW[:, 0]                                                     # [T,H,W] int8
        vals = tgt.to(torch.int16) * v_post.view(1, -1, 1, 1).to(torch.int16)    # [T,M,H,W]
        write_val  = vals.sum(dim=1).to(torch.int8)                               # [T,H,W]
        write_mask = tgt.any(dim=1)                                               # [T,H,W]

        self.TCHW[:, 0] = torch.where(write_mask, write_val, cs)
        self.TCHW_applied[:, 0] |= write_mask

    # デバッグ情報
    def debug(self, show_per_trial: bool = False):
        # デバッグ情報
        print(f"Pattern matching debug:")
        print(f"  TNHW_boolMask shape: {self.TNHW_boolMask.shape}")
        
        # 全体の統計
        total_matches = self.TNHW_boolMask.sum().item()
        print(f"  Total matches: {total_matches}")
        print(f"  TCHW unique values: {torch.unique(self.TCHW)}")
        print(f"  Rule pattern unique values: {torch.unique(self.rule_arrays_tensor[:, 0])}")
        
        # 各ルールのマッチ数を計算（全トライアル合計）
        rule_matches = self.TNHW_boolMask.sum(dim=(0, 2, 3))  # [N] - 各ルールのマッチ数
        matched_rules = torch.nonzero(rule_matches).squeeze(-1)
        print(f"  Matched rules count: {len(matched_rules)}")
        
        if len(matched_rules) > 0:
            print(f"  Rules with matches:")
            for rule_idx in matched_rules:
                count = rule_matches[rule_idx].item()
                print(f"    Rule {rule_idx}: {count} matches")
                
                # 各ルールの具体的なマッチ位置を表示（最初の5個まで）
                rule_positions = torch.nonzero(self.TNHW_boolMask[:, rule_idx, :, :])
                if len(rule_positions) > 0:
                    positions_to_show = min(5, len(rule_positions))
                    print(f"      Positions (first {positions_to_show}): {rule_positions[:positions_to_show]}")
        else:
            print(f"  No matches found - pattern mismatch")
        
        # トライアル別詳細統計
        if show_per_trial:
            T = self.TNHW_boolMask.shape[0]
            print(f"\n  Per-trial statistics:")
            for trial_idx in range(T):
                trial_matches = self.TNHW_boolMask[trial_idx].sum().item()
                print(f"    Trial {trial_idx}: {trial_matches} matches")
                
                # トライアル別ルールマッチ数
                trial_rule_matches = self.TNHW_boolMask[trial_idx].sum(dim=(1, 2))  # [N]
                trial_matched_rules = torch.nonzero(trial_rule_matches).squeeze(-1)
                
                if len(trial_matched_rules) > 0:
                    rule_details = []
                    for rule_idx in trial_matched_rules:
                        count = trial_rule_matches[rule_idx].item()
                        rule_details.append(f"Rule {rule_idx}:{count}")
                    print(f"      Rules: {', '.join(rule_details)}")
                else:
                    print(f"      No matches in this trial")
        
        print()

    def save_event_histry_for_dataframe(
        self,
        path: str | None = None,
        format: str = "parquet",              # "parquet" | "csv" | "jsonl" | "yaml" | "jsonl_trials" | "jsonl_trials_dict"
        deduplicate: bool = False,            # Trueなら各(eventのsteps)を集合化→昇順
        return_df: bool = True,               # TrueならDataFrameを返す（jsonl_trials系は保存が主目的）
        parquet_compression: str = "snappy"   # "snappy" | "zstd" など
    ):
        """
        event_history を保存/変換するユーティリティ。

        event_history 仕様:
          self.event_history: List[Dict[str, List[int]]]
            └ 各trialについて { event_name: [fired_steps...] } の辞書
        追加対応:
          - format="jsonl_trials": 1 trial = 1行。{"trial":t,"events":[["name",[...]], ...]}
          - format="jsonl_trials_dict": 1 trial = 1行。{"trial":t,"events":{"name":[...], ...}}

        既存:
          - "parquet"/"csv"/"jsonl"/"yaml" はフラット表 (trial,event,step)

        注意:
          - deduplicate=True のとき各イベントの step を set→昇順にします
          - "jsonl_trials"系は保存が主目的。return_df=Trueでもフラット表を返します
        """
        if getattr(self, "event_history", None) is None:
            raise RuntimeError("event_history がありません。set_ParallelTrial() と run_steps() の後に呼んでください。")

        import os, json
        import pandas as pd

        # ---- フラット表（既存互換; DataFrame作成用）----
        rows = []
        for t, edict in enumerate(self.event_history):
            for name, steps in edict.items():
                steps_iter = sorted(set(int(s) for s in steps)) if deduplicate else (int(s) for s in steps)
                for s in steps_iter:
                    rows.append({"trial": int(t), "event": str(name), "step": int(s)})

        df = pd.DataFrame(rows, columns=["trial", "event", "step"])
        if not df.empty:
            df.sort_values(["trial", "event", "step"], inplace=True, kind="mergesort")
            df.reset_index(drop=True, inplace=True)
        else:
            try:
                df = df.astype({"trial": "int64", "event": "string", "step": "int64"})
            except Exception:
                pass

        # 保存不要ならここで返す
        if path is None:
            return df if return_df else None

        fmt = format.lower()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # ---- 追加: 1 trial = 1行 の JSONL 出力（ペア配列 or dict）----
        if fmt in ("jsonl_trials", "jsonl_trials_dict"):
            # 全イベント名を揃えたい場合は self.event_names を使用、無ければ union(keys)
            if hasattr(self, "event_names") and self.event_names:
                all_names = list(self.event_names)
            else:
                name_set = set()
                for ed in self.event_history:
                    name_set.update(ed.keys())
                all_names = sorted(name_set)

            with open(path, "w", encoding="utf-8") as f:
                for t, edict in enumerate(self.event_history):
                    # イベントごとの steps を準備（存在しないイベントは空リスト）
                    def _steps_list(v):
                        arr = list(map(int, v)) if v is not None else []
                        return sorted(set(arr)) if deduplicate else arr

                    if fmt == "jsonl_trials":
                        # 期待例: ["core_input_1",[...]], ...
                        events_payload = [[name, _steps_list(edict.get(name, []))] for name in all_names]
                    else:
                        # dict 版
                        events_payload = {name: _steps_list(edict.get(name, [])) for name in all_names}

                    rec = {"trial": int(t), "events": events_payload}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            return df if return_df else path

        # ---- 既存のフラット表フォーマット出力 ----
        if fmt == "parquet":
            try:
                df.to_parquet(path, index=False, compression=parquet_compression)
            except Exception as e:
                raise RuntimeError("to_parquet失敗（pandas/pyarrow/fastparquetが必要）: %s" % e)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "jsonl":
            df.to_json(path, orient="records", lines=True, force_ascii=False)
        elif fmt == "yaml":
            try:
                import yaml
            except ImportError as e:
                raise RuntimeError("YAML出力には PyYAML が必要です（pip install pyyaml）。") from e
            with open(path, "w", encoding="utf-8") as f:
                import pandas as pd
                yaml.safe_dump(df.to_dict(orient="records"), f, allow_unicode=True, sort_keys=False)
        else:
            raise ValueError("format は parquet/csv/jsonl/yaml/jsonl_trials/jsonl_trials_dict のいずれかです。")

        return df if return_df else path