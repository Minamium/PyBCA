# GUI, CLIツール共通のライブラリを作成
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import yaml
import torch


# yamlセル空間ファイルをnumpy配列に変換する
def load_cell_space_yaml_to_numpy(path: str, progress_callback=None, include_offset=True) -> np.ndarray:
    """
    セル空間YAML（形式: [{'coord':{'x':..,'y':..}, 'value':..}, ...]）を
    最小外接矩形で切り出した 2D NumPy 配列に変換して返す。
    - 未指定セルは 0 で初期化
    - 負の座標は配列内で 0 起点となるよう平行移動
    - 値はそのまま格納（例: -1, 1, 2 など）

    Args:
        path: セル空間YAMLファイルのパス
        progress_callback: プログレス報告用コールバック関数 (current, total)

    Returns:
        np.ndarray[int8] 形状 (H, W)
    """
    if progress_callback:
        progress_callback(0, 100)  # 開始
    
    # ファイル読み込み
    if progress_callback:
        progress_callback(5, 100)  # ファイルオープン
    
    with open(path, "r", encoding="utf-8") as f:
        if progress_callback:
            progress_callback(10, 100)  # ファイル読み込み開始
        items: List[Dict[str, Any]] = yaml.safe_load(f)
    
    if progress_callback:
        progress_callback(15, 100)  # YAML読み込み完了

    if not isinstance(items, list) or not items:
        raise ValueError("YAMLのトップは非空のリストである必要があります")

    xs, ys, vs = [], [], []
    total_items = len(items)
    
    if progress_callback:
        progress_callback(20, 100)  # 座標解析開始
    
    # 座標解析（これが最も時間がかかる処理）
    for i, it in enumerate(items):
        c = it.get("coord", {})
        xs.append(int(c["x"]))
        ys.append(int(c["y"]))
        vs.append(int(it.get("value", 0)))
        
        # より頻繁な進捗報告（座標解析: 20-85%）
        if progress_callback and i % max(1, total_items // 50) == 0:
            progress = 20 + int(65 * i / total_items)
            progress_callback(progress, 100)

    if progress_callback:
        progress_callback(85, 100)  # 座標解析完了
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = max_x - min_x + 1
    H = max_y - min_y + 1

    if progress_callback:
        progress_callback(87, 100)  # 配列サイズ計算完了

    # YAML座標→配列添字の平行移動
    off_x = -min_x
    off_y = -min_y

    if progress_callback:
        progress_callback(90, 100)  # 配列初期化開始
    
    if include_offset:
        # オフセット情報を埋め込んだ配列を作成 (H+1, W+2)
        # セル空間はint8、オフセット情報はint16で保存
        arr = np.zeros((H + 1, W + 2), dtype=np.int16)
        
        if progress_callback:
            progress_callback(93, 100)  # 配列初期化完了
        
        # セル空間データを配置
        xs_a = np.asarray(xs) + off_x
        ys_a = np.asarray(ys) + off_y
        vs_a = np.asarray(vs, dtype=np.int8)
        arr[ys_a, xs_a] = vs_a
        
        # オフセット情報を埋め込み (元座標系のmin値を保存)
        arr[H, 0] = min_x  # 元座標系のmin_x
        arr[H, 1] = min_y  # 元座標系のmin_y
        
        if progress_callback:
            progress_callback(97, 100)  # 配列代入完了
    else:
        # 従来通りのセル空間のみ
        arr = np.zeros((H, W), dtype=np.int8)
        
        if progress_callback:
            progress_callback(93, 100)  # 配列初期化完了
        
        xs_a = np.asarray(xs) + off_x
        ys_a = np.asarray(ys) + off_y
        vs_a = np.asarray(vs, dtype=np.int8)
        arr[ys_a, xs_a] = vs_a
        
        if progress_callback:
            progress_callback(97, 100)  # 配列代入完了
    
    if progress_callback:
        progress_callback(100, 100)  # 完了
    
    return arr

def extract_cellspace_and_offset(arr_with_offset: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    オフセット情報付きNumPy配列からセル空間とオフセット情報を分離
    
    Args:
        arr_with_offset: オフセット情報付き配列 (H+1, W+2)
    
    Returns:
        (cellspace, min_x, min_y)
        cellspace: セル空間データ (H, W) int8型
        min_x, min_y: 元座標系の最小値
    """
    if arr_with_offset.ndim != 2 or arr_with_offset.shape[1] < 2:
        raise ValueError("Invalid array format for offset extraction")
    
    H_plus_1, W_plus_2 = arr_with_offset.shape
    H, W = H_plus_1 - 1, W_plus_2 - 2
    
    # セル空間データを抽出してint8に変換
    cellspace = arr_with_offset[:H, :W].astype(np.int8)
    
    # オフセット情報を抽出
    min_x = int(arr_with_offset[H, 0])
    min_y = int(arr_with_offset[H, 1])
    
    return cellspace, min_x, min_y

def has_offset_info(arr: np.ndarray) -> bool:
    """
    配列にオフセット情報が含まれているかチェック
    
    Args:
        arr: チェック対象の配列
    
    Returns:
        bool: オフセット情報が含まれている場合True
    """
    return arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 1



# numpy配列のセル空間をyamlファイルに変換する
def numpy_to_cell_space_yaml(arr: np.ndarray, path: str) -> None:
    """
    2次元 NumPy 配列を、読み込みと同じ YAML 形式
      [{'coord': {'x': int, 'y': int}, 'value': int}, ...]
    に変換して保存する。

    仕様:
      - 配列は (H, W) の2次元を想定
      - 値が 0 のセルは省略（未指定セルは 0 とみなす規則に合わせる）
      - すべて 0 の場合は [{'coord':{'x':0,'y':0}, 'value':0}] を書き出す
        （読み込み側の「非空リスト必須」制約を満たすため）
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError("2次元の NumPy 配列を渡してください")

    H, W = arr.shape
    items: List[Dict[str, Any]] = []

    # 行優先（y→x）の順でソートされるように走査
    for y in range(H):
        row = arr[y]
        for x in range(W):
            v = int(row[x])
            if v != 0:
                items.append({"coord": {"x": int(x), "y": int(y)}, "value": v})

    # 全要素0なら、非空リストにするためダミー1要素を出力
    if not items:
        items = [{"coord": {"x": 0, "y": 0}, "value": 0}]

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(items, f, sort_keys=False, allow_unicode=True)
    
# 遷移規則をyamlファイルより読み込み、numpy配列に変換する
@dataclass
class TransitionRule:
    """遷移規則の前状態・後状態ペア"""
    rule_id: int
    prev_pattern: np.ndarray  # 前状態パターン
    next_pattern: np.ndarray  # 後状態パターン
    probability: float = 1.0  # 遷移確率（デフォルト1.0=確定的遷移）

def load_transition_rules_yaml(path: str) -> List[TransitionRule]:
    """
    rule.yaml から全ての遷移規則を読み込み、TransitionRule のリストとして返す。
    各規則の prev と next を最小外接矩形で切り出した NumPy 配列に変換。
    """
    import re
    from yaml import SafeLoader

    # ---- YAMLローダ: !OccupiedBy N をそのまま int(N) にする ----
    class _Loader(SafeLoader):
        pass

    def _occupied_by(loader, node):
        value = loader.construct_scalar(node)
        try:
            return int(value)
        except Exception:
            m = re.search(r"-?\d+", str(value))
            return int(m.group(0)) if m else 0

    _Loader.add_constructor("!OccupiedBy", _occupied_by)

    # ---- YAML読込 ----
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.load(f, Loader=_Loader)

    if not isinstance(doc, list):
        raise ValueError("rule.yaml のトップレベルはルールのリストである必要があります")

    rules = []

    def _to_state(v) -> int:
        if v is None:
            return 0
        if isinstance(v, (int, np.integer)):
            return int(v)
        s = str(v)
        if s.lower() == "vacant":
            return 0
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else 0

    def _pattern_to_array(pattern_list) -> np.ndarray:
        """パターンリストを NumPy 配列に変換"""
        if not pattern_list:
            return np.zeros((1, 1), dtype=np.int8)
        
        xs, ys, vs = [], [], []
        for it in pattern_list:
            if not isinstance(it, dict):
                continue
            if "coord" in it and isinstance(it["coord"], dict):
                x = int(it["coord"]["x"])
                y = int(it["coord"]["y"])
            else:
                x = int(it["x"])
                y = int(it["y"])
            state = _to_state(it.get("state", 0))
            xs.append(x); ys.append(y); vs.append(state)

        if not xs:
            return np.zeros((1, 1), dtype=np.int8)

        # 最小外接矩形
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        W = max_x - min_x + 1
        H = max_y - min_y + 1

        off_x = -min_x
        off_y = -min_y

        arr = np.zeros((H, W), dtype=np.int8)
        xs_a = np.asarray(xs, dtype=np.int32) + off_x
        ys_a = np.asarray(ys, dtype=np.int32) + off_y
        vs_a = np.asarray(vs, dtype=np.int16)
        vs_a = np.clip(vs_a, -128, 127).astype(np.int8)

        arr[ys_a, xs_a] = vs_a
        return arr

    for rule_item in doc:
        if not isinstance(rule_item, dict):
            continue
        
        rule_id = rule_item.get("id", 0)
        rule_data = rule_item.get("rule", {})
        
        if not isinstance(rule_data, dict):
            continue
        
        prev_list = rule_data.get("prev", [])
        next_list = rule_data.get("next", [])
        
        if not isinstance(prev_list, list) or not isinstance(next_list, list):
            continue
        
        prev_arr = _pattern_to_array(prev_list)
        next_arr = _pattern_to_array(next_list)
        
        # 確率の読み込み（オプション）
        probability = rule_item.get("probability", 1.0)
        if not isinstance(probability, (int, float)):
            probability = 1.0
        probability = max(0.0, min(1.0, float(probability)))  # 0.0-1.0の範囲に制限
        
        rules.append(TransitionRule(
            rule_id=rule_id,
            prev_pattern=prev_arr,
            next_pattern=next_arr,
            probability=probability
        ))

    return rules

def load_multiple_transition_rules_to_numpy(rule_file_paths: List[str]) -> np.ndarray:
    """
    複数の遷移規則ファイルを読み込み、統合されたnumpy配列として返す（確率情報なし）
    
    Args:
        rule_file_paths: 遷移規則YAMLファイルのパスリスト
    
    Returns:
        統合された遷移規則配列 (N, 2, 3, 3) 形状
        N: 規則数, 2: [prev_pattern, next_pattern], 3x3: パターンサイズ
    """
    import os
    all_rules = []
    
    # 複数ファイルから規則を読み込み
    for rule_path in rule_file_paths:
        if os.path.exists(rule_path):
            rules = load_transition_rules_yaml(rule_path)
            all_rules.extend(rules)
        else:
            raise FileNotFoundError(f"遷移規則ファイルが見つかりません: {rule_path}")
    
    if not all_rules:
        raise ValueError("遷移規則が読み込まれませんでした")
    
    # numpy配列に変換
    num_rules = len(all_rules)
    rule_array = np.zeros((num_rules, 2, 3, 3), dtype=np.int8)
    
    for i, rule in enumerate(all_rules):
        rule_array[i, 0] = rule.prev_pattern  # 前状態パターン
        rule_array[i, 1] = rule.next_pattern  # 後状態パターン
    
    return rule_array

def load_multiple_transition_rules_with_probability(rule_file_paths: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    複数の遷移規則ファイルを読み込み、パターン配列と確率配列を返す
    
    Args:
        rule_file_paths: 遷移規則YAMLファイルのパスリスト
    
    Returns:
        tuple: (rule_array, probability_array)
        - rule_array: 遷移規則配列 (N, 2, 3, 3) 形状
        - probability_array: 確率配列 (N,) 形状
    """
    import os
    all_rules = []
    
    # 複数ファイルから規則を読み込み
    for rule_path in rule_file_paths:
        if os.path.exists(rule_path):
            rules = load_transition_rules_yaml(rule_path)
            all_rules.extend(rules)
        else:
            raise FileNotFoundError(f"遷移規則ファイルが見つかりません: {rule_path}")
    
    if not all_rules:
        raise ValueError("遷移規則が読み込まれませんでした")
    
    # numpy配列に変換
    num_rules = len(all_rules)
    rule_array = np.zeros((num_rules, 2, 3, 3), dtype=np.int8)
    probability_array = np.ones(num_rules, dtype=np.float32)
    
    for i, rule in enumerate(all_rules):
        rule_array[i, 0] = rule.prev_pattern  # 前状態パターン
        rule_array[i, 1] = rule.next_pattern  # 後状態パターン
        probability_array[i] = rule.probability  # 遷移確率
    
    return rule_array, probability_array

def get_rule_ids_from_files(rule_file_paths: List[str]) -> List[int]:
    """
    複数の遷移規則ファイルからrule_idのリストを取得
    
    Args:
        rule_file_paths: 遷移規則YAMLファイルのパスリスト
    
    Returns:
        rule_idのリスト
    """
    import os
    all_rules = []
    
    for rule_path in rule_file_paths:
        if os.path.exists(rule_path):
            rules = load_transition_rules_yaml(rule_path)
            all_rules.extend(rules)
    
    return [rule.rule_id for rule in all_rules]

def load_special_events_from_file(event_file_path: str) -> List[tuple]:
    """
    特殊イベント定義ファイル(.py)から特殊イベントを読み込む
    
    Args:
        event_file_path: 特殊イベント定義Pythonファイルのパス
    
    Returns:
        特殊イベントのリスト [(name, ref_coord, ref_state, write_coord, write_state), ...]
    """
    import os
    
    if not os.path.exists(event_file_path):
        raise FileNotFoundError(f"特殊イベントファイルが見つかりません: {event_file_path}")
    
    # Pythonファイルを動的に実行してeventsを取得
    spec = {}
    try:
        with open(event_file_path, 'r', encoding='utf-8') as f:
            exec(f.read(), spec)
        
        if 'events' not in spec:
            raise ValueError(f"'events'変数が見つかりません: {event_file_path}")
        
        events = spec['events']
        if not isinstance(events, list):
            raise ValueError(f"'events'はリストである必要があります: {event_file_path}")
        
        # イベント形式の検証
        for i, event in enumerate(events):
            if not isinstance(event, tuple) or len(event) != 5:
                raise ValueError(f"イベント{i}の形式が不正です。(name, ref_coord, ref_state, write_coord, write_state)の形式である必要があります")
        
        return events
        
    except Exception as e:
        raise RuntimeError(f"特殊イベントファイルの読み込みに失敗しました: {event_file_path}\nエラー: {str(e)}")

def convert_events_to_array_coordinates(events: List[tuple], min_x: int, min_y: int) -> np.ndarray:
    """
    特殊イベントを配列座標系に変換し、numpy配列として返す
    
    Args:
        events: 特殊イベントのリスト
        min_x, min_y: セル空間の最小座標（オフセット）
    
    Returns:
        変換された特殊イベント配列 (N, 6) 形状
        各行: [ref_x, ref_y, ref_state, write_x, write_y, write_state]
    """
    if not events:
        return np.empty((0, 6), dtype=np.int32)
    
    num_events = len(events)
    event_array = np.zeros((num_events, 6), dtype=np.int32)
    
    for i, event in enumerate(events):
        name, ref_coord, ref_state, write_coord, write_state = event
        
        # 座標を配列インデックスに変換
        ref_x, ref_y = ref_coord
        write_x, write_y = write_coord
        
        event_array[i, 0] = ref_x - min_x    # ref_x (配列座標)
        event_array[i, 1] = ref_y - min_y    # ref_y (配列座標)
        event_array[i, 2] = ref_state        # ref_state
        event_array[i, 3] = write_x - min_x  # write_x (配列座標)
        event_array[i, 4] = write_y - min_y  # write_y (配列座標)
        event_array[i, 5] = write_state      # write_state
    
    return event_array

def get_event_names_from_file(event_file_path: str) -> List[str]:
    """
    特殊イベント定義ファイルからイベント名のリストを取得
    
    Args:
        event_file_path: 特殊イベント定義Pythonファイルのパス
    
    Returns:
        イベント名のリスト
    """
    events = load_special_events_from_file(event_file_path)
    return [event[0] for event in events]  # name部分を抽出

###################################
# セル空間を更新する関数を定義する
# GUIツール、CLIツールどちらもで、同じtorchを使ったセル空間更新関数を呼び出す
# 引数: すべての遷移規則配列(id別適用確率付き), セル空間, 進めるステップ数, 特殊イベントを定義した配列
# 戻り値: 更新後のセル空間, 各特殊イベントの発生した時の更新回数を記録した配列
# 引数は全てtorchテンソルにして、cudaデバイス上で実行できるようにする
###################################
def update_cellspace(
    state: torch.Tensor,                # (T,H,W) or (H,W), dtype=torch.int8 or int16
    rule_arrays: torch.Tensor,          # (N,2,3,3) [prev,next], dtype=torch.int8
    rule_probs: torch.Tensor | None,    # (N,), dtype=torch.float32, Noneなら常に適用可
    seed: int | None = None,
) -> torch.Tensor:
    """
    """
    # 遷移規則を適用する

    return state

# 特殊イベントを適用する
def apply_spatial_events(
    state: torch.Tensor,
    event_array: torch.Tensor
) -> torch.Tensor:
    """
    """
    return state
###################################