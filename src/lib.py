# GUI, CLIツール共通のライブラリを作成
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import yaml


# yamlセル空間ファイルをnumpy配列に変換する
def load_cell_space_yaml_to_numpy(path: str, progress_callback=None) -> np.ndarray:
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
    
    arr = np.zeros((H, W), dtype=np.int8)
    
    if progress_callback:
        progress_callback(93, 100)  # 配列初期化完了
    
    xs_a = np.asarray(xs) + off_x
    ys_a = np.asarray(ys) + off_y
    vs_a = np.asarray(vs, dtype=np.int8)
    
    if progress_callback:
        progress_callback(97, 100)  # 配列代入開始
    
    arr[ys_a, xs_a] = vs_a
    
    if progress_callback:
        progress_callback(100, 100)  # 完了
    
    return arr

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
        
        rules.append(TransitionRule(
            rule_id=rule_id,
            prev_pattern=prev_arr,
            next_pattern=next_arr
        ))

    return rules
