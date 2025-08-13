# GUI, CLIツール共通のライブラリを作成

import yaml
import numpy as np
from typing import List, Dict, Any

def load_cell_space_yaml_to_numpy(path: str) -> np.ndarray:
    """
    セル空間YAML（形式: [{'coord':{'x':..,'y':..}, 'value':..}, ...]）を
    最小外接矩形で切り出した 2D NumPy 配列に変換して返す。
    - 未指定セルは 0 で初期化
    - 負の座標は配列内で 0 起点となるよう平行移動
    - 値はそのまま格納（例: -1, 1, 2 など）

    Args:
        path: セル空間YAMLファイルのパス

    Returns:
        np.ndarray[int8] 形状 (H, W)
    """
    with open(path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = yaml.safe_load(f)

    if not isinstance(items, list) or not items:
        raise ValueError("YAMLのトップは非空のリストである必要があります")

    xs, ys, vs = [], [], []
    for it in items:
        c = it.get("coord", {})
        xs.append(int(c["x"]))
        ys.append(int(c["y"]))
        vs.append(int(it.get("value", 0)))

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = max_x - min_x + 1
    H = max_y - min_y + 1

    # YAML座標→配列添字の平行移動
    off_x = -min_x
    off_y = -min_y

    arr = np.zeros((H, W), dtype=np.int8)
    xs_a = np.asarray(xs) + off_x
    ys_a = np.asarray(ys) + off_y
    vs_a = np.asarray(vs, dtype=np.int8)
    arr[ys_a, xs_a] = vs_a
    return arr
