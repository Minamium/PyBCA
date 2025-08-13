import sys
import os
import numpy as np
from typing import List, Dict, Any

# 直接実行時は絶対インポート、パッケージ実行時は相対インポート
try:
    # パッケージとして実行される場合
    from . import (
        load_cell_space_yaml_to_numpy, 
        load_multiple_transition_rules_to_numpy,
        get_rule_ids_from_files,
        extract_cellspace_and_offset, 
        has_offset_info,
        load_special_events_from_file,
        convert_events_to_array_coordinates,
        get_event_names_from_file
    )
except ImportError:
    # 直接実行される場合
    from lib import (
        load_cell_space_yaml_to_numpy, 
        load_multiple_transition_rules_to_numpy,
        get_rule_ids_from_files,
        extract_cellspace_and_offset, 
        has_offset_info,
        load_special_events_from_file,
        convert_events_to_array_coordinates,
        get_event_names_from_file
    )

###################################
# セル空間を任意回数更新する関数を定義する
# GUIツール、CLIツールどちらもで、同じtorchを使ったセル空間更新関数を呼び出す
# 引数: すべての遷移規則配列(id別適用確率付き), セル空間, 進めるステップ数, 特殊イベントを定義した配列
# 戻り値: 更新後のセル空間, 各特殊イベントの発生した時の更新回数を記録した配列
###################################
def update_cellspace():
    pass

###################################



# デバッグ用関数
def format_pattern_matrix(pattern: np.ndarray) -> str:
    """3x3パターンを見やすいマトリックス形式で表示"""
    lines = []
    lines.append("{[" + ",".join(map(str, pattern[0])) + "]")
    lines.append(" [" + ",".join(map(str, pattern[1])) + "]")
    lines.append(" [" + ",".join(map(str, pattern[2])) + "]}")
    return lines

def test_load_cellspace_and_rules(rule_paths: List[str], cellspace_path: str):
    """セル空間と遷移規則の読み込みテスト"""
    print("=== PyBCA Debug: セル空間・遷移規則読み込みテスト ===")
    
    # セル空間ファイルの読み込み
    print(f"\n1. セル空間読み込み: {cellspace_path}")
    try:
        if os.path.exists(cellspace_path):
            arr_with_offset = load_cell_space_yaml_to_numpy(cellspace_path, include_offset=True)
            print(f"   読み込み成功: 形状={arr_with_offset.shape}, dtype={arr_with_offset.dtype}")
            
            if has_offset_info(arr_with_offset):
                cellspace, min_x, min_y = extract_cellspace_and_offset(arr_with_offset)
                print(f"   セル空間: 形状={cellspace.shape}, dtype={cellspace.dtype}")
                print(f"   オフセット: min_x={min_x}, min_y={min_y}")
                print(f"   非零セル数: {np.count_nonzero(cellspace)}")
                
                unique, counts = np.unique(cellspace, return_counts=True)
                print(f"   セル状態分布:")
                for state, count in zip(unique, counts):
                    print(f"      状態{state}: {count}個")
            else:
                print(f"   警告: オフセット情報なし")
        else:
            print(f"   エラー: ファイルが見つかりません: {cellspace_path}")
    except Exception as e:
        print(f"   エラー: {str(e)}")

    # 新しい統合関数を使用して遷移規則をnumpy配列として読み込む
    print(f"\n2. 遷移規則読み込み: {len(rule_paths)}個のファイル")
    try:
        rule_array = load_multiple_transition_rules_to_numpy(rule_paths)
        rule_ids = get_rule_ids_from_files(rule_paths)
        print(f"   読み込み成功: 形状={rule_array.shape}, dtype={rule_array.dtype}")
        print(f"   規則数: {len(rule_ids)}個")
        
        for rule_path in rule_paths:
            print(f"   ファイル: {rule_path}")
            
    except Exception as e:
        print(f"   エラー: {str(e)}")
        return None, None

    # 読み込んだ遷移規則を標準出力に表示する
    print(f"\n3. 読み込み済み遷移規則一覧: 合計{rule_array.shape[0]}個")
    for i in range(rule_array.shape[0]):
        print(f"\nid: {rule_ids[i]}")
        prev_pattern = rule_array[i, 0]  # 前状態パターン
        next_pattern = rule_array[i, 1]  # 後状態パターン
        
        prev_lines = format_pattern_matrix(prev_pattern)
        next_lines = format_pattern_matrix(next_pattern)
        
        # 左右並列表示
        for j in range(3):
            if j == 1:  # 中央行に矢印を表示
                print(f"{prev_lines[j]}            →            {next_lines[j]}")
            else:
                print(f"{prev_lines[j]}                         {next_lines[j]}")
    
    # 特殊イベント読み込みテスト
    print(f"\n4. 特殊イベント読み込みテスト")
    event_file_path = "../SampleCP/BCA-IP_event.py"
    try:
        events = load_special_events_from_file(event_file_path)
        event_names = get_event_names_from_file(event_file_path)
        
        print(f"   読み込み成功: {len(events)}個のイベント")
        print(f"   ファイル: {event_file_path}")
        
        # セル空間のオフセット情報を使用してイベント座標を変換
        if 'cellspace' in locals() and 'min_x' in locals() and 'min_y' in locals():
            event_array = convert_events_to_array_coordinates(events, min_x, min_y)
            print(f"   変換後配列形状: {event_array.shape}")
            print(f"   配列座標系に変換完了")
            
            # 最初の3個のイベントを表示
            for i in range(min(3, len(events))):
                name = event_names[i]
                orig_event = events[i]
                array_coords = event_array[i]
                
                print(f"   イベント{i+1}: {name}")
                print(f"     元座標: 参照{orig_event[1]}(状態{orig_event[2]}) → 書込{orig_event[3]}(状態{orig_event[4]})")
                print(f"     配列座標: 参照({array_coords[0]},{array_coords[1]})(状態{array_coords[2]}) → 書込({array_coords[3]},{array_coords[4]})(状態{array_coords[5]})")
            
            if len(events) > 3:
                print(f"   ... (他{len(events)-3}個)")
        else:
            print(f"   警告: セル空間オフセット情報がないため座標変換をスキップ")
            
    except Exception as e:
        print(f"   エラー: {str(e)}")
    
    print(f"\n=== テスト完了 ===")
    return cellspace if 'cellspace' in locals() else None, rule_array

def main(argv=None):
    """デバッグ用メイン関数"""
    if argv is None:
        argv = sys.argv
    
    print("PyBCA CUDA Simulator Debug Mode")
    print("現在のディレクトリ:", os.getcwd())
    
    # テスト用パス
    cellspace_path = "../SampleCP/test.yaml"
    rule_paths = [
        "rule/base-rule.yaml",
        "rule/C-Join_err-rule.yaml"
    ]
    
    # セル空間と遷移規則の読み込みテスト
    cellspace, rules = test_load_cellspace_and_rules(rule_paths, cellspace_path)
    
    if cellspace is not None and rules is not None:
        print(f"\n読み込み完了: セル空間{cellspace.shape}, 規則配列{rules.shape}")
    else:
        print(f"\n読み込み失敗")

if __name__ == "__main__":
    main()