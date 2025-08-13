import sys
import os
import numpy as np
from typing import List, Dict, Any

# 直接実行時は絶対インポート、パッケージ実行時は相対インポート
try:
    # パッケージとして実行される場合
    from . import (
        load_cell_space_yaml_to_numpy, 
        load_transition_rules_yaml,
        load_multiple_transition_rules_to_numpy,
        load_multiple_transition_rules_with_probability,
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
        load_transition_rules_yaml,
        load_multiple_transition_rules_to_numpy,
        load_multiple_transition_rules_with_probability,
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
def update_cellspace(
    rules_with_prob: tuple[np.ndarray, np.ndarray],  # (rule_array, probability_array) from load_multiple_transition_rules_with_probability
    cellspace: np.ndarray,            # セル空間配列 (H, W) int8
    step_count: int,                  # 進めるステップ数
    event_array: np.ndarray = None,   # 特殊イベント配列 (M, 6) int32, optional
    global_prob: float = 0.5,         # グローバル確率 (0.0-1.0)
    seed: int = None,                 # 乱数シード (再現性用)
    device: str = "cuda"              # 計算デバイス ("cuda" or "cpu")
) -> tuple[np.ndarray, dict]:
    """
    ブラウン運動セルオートマトンのステップ更新
    
    Args:
        rules_with_prob: load_multiple_transition_rules_with_probability()の戻り値
                        (rule_array, probability_array)のタプル
                        - rule_array: 遷移規則パターン配列 (N, 2, 3, 3)
                        - probability_array: 各規則の適用確率 (N,)
        cellspace: 現在のセル空間状態 (H, W)
        step_count: 実行するステップ数
        event_array: 特殊イベント定義 (M, 6) [ref_x, ref_y, ref_state, write_x, write_y, write_state]
        global_prob: ステップ全体の実行確率 (デフォルト0.5)
        seed: 乱数シード (Noneの場合は非決定的)
        device: 計算デバイス
    
    Returns:
        tuple: (updated_cellspace, event_log)
        - updated_cellspace: 更新後のセル空間 (H, W) int8
        - event_log: イベント発生記録（プロット用に最適化）
          {
              "event_step_lists": {
                  "event_name_1": [50, 100, 110, 205, ...],  # イベント発生ステップ番号のリスト
                  "event_name_2": [75, 150, 180, ...],
                  ...
              },
              "rule_application_counts": {
                  rule_id: total_count,  # 各規則の総適用回数
                  ...
              },
              "simulation_stats": {
                  "total_steps": int,           # 総ステップ数
                  "skipped_steps": int,         # グローバル確率でスキップされたステップ数
                  "active_steps": int,          # 実際に規則適用を試行したステップ数
                  "total_rule_applications": int # 全規則の総適用回数
              }
          }
    """
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

def test_probabilistic_rules():
    """
    確率的遷移規則の読み込みテスト
    """
    print("=== 確率的遷移規則テスト ===")
    
    # テスト用ファイルパス
    rule_files = ["rule/base-rule.yaml", "rule/C-Join_err-rule.yaml"]
    
    print(f"テストファイル: {rule_files}")
    print()
    
    # === 1. 従来の読み込み方法（確率情報なし） ===
    print("1. 従来の読み込み方法（確率情報なし）")
    rule_array = load_multiple_transition_rules_to_numpy(rule_files)
    rule_ids = get_rule_ids_from_files(rule_files)
    
    print(f"  規則配列形状: {rule_array.shape}")
    print(f"  規則数: {len(rule_ids)}")
    print()
    
    # === 2. 新しい読み込み方法（確率情報あり） ===
    print("2. 新しい読み込み方法（確率情報あり）")
    rule_array_prob, probability_array = load_multiple_transition_rules_with_probability(rule_files)
    
    print(f"  規則配列形状: {rule_array_prob.shape}")
    print(f"  確率配列形状: {probability_array.shape}")
    print(f"  確率配列dtype: {probability_array.dtype}")
    print()
    
    # === 3. 配列の一致確認 ===
    print("3. 配列の一致確認")
    arrays_match = np.array_equal(rule_array, rule_array_prob)
    print(f"  規則配列一致: {arrays_match}")
    print()
    
    # === 4. 確率情報の詳細表示 ===
    print("4. 確率情報の詳細表示")
    
    # TransitionRuleオブジェクトで詳細確認
    all_rules = []
    for rule_path in rule_files:
        rules = load_transition_rules_yaml(rule_path)
        all_rules.extend(rules)
    
    print(f"  読み込み済み規則数: {len(all_rules)}")
    print()
    
    # 確率が1.0でない規則を表示
    probabilistic_rules = [rule for rule in all_rules if rule.probability < 1.0]
    deterministic_rules = [rule for rule in all_rules if rule.probability == 1.0]
    
    print(f"  確定的規則数: {len(deterministic_rules)} (probability=1.0)")
    print(f"  確率的規則数: {len(probabilistic_rules)} (probability<1.0)")
    print()
    
    if probabilistic_rules:
        print("  確率的規則の詳細:")
        for rule in probabilistic_rules:
            print(f"    ID {rule.rule_id}: probability={rule.probability}")
            print(f"      前状態パターン形状: {rule.prev_pattern.shape}")
            print(f"      後状態パターン形状: {rule.next_pattern.shape}")
        print()
    
    # === 5. 確率配列の統計 ===
    print("5. 確率配列の統計")
    print(f"  最小確率: {probability_array.min():.3f}")
    print(f"  最大確率: {probability_array.max():.3f}")
    print(f"  平均確率: {probability_array.mean():.3f}")
    print(f"  確率1.0の規則数: {np.sum(probability_array == 1.0)}")
    print(f"  確率1.0未満の規則数: {np.sum(probability_array < 1.0)}")
    print()
    
    return len(probabilistic_rules) > 0

def test_data_consistency():
    """
    GUIツールとlib.py関数で生成されるデータの整合性をテスト
    """
    print("=== データ整合性テスト ===")
    
    # テスト用ファイルパス
    cellspace_file = "../SampleCP/test.yaml"
    rule_files = ["rule/base-rule.yaml", "rule/C-Join_err-rule.yaml"]
    event_file = "../SampleCP/BCA-IP_event.py"
    
    print(f"テストファイル:")
    print(f"  セル空間: {cellspace_file}")
    print(f"  遷移規則: {rule_files}")
    print(f"  特殊イベント: {event_file}")
    print()
    
    # === 1. セル空間データの整合性テスト ===
    print("1. セル空間データ整合性テスト")
    
    # lib.py関数で直接読み込み
    arr_with_offset = load_cell_space_yaml_to_numpy(cellspace_file)
    cellspace_lib, min_x_lib, min_y_lib = extract_cellspace_and_offset(arr_with_offset)
    
    # GUIツール相当の処理（set_arrayメソッドと同等）
    if has_offset_info(arr_with_offset):
        cellspace_gui, min_x_gui, min_y_gui = extract_cellspace_and_offset(arr_with_offset)
    else:
        cellspace_gui = arr_with_offset
        min_x_gui = min_y_gui = 0
    
    # 比較
    cellspace_match = np.array_equal(cellspace_lib, cellspace_gui)
    offset_match = (min_x_lib == min_x_gui) and (min_y_lib == min_y_gui)
    
    print(f"  セル空間配列一致: {cellspace_match}")
    print(f"  オフセット一致: {offset_match} (lib: ({min_x_lib},{min_y_lib}), gui: ({min_x_gui},{min_y_gui}))")
    print(f"  配列形状: {cellspace_lib.shape}, dtype: {cellspace_lib.dtype}")
    print()
    
    # === 2. 遷移規則データの整合性テスト ===
    print("2. 遷移規則データ整合性テスト")
    
    # lib.py関数で直接読み込み
    rule_array_lib = load_multiple_transition_rules_to_numpy(rule_files)
    rule_ids_lib = get_rule_ids_from_files(rule_files)
    
    # GUIツール相当の処理（_action_add_rule_fileメソッドと同等）
    loaded_rule_files_gui = []
    loaded_rule_array_gui = None
    loaded_rule_ids_gui = []
    
    for rule_file in rule_files:
        if rule_file not in loaded_rule_files_gui:
            loaded_rule_files_gui.append(rule_file)
    
    # 統合配列を作成
    if loaded_rule_files_gui:
        loaded_rule_array_gui = load_multiple_transition_rules_to_numpy(loaded_rule_files_gui)
        loaded_rule_ids_gui = get_rule_ids_from_files(loaded_rule_files_gui)
    
    # 比較
    rule_array_match = np.array_equal(rule_array_lib, loaded_rule_array_gui)
    rule_ids_match = rule_ids_lib == loaded_rule_ids_gui
    
    print(f"  遷移規則配列一致: {rule_array_match}")
    print(f"  規則ID一致: {rule_ids_match}")
    print(f"  配列形状: {rule_array_lib.shape}, dtype: {rule_array_lib.dtype}")
    print(f"  規則数: {len(rule_ids_lib)}")
    print()
    
    # === 3. 特殊イベントデータの整合性テスト ===
    print("3. 特殊イベントデータ整合性テスト")
    
    # lib.py関数で直接読み込み
    events_lib = load_special_events_from_file(event_file)
    event_names_lib = get_event_names_from_file(event_file)
    event_array_lib = convert_events_to_array_coordinates(events_lib, min_x_lib, min_y_lib)
    
    # GUIツール相当の処理（_action_load_special_eventsメソッドと同等）
    events_gui = load_special_events_from_file(event_file)
    event_names_gui = get_event_names_from_file(event_file)
    event_array_gui = convert_events_to_array_coordinates(events_gui, min_x_gui, min_y_gui)
    
    # 比較
    events_match = events_lib == events_gui
    event_names_match = event_names_lib == event_names_gui
    event_array_match = np.array_equal(event_array_lib, event_array_gui)
    
    print(f"  特殊イベントリスト一致: {events_match}")
    print(f"  イベント名一致: {event_names_match}")
    print(f"  イベント配列一致: {event_array_match}")
    print(f"  配列形状: {event_array_lib.shape}, dtype: {event_array_lib.dtype}")
    print(f"  イベント数: {len(events_lib)}")
    print()
    
    # === 4. 詳細比較（最初の数個のイベント） ===
    print("4. 特殊イベント詳細比較（最初の3個）")
    for i in range(min(3, len(events_lib))):
        event_lib = events_lib[i]
        event_gui = events_gui[i]
        array_row_lib = event_array_lib[i]
        array_row_gui = event_array_gui[i]
        
        print(f"  イベント{i+1}: {event_lib[0]}")
        print(f"    lib: {event_lib}")
        print(f"    gui: {event_gui}")
        print(f"    配列(lib): {array_row_lib}")
        print(f"    配列(gui): {array_row_gui}")
        print(f"    一致: {np.array_equal(array_row_lib, array_row_gui)}")
        print()
    
    # === 5. 総合結果 ===
    print("=== 総合結果 ===")
    all_match = cellspace_match and offset_match and rule_array_match and rule_ids_match and events_match and event_names_match and event_array_match
    
    print(f"全データ整合性: {'✅ 完全一致' if all_match else '❌ 不一致あり'}")
    
    if not all_match:
        print("不一致項目:")
        if not cellspace_match: print("  - セル空間配列")
        if not offset_match: print("  - オフセット")
        if not rule_array_match: print("  - 遷移規則配列")
        if not rule_ids_match: print("  - 規則ID")
        if not events_match: print("  - 特殊イベントリスト")
        if not event_names_match: print("  - イベント名")
        if not event_array_match: print("  - イベント配列")
    
    print()
    return all_match

if __name__ == "__main__":
    # main()
    # test_data_consistency()
    test_probabilistic_rules()