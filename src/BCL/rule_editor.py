# src/BCL/rule_editor.py
"""
Rule Editor - 遷移規則のビジュアルエディタ

機能:
- YAML形式の遷移規則ファイル読み込み/保存
- prev→nextパターンのビジュアル編集
- 規則の追加/削除
- 確率値の設定
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
from PySide6 import QtCore, QtGui, QtWidgets


# ========= データ構造 =========

@dataclass
class TransitionRule:
    """遷移規則"""
    rule_id: int
    prev_pattern: np.ndarray  # 前状態パターン
    next_pattern: np.ndarray  # 後状態パターン
    probability: float = 1.0


# ========= パレット・画像変換 =========

def build_palette() -> Dict[int, Tuple[int, int, int]]:
    """状態→色のパレット"""
    return {
        0: (255, 255, 255),   # Vacant: 白
        1: (180, 180, 180),   # Wire: グレー
        2: (0, 0, 0),         # Token: 黒
        -1: (255, 0, 0),      # RecycleBin: 赤
        3: (0, 200, 0),       # Join等: 緑
        4: (200, 200, 0),     # 状態4: 黄
        5: (200, 0, 200),     # 状態5: マゼンタ
        6: (0, 200, 200),     # 状態6: シアン
    }


def array_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """numpy配列をQImageに変換"""
    assert arr.ndim == 2
    palette = build_palette()
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            v = int(arr[y, x])
            r, g, b = palette.get(v, (128, 128, 128))
            rgb[y, x] = (r, g, b)
    
    rgb_contiguous = np.ascontiguousarray(rgb)
    qimg = QtGui.QImage(rgb_contiguous.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return qimg.copy()


# ========= YAML読み込み =========

def load_transition_rules_yaml(path: str) -> Tuple[List[TransitionRule], dict, Dict[int, str]]:
    """
    遷移規則YAMLを読み込み
    
    Returns:
        rules: TransitionRuleリスト
        full_doc: 元のYAMLドキュメント（constants, conv等）
        prob_aliases: rule_id -> 定数名 のマッピング
    """
    # まず生のテキストからAlias参照を抽出
    prob_aliases: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # probability: *alias_name のパターンを抽出
    # 前後のコンテキストからrule_idを特定
    current_rule_id = None
    for line in raw_text.split("\n"):
        # id: NNN を検出
        id_match = re.match(r"^-?\s*id:\s*(\d+)", line)
        if id_match:
            current_rule_id = int(id_match.group(1))
        
        # probability: *alias を検出
        prob_match = re.match(r"^\s*probability:\s*\*(\w+)", line)
        if prob_match and current_rule_id is not None:
            prob_aliases[current_rule_id] = prob_match.group(1)
    
    class _Loader(yaml.SafeLoader):
        pass

    def _occupied_by(loader, node):
        value = loader.construct_scalar(node)
        try:
            return int(value)
        except Exception:
            m = re.search(r"-?\d+", str(value))
            return int(m.group(0)) if m else 0

    _Loader.add_constructor("!OccupiedBy", _occupied_by)

    full_doc = yaml.load(raw_text, Loader=_Loader)

    # ルールリストを抽出
    if isinstance(full_doc, list):
        rules_list = full_doc
    elif isinstance(full_doc, dict):
        rules_list = full_doc.get("rules", [])
    else:
        raise ValueError("Invalid rule file format")

    rules = []
    for item in rules_list:
        if not isinstance(item, dict) or "id" not in item:
            continue
        
        rule_id = item["id"]
        probability = item.get("probability", 1.0)
        rule_data = item.get("rule", {})
        
        prev_list = rule_data.get("prev", [])
        next_list = rule_data.get("next", [])
        
        prev_pattern = _pattern_to_array(prev_list)
        next_pattern = _pattern_to_array(next_list)
        
        rules.append(TransitionRule(
            rule_id=rule_id,
            prev_pattern=prev_pattern,
            next_pattern=next_pattern,
            probability=probability
        ))
    
    return rules, full_doc, prob_aliases


def _pattern_to_array(pattern_list: list) -> np.ndarray:
    """パターンリストをnumpy配列に変換"""
    if not pattern_list:
        return np.zeros((3, 3), dtype=np.int8)
    
    # 座標と状態を抽出
    cells = []
    for item in pattern_list:
        coord = item.get("coord", {})
        x = coord.get("x", 0)
        y = coord.get("y", 0)
        state = item.get("state", 0)
        if isinstance(state, dict):
            state = 0
        cells.append((x, y, int(state)))
    
    if not cells:
        return np.zeros((3, 3), dtype=np.int8)
    
    # 最小外接矩形を計算
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # 3x3以上を確保
    w = max(3, max_x - min_x + 1)
    h = max(3, max_y - min_y + 1)
    
    # 中心を(0,0)として配列を作成
    arr = np.zeros((h, w), dtype=np.int8)
    offset_x = -min_x + (w - (max_x - min_x + 1)) // 2
    offset_y = -min_y + (h - (max_y - min_y + 1)) // 2
    
    for x, y, state in cells:
        ax = x + offset_x
        ay = y + offset_y
        if 0 <= ay < h and 0 <= ax < w:
            arr[ay, ax] = state
    
    return arr


class OccupiedByTag:
    """!OccupiedByタグ用のカスタムクラス"""
    def __init__(self, value: int):
        self.value = value


class ProbabilityAlias:
    """確率値のAlias参照を保持するクラス"""
    def __init__(self, alias_name: str, value: float):
        self.alias_name = alias_name
        self.value = value


def save_transition_rules_yaml(path: str, rules: List[TransitionRule], 
                                constants: dict = None, conv: list = None,
                                prob_aliases: Dict[int, str] = None):
    """
    遷移規則をYAMLに保存
    
    Args:
        path: 保存先パス
        rules: 遷移規則リスト
        constants: 定数定義（Anchors付きで保存）
        conv: 大域状態変換
        prob_aliases: rule_id -> 定数名 のマッピング（確率のAlias参照を復元）
    """
    if prob_aliases is None:
        prob_aliases = {}
    
    # ファイルを手動で構築（Anchors/Aliasesを正確に制御するため）
    lines = []
    
    # constants セクション（Anchors付き）
    if constants:
        lines.append("constants:")
        for key, value in constants.items():
            lines.append(f"  {key}: &{key} {value}")
        lines.append("")
    
    # conv セクション
    if conv:
        lines.append("conv:")
        for item in conv:
            lines.append(f"  - prev: {item['prev']}")
            lines.append(f"    next: {item['next']}")
        lines.append("")
    
    # rules セクション
    lines.append("rules:")
    for rule in rules:
        lines.append(f"- id: {rule.rule_id}")
        
        # probability（Alias参照を復元）
        if rule.probability != 1.0:
            alias_name = prob_aliases.get(rule.rule_id)
            if alias_name and constants and alias_name in constants:
                lines.append(f"  probability: *{alias_name}")
            else:
                lines.append(f"  probability: {rule.probability}")
        
        lines.append("  rule:")
        
        # prev パターン
        lines.append("    prev:")
        for cell in _array_to_pattern_list_raw(rule.prev_pattern):
            lines.append(f"    - coord:")
            lines.append(f"        x: {cell['x']}")
            lines.append(f"        y: {cell['y']}")
            lines.append(f"      state: !OccupiedBy {cell['state']}")
        
        # next パターン
        lines.append("    next:")
        for cell in _array_to_pattern_list_raw(rule.next_pattern):
            lines.append(f"    - coord:")
            lines.append(f"        x: {cell['x']}")
            lines.append(f"        y: {cell['y']}")
            lines.append(f"      state: !OccupiedBy {cell['state']}")
        
        lines.append("")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _array_to_pattern_list_raw(arr: np.ndarray) -> list:
    """numpy配列をパターンリスト（生データ）に変換"""
    h, w = arr.shape
    # 中心を(0,0)とする
    cx, cy = w // 2, h // 2
    
    pattern = []
    for y in range(h):
        for x in range(w):
            v = int(arr[y, x])
            if v != 0:  # 0は省略
                pattern.append({
                    "x": x - cx,
                    "y": y - cy,
                    "state": v
                })
    
    return pattern


def _array_to_pattern_list(arr: np.ndarray) -> list:
    """numpy配列をパターンリストに変換（後方互換性用）"""
    h, w = arr.shape
    cx, cy = w // 2, h // 2
    
    pattern = []
    for y in range(h):
        for x in range(w):
            v = int(arr[y, x])
            if v != 0:
                pattern.append({
                    "coord": {"x": x - cx, "y": y - cy},
                    "state": OccupiedByTag(v)
                })
    
    return pattern


# ========= 編集可能グリッドウィジェット =========

class EditablePatternWidget(QtWidgets.QWidget):
    """編集可能なパターングリッドウィジェット"""
    
    patternChanged = QtCore.Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pattern: np.ndarray = np.zeros((3, 3), dtype=np.int8)
        self._cell_size = 40
        self._brush_value = 1
        self._palette = build_palette()
        
        self.setMinimumSize(150, 150)
        self.setMouseTracking(True)
    
    def set_pattern(self, pattern: np.ndarray):
        """パターンを設定"""
        self._pattern = pattern.copy()
        self.update()
    
    def get_pattern(self) -> np.ndarray:
        """パターンを取得"""
        return self._pattern.copy()
    
    def set_brush_value(self, value: int):
        """ブラシ値を設定"""
        self._brush_value = value
    
    def resize_pattern(self, new_h: int, new_w: int):
        """パターンサイズを変更"""
        old_pattern = self._pattern
        old_h, old_w = old_pattern.shape
        
        new_pattern = np.zeros((new_h, new_w), dtype=np.int8)
        
        # 中央揃えでコピー
        off_y = (new_h - old_h) // 2
        off_x = (new_w - old_w) // 2
        
        for y in range(old_h):
            for x in range(old_w):
                ny = y + off_y
                nx = x + off_x
                if 0 <= ny < new_h and 0 <= nx < new_w:
                    new_pattern[ny, nx] = old_pattern[y, x]
        
        self._pattern = new_pattern
        self.update()
        self.patternChanged.emit()
    
    def paintEvent(self, event):
        """描画"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        h, w = self._pattern.shape
        cell_w = self.width() / w
        cell_h = self.height() / h
        cell_size = min(cell_w, cell_h)
        
        # 中央揃え
        offset_x = (self.width() - cell_size * w) / 2
        offset_y = (self.height() - cell_size * h) / 2
        
        # セル描画
        for y in range(h):
            for x in range(w):
                v = int(self._pattern[y, x])
                r, g, b = self._palette.get(v, (128, 128, 128))
                
                rect = QtCore.QRectF(
                    offset_x + x * cell_size,
                    offset_y + y * cell_size,
                    cell_size, cell_size
                )
                
                painter.fillRect(rect, QtGui.QColor(r, g, b))
                painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 1))
                painter.drawRect(rect)
                
                # 中心セル（0,0）にマーカー
                cx, cy = w // 2, h // 2
                if x == cx and y == cy:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
                    painter.drawRect(rect.adjusted(2, 2, -2, -2))
    
    def mousePressEvent(self, event):
        """マウスクリックでセル編集"""
        if event.button() == QtCore.Qt.LeftButton:
            self._paint_cell(event.position())
        elif event.button() == QtCore.Qt.RightButton:
            self._paint_cell(event.position(), value=0)
    
    def mouseMoveEvent(self, event):
        """ドラッグで連続編集"""
        if event.buttons() & QtCore.Qt.LeftButton:
            self._paint_cell(event.position())
        elif event.buttons() & QtCore.Qt.RightButton:
            self._paint_cell(event.position(), value=0)
    
    def _paint_cell(self, pos: QtCore.QPointF, value: int = None):
        """セルを塗る"""
        h, w = self._pattern.shape
        cell_w = self.width() / w
        cell_h = self.height() / h
        cell_size = min(cell_w, cell_h)
        
        offset_x = (self.width() - cell_size * w) / 2
        offset_y = (self.height() - cell_size * h) / 2
        
        x = int((pos.x() - offset_x) / cell_size)
        y = int((pos.y() - offset_y) / cell_size)
        
        if 0 <= x < w and 0 <= y < h:
            if value is None:
                value = self._brush_value
            if self._pattern[y, x] != value:
                self._pattern[y, x] = value
                self.update()
                self.patternChanged.emit()


# ========= メインエディタウィンドウ =========

class RuleEditorWindow(QtWidgets.QMainWindow):
    """遷移規則エディタのメインウィンドウ"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rule Editor")
        
        # 内部状態
        self._rules: List[TransitionRule] = []
        self._current_index = 0
        self._current_file: Optional[str] = None
        self._modified = False
        
        # 元のYAMLドキュメント（constants, conv等を保持）
        self._original_doc: dict = {}
        
        # 確率のAlias参照マッピング（rule_id -> 定数名）
        self._prob_aliases: Dict[int, str] = {}
        
        # UI構築
        self._build_ui()
        self._build_menu()
        
        self.resize(900, 700)
        
        # 初期状態で空の規則を1つ作成
        self._add_new_rule()
    
    def _build_ui(self):
        """UI構築"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # 規則情報
        info_layout = QtWidgets.QHBoxLayout()
        
        info_layout.addWidget(QtWidgets.QLabel("Rule ID:"))
        self._id_spin = QtWidgets.QSpinBox()
        self._id_spin.setRange(1, 99999)
        self._id_spin.valueChanged.connect(self._on_id_changed)
        info_layout.addWidget(self._id_spin)
        
        info_layout.addSpacing(20)
        
        info_layout.addWidget(QtWidgets.QLabel("Probability:"))
        self._prob_spin = QtWidgets.QDoubleSpinBox()
        self._prob_spin.setRange(0.0, 1.0)
        self._prob_spin.setSingleStep(0.01)
        self._prob_spin.setDecimals(4)
        self._prob_spin.setValue(1.0)
        self._prob_spin.valueChanged.connect(self._on_prob_changed)
        info_layout.addWidget(self._prob_spin)
        
        # 定数選択コンボボックス
        info_layout.addWidget(QtWidgets.QLabel("or Const:"))
        self._const_combo = QtWidgets.QComboBox()
        self._const_combo.setMinimumWidth(150)
        self._const_combo.addItem("(direct value)")  # 直接値を使用
        self._const_combo.currentIndexChanged.connect(self._on_const_selected)
        info_layout.addWidget(self._const_combo)
        
        info_layout.addStretch()
        
        # パターンサイズ
        info_layout.addWidget(QtWidgets.QLabel("Pattern Size:"))
        self._size_combo = QtWidgets.QComboBox()
        self._size_combo.addItems(["3x3", "5x5", "7x7", "9x9"])
        self._size_combo.currentIndexChanged.connect(self._on_size_changed)
        info_layout.addWidget(self._size_combo)
        
        main_layout.addLayout(info_layout)
        
        # prev→next表示エリア
        display_layout = QtWidgets.QHBoxLayout()
        
        # prev表示
        prev_group = QtWidgets.QGroupBox("Previous State (条件)")
        prev_layout = QtWidgets.QVBoxLayout(prev_group)
        self._prev_widget = EditablePatternWidget()
        self._prev_widget.patternChanged.connect(self._on_pattern_changed)
        prev_layout.addWidget(self._prev_widget)
        display_layout.addWidget(prev_group)
        
        # 矢印
        arrow_layout = QtWidgets.QVBoxLayout()
        arrow_layout.addStretch()
        arrow_label = QtWidgets.QLabel("→")
        arrow_label.setAlignment(QtCore.Qt.AlignCenter)
        arrow_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        arrow_layout.addWidget(arrow_label)
        arrow_layout.addStretch()
        display_layout.addLayout(arrow_layout)
        
        # next表示
        next_group = QtWidgets.QGroupBox("Next State (結果)")
        next_layout = QtWidgets.QVBoxLayout(next_group)
        self._next_widget = EditablePatternWidget()
        self._next_widget.patternChanged.connect(self._on_pattern_changed)
        next_layout.addWidget(self._next_widget)
        display_layout.addWidget(next_group)
        
        main_layout.addLayout(display_layout, stretch=1)
        
        # ブラシ選択
        brush_layout = QtWidgets.QHBoxLayout()
        brush_layout.addWidget(QtWidgets.QLabel("Brush:"))
        
        self._brush_group = QtWidgets.QButtonGroup(self)
        # 固定ブラシ（0, 1, 2のみ）
        fixed_brushes = [
            (0, "0: Vacant", (255, 255, 255)),
            (1, "1: Wire", (180, 180, 180)),
            (2, "2: Token", (0, 0, 0)),
        ]
        
        for value, label, color in fixed_brushes:
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(f"background-color: rgb({color[0]},{color[1]},{color[2]}); "
                            f"color: {'white' if sum(color) < 400 else 'black'};")
            btn.clicked.connect(lambda checked, v=value: self._set_brush(v))
            self._brush_group.addButton(btn, value)
            brush_layout.addWidget(btn)
            if value == 1:
                btn.setChecked(True)
        
        # カスタム数値ブラシ
        brush_layout.addWidget(QtWidgets.QLabel(" | Custom:"))
        self._custom_brush_spin = QtWidgets.QSpinBox()
        self._custom_brush_spin.setRange(-128, 127)
        self._custom_brush_spin.setValue(3)
        self._custom_brush_spin.setFixedWidth(60)
        brush_layout.addWidget(self._custom_brush_spin)
        
        self._custom_brush_btn = QtWidgets.QPushButton("Use")
        self._custom_brush_btn.setCheckable(True)
        self._custom_brush_btn.clicked.connect(self._use_custom_brush)
        self._brush_group.addButton(self._custom_brush_btn, 999)  # 特殊ID
        brush_layout.addWidget(self._custom_brush_btn)
        
        brush_layout.addStretch()
        main_layout.addLayout(brush_layout)
        
        # コントロールボタン
        control_layout = QtWidgets.QHBoxLayout()
        
        self._prev_btn = QtWidgets.QPushButton("◀ Previous")
        self._prev_btn.clicked.connect(self._prev_rule)
        control_layout.addWidget(self._prev_btn)
        
        self._rule_selector = QtWidgets.QComboBox()
        self._rule_selector.setMinimumWidth(200)
        self._rule_selector.currentIndexChanged.connect(self._rule_selected)
        control_layout.addWidget(self._rule_selector)
        
        self._next_btn = QtWidgets.QPushButton("Next ▶")
        self._next_btn.clicked.connect(self._next_rule)
        control_layout.addWidget(self._next_btn)
        
        control_layout.addSpacing(30)
        
        self._add_btn = QtWidgets.QPushButton("+ Add Rule")
        self._add_btn.clicked.connect(self._add_new_rule)
        control_layout.addWidget(self._add_btn)
        
        self._del_btn = QtWidgets.QPushButton("× Delete Rule")
        self._del_btn.clicked.connect(self._delete_current_rule)
        control_layout.addWidget(self._del_btn)
        
        main_layout.addLayout(control_layout)
        
        # ステータスバー
        self._status = self.statusBar()
    
    def _build_menu(self):
        """メニュー構築"""
        menubar = self.menuBar()
        
        # File メニュー
        file_menu = menubar.addMenu("File")
        
        act_open = QtGui.QAction("Open YAML...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._action_open)
        file_menu.addAction(act_open)
        
        act_save = QtGui.QAction("Save", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._action_save)
        file_menu.addAction(act_save)
        
        act_save_as = QtGui.QAction("Save As...", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._action_save_as)
        file_menu.addAction(act_save_as)
        
        file_menu.addSeparator()
        
        act_close = QtGui.QAction("Close", self)
        act_close.setShortcut("Ctrl+W")
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)
        
        # Edit メニュー
        edit_menu = menubar.addMenu("Edit")
        
        act_copy_prev = QtGui.QAction("Copy Prev to Next", self)
        act_copy_prev.triggered.connect(self._copy_prev_to_next)
        edit_menu.addAction(act_copy_prev)
        
        act_copy_next = QtGui.QAction("Copy Next to Prev", self)
        act_copy_next.triggered.connect(self._copy_next_to_prev)
        edit_menu.addAction(act_copy_next)
        
        edit_menu.addSeparator()
        
        act_clear_prev = QtGui.QAction("Clear Prev Pattern", self)
        act_clear_prev.triggered.connect(lambda: self._clear_pattern("prev"))
        edit_menu.addAction(act_clear_prev)
        
        act_clear_next = QtGui.QAction("Clear Next Pattern", self)
        act_clear_next.triggered.connect(lambda: self._clear_pattern("next"))
        edit_menu.addAction(act_clear_next)
    
    def _set_brush(self, value: int):
        """ブラシ値を設定"""
        self._prev_widget.set_brush_value(value)
        self._next_widget.set_brush_value(value)
    
    def _use_custom_brush(self):
        """カスタムブラシを使用"""
        value = self._custom_brush_spin.value()
        self._set_brush(value)
        self._custom_brush_btn.setText(f"Use ({value})")
    
    def _update_const_combo(self):
        """定数コンボボックスを更新"""
        self._const_combo.blockSignals(True)
        self._const_combo.clear()
        self._const_combo.addItem("(direct value)", None)
        
        constants = self._original_doc.get("constants", {}) if isinstance(self._original_doc, dict) else {}
        for name, value in constants.items():
            self._const_combo.addItem(f"{name} = {value}", name)
        
        self._const_combo.blockSignals(False)
    
    def _on_const_selected(self, index: int):
        """定数が選択された"""
        if not self._rules or index < 0:
            return
        
        rule = self._rules[self._current_index]
        const_name = self._const_combo.itemData(index)
        
        if const_name is None:
            # 直接値を使用 - alias参照を削除
            if rule.rule_id in self._prob_aliases:
                del self._prob_aliases[rule.rule_id]
        else:
            # 定数を使用
            constants = self._original_doc.get("constants", {})
            if const_name in constants:
                self._prob_spin.blockSignals(True)
                self._prob_spin.setValue(constants[const_name])
                self._prob_spin.blockSignals(False)
                rule.probability = constants[const_name]
                self._prob_aliases[rule.rule_id] = const_name
        
        self._modified = True
        self._update_rule_selector()
    
    def _update_rule_selector(self):
        """規則セレクタを更新"""
        self._rule_selector.blockSignals(True)
        self._rule_selector.clear()
        for i, rule in enumerate(self._rules):
            # 定数参照があれば表示
            alias = self._prob_aliases.get(rule.rule_id)
            if alias:
                prob_text = f" (*{alias})"
            elif rule.probability != 1.0:
                prob_text = f" (p={rule.probability:.3f})"
            else:
                prob_text = ""
            self._rule_selector.addItem(f"Rule {rule.rule_id}{prob_text}")
        if self._rules:
            self._rule_selector.setCurrentIndex(self._current_index)
        self._rule_selector.blockSignals(False)
    
    def _update_display(self):
        """現在の規則を表示"""
        if not self._rules:
            return
        
        rule = self._rules[self._current_index]
        
        # ID、確率
        self._id_spin.blockSignals(True)
        self._id_spin.setValue(rule.rule_id)
        self._id_spin.blockSignals(False)
        
        self._prob_spin.blockSignals(True)
        self._prob_spin.setValue(rule.probability)
        self._prob_spin.blockSignals(False)
        
        # 定数コンボボックス更新
        self._update_const_combo()
        self._const_combo.blockSignals(True)
        alias = self._prob_aliases.get(rule.rule_id)
        if alias:
            # 定数名を検索
            idx = self._const_combo.findData(alias)
            if idx >= 0:
                self._const_combo.setCurrentIndex(idx)
            else:
                self._const_combo.setCurrentIndex(0)
        else:
            self._const_combo.setCurrentIndex(0)
        self._const_combo.blockSignals(False)
        
        # パターン
        self._prev_widget.set_pattern(rule.prev_pattern)
        self._next_widget.set_pattern(rule.next_pattern)
        
        # サイズコンボ
        h, w = rule.prev_pattern.shape
        size_map = {3: 0, 5: 1, 7: 2, 9: 3}
        self._size_combo.blockSignals(True)
        self._size_combo.setCurrentIndex(size_map.get(w, 0))
        self._size_combo.blockSignals(False)
        
        # ボタン状態
        self._prev_btn.setEnabled(self._current_index > 0)
        self._next_btn.setEnabled(self._current_index < len(self._rules) - 1)
        self._del_btn.setEnabled(len(self._rules) > 1)
        
        self._update_rule_selector()
        self._status.showMessage(f"Rule {rule.rule_id} ({self._current_index + 1}/{len(self._rules)})")
    
    def _prev_rule(self):
        if self._current_index > 0:
            self._current_index -= 1
            self._update_display()
    
    def _next_rule(self):
        if self._current_index < len(self._rules) - 1:
            self._current_index += 1
            self._update_display()
    
    def _rule_selected(self, index: int):
        if 0 <= index < len(self._rules):
            self._current_index = index
            self._update_display()
    
    def _add_new_rule(self):
        """新規規則を追加"""
        # 新しいIDを決定
        if self._rules:
            max_id = max(r.rule_id for r in self._rules)
            new_id = max_id + 1
        else:
            new_id = 1
        
        new_rule = TransitionRule(
            rule_id=new_id,
            prev_pattern=np.zeros((3, 3), dtype=np.int8),
            next_pattern=np.zeros((3, 3), dtype=np.int8),
            probability=1.0
        )
        self._rules.append(new_rule)
        self._current_index = len(self._rules) - 1
        self._modified = True
        self._update_display()
    
    def _delete_current_rule(self):
        """現在の規則を削除"""
        if len(self._rules) <= 1:
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Rule",
            f"Delete Rule {self._rules[self._current_index].rule_id}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        del self._rules[self._current_index]
        if self._current_index >= len(self._rules):
            self._current_index = len(self._rules) - 1
        self._modified = True
        self._update_display()
    
    def _on_id_changed(self, value: int):
        if self._rules and 0 <= self._current_index < len(self._rules):
            self._rules[self._current_index].rule_id = value
            self._modified = True
            self._update_rule_selector()
    
    def _on_prob_changed(self, value: float):
        if self._rules and 0 <= self._current_index < len(self._rules):
            rule = self._rules[self._current_index]
            rule.probability = value
            
            # 確率を直接変更したら定数参照を解除
            if rule.rule_id in self._prob_aliases:
                del self._prob_aliases[rule.rule_id]
                self._const_combo.blockSignals(True)
                self._const_combo.setCurrentIndex(0)
                self._const_combo.blockSignals(False)
            
            self._modified = True
            self._update_rule_selector()
    
    def _on_size_changed(self, index: int):
        sizes = [3, 5, 7, 9]
        new_size = sizes[index]
        self._prev_widget.resize_pattern(new_size, new_size)
        self._next_widget.resize_pattern(new_size, new_size)
        self._on_pattern_changed()
    
    def _on_pattern_changed(self):
        if self._rules and 0 <= self._current_index < len(self._rules):
            self._rules[self._current_index].prev_pattern = self._prev_widget.get_pattern()
            self._rules[self._current_index].next_pattern = self._next_widget.get_pattern()
            self._modified = True
    
    def _copy_prev_to_next(self):
        self._next_widget.set_pattern(self._prev_widget.get_pattern())
        self._on_pattern_changed()
    
    def _copy_next_to_prev(self):
        self._prev_widget.set_pattern(self._next_widget.get_pattern())
        self._on_pattern_changed()
    
    def _clear_pattern(self, which: str):
        if which == "prev":
            h, w = self._prev_widget.get_pattern().shape
            self._prev_widget.set_pattern(np.zeros((h, w), dtype=np.int8))
        else:
            h, w = self._next_widget.get_pattern().shape
            self._next_widget.set_pattern(np.zeros((h, w), dtype=np.int8))
        self._on_pattern_changed()
    
    def _action_open(self):
        """YAMLファイルを開く"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Rule YAML", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if not path:
            return
        
        try:
            self._rules, self._original_doc, self._prob_aliases = load_transition_rules_yaml(path)
            self._current_file = path
            self._current_index = 0
            self._modified = False
            self._update_display()
            self.setWindowTitle(f"Rule Editor - {Path(path).name}")
            alias_count = len(self._prob_aliases)
            self._status.showMessage(f"Loaded {len(self._rules)} rules from {path} ({alias_count} probability aliases)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")
    
    def _action_save(self):
        """保存"""
        if self._current_file:
            self._save_to_file(self._current_file)
        else:
            self._action_save_as()
    
    def _action_save_as(self):
        """名前を付けて保存"""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Rule YAML", "", "YAML Files (*.yaml);;All Files (*)")
        if not path:
            return
        if not path.endswith(".yaml"):
            path += ".yaml"
        self._save_to_file(path)
    
    def _save_to_file(self, path: str):
        """ファイルに保存"""
        try:
            constants = self._original_doc.get("constants") if isinstance(self._original_doc, dict) else None
            conv = self._original_doc.get("conv") if isinstance(self._original_doc, dict) else None
            save_transition_rules_yaml(path, self._rules, constants, conv, self._prob_aliases)
            self._current_file = path
            self._modified = False
            self.setWindowTitle(f"Rule Editor - {Path(path).name}")
            self._status.showMessage(f"Saved to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
    
    def closeEvent(self, event):
        """終了時の確認"""
        if self._modified:
            reply = QtWidgets.QMessageBox.question(
                self, "Unsaved Changes",
                "There are unsaved changes. Save before closing?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
            )
            if reply == QtWidgets.QMessageBox.Save:
                self._action_save()
                event.accept()
            elif reply == QtWidgets.QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ========= エントリポイント =========

def main(argv=None):
    app = QtWidgets.QApplication(argv or sys.argv)
    win = RuleEditorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
