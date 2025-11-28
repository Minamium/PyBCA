# src/BCL/editor.py
"""
BCL Editor - BCL形式セル空間のビジュアルエディタ

機能:
- BCLファイル読み込み/保存
- ビジュアルキャンバスでのセル編集
- BCLソースのリアルタイム表示
- YAMLエクスポート
"""
from __future__ import annotations

import sys
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .compiler import BCLCompiler, CompileResult
from .rule_editor import RuleEditorWindow


# ========= 定数・列挙型 =========

class EditTool(Enum):
    """編集ツール種別"""
    POINT = auto()
    RECT = auto()
    LINE = auto()


@dataclass
class EditAction:
    """Undo/Redo用の編集アクション"""
    before: np.ndarray
    after: np.ndarray
    description: str = ""


@dataclass
class ElementDefinition:
    """BCL element定義"""
    name: str
    param: str
    body: List[str] = field(default_factory=list)


# ========= パレット・画像変換 =========

def build_palette(states: List[int]) -> Dict[int, Tuple[int, int, int]]:
    """状態→色の簡易パレット"""
    pal: Dict[int, Tuple[int, int, int]] = {
        0: (255, 255, 255),   # Vacant: 白
        1: (180, 180, 180),   # Wire: グレー
        2: (0, 0, 0),         # Token: 黒
        -1: (255, 0, 0),      # RecycleBin: 赤
        3: (0, 200, 0),       # Join等: 緑
        4: (200, 200, 0),     # 状態4: 黄
        5: (200, 0, 200),     # 状態5: マゼンタ
        6: (0, 200, 200),     # 状態6: シアン
    }
    for s in states:
        if s not in pal:
            pal[s] = (random.randrange(40, 256),
                      random.randrange(40, 256),
                      random.randrange(40, 256))
    return pal


def array_to_qimage(arr: np.ndarray,
                    palette: Optional[Dict[int, Tuple[int, int, int]]] = None
                    ) -> QtGui.QImage:
    """numpy配列をQImageに変換（LUT使用）"""
    assert arr.ndim == 2
    vals = np.unique(arr)
    val2idx = {int(v): i for i, v in enumerate(vals.tolist())}
    idx_map = np.vectorize(lambda v: val2idx[int(v)], otypes=[np.int32])(arr)

    if palette is None:
        palette = build_palette([int(v) for v in vals])

    lut = np.zeros((len(vals), 3), dtype=np.uint8)
    for v, i in val2idx.items():
        r, g, b = palette.get(v, (128, 128, 128))
        lut[i] = (r, g, b)

    rgb = lut[idx_map]  # HxWx3
    h, w = rgb.shape[:2]
    rgb_contiguous = np.ascontiguousarray(rgb)
    qimg = QtGui.QImage(rgb_contiguous.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return qimg.copy()


# ========= カスタムウィジェット =========

class ElementListWidget(QtWidgets.QListWidget):
    """element定義をドラッグ可能なリストウィジェット"""
    
    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        
        elem = item.data(QtCore.Qt.UserRole)
        if elem is None:
            return
        
        # MIMEデータにelement名を設定
        mime_data = QtCore.QMimeData()
        mime_data.setText(elem.name)
        
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(QtCore.Qt.CopyAction)


class CanvasView(QtWidgets.QGraphicsView):
    """ズーム/パン対応のキャンバスビュー（右クリックでドラッグ移動）"""
    zoomChanged = QtCore.Signal(float)
    elementDropped = QtCore.Signal(str, int, int)  # element_name, x, y
    elementDragMove = QtCore.Signal(str, int, int)  # element_name, x, y (プレビュー用)
    elementDragLeave = QtCore.Signal()  # ドラッグ終了/離脱時

    def __init__(self, scene: QtWidgets.QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setAcceptDrops(True)
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QtCore.QPoint()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasText():
            text = event.mimeData().text()
            pos = self.mapToScene(event.position().toPoint())
            x, y = int(pos.x()), int(pos.y())
            self.elementDragMove.emit(text, x, y)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        self.elementDragLeave.emit()
        event.accept()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasText():
            text = event.mimeData().text()
            pos = self.mapToScene(event.position().toPoint())
            x, y = int(pos.x()), int(pos.y())
            self.elementDragLeave.emit()  # プレビューをクリア
            self.elementDropped.emit(text, x, y)
            event.acceptProposedAction()
        else:
            event.ignore()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 1 / 1.25
        self._zoom *= factor
        self.scale(factor, factor)
        self.zoomChanged.emit(self._zoom)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ========= メインエディタウィンドウ =========

class BCLEditorWindow(QtWidgets.QMainWindow):
    """BCLエディタのメインウィンドウ"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCL Editor")
        
        # 内部状態
        self._arr: np.ndarray = np.zeros((50, 50), dtype=np.int8)
        self._current_file: Optional[str] = None
        self._modified = False
        self._brush_value = 1
        
        # ツール状態
        self._current_tool = EditTool.POINT
        self._drag_start: Optional[Tuple[int, int]] = None
        self._is_dragging = False
        
        # Undo/Redo
        self._undo_stack: List[EditAction] = []
        self._redo_stack: List[EditAction] = []
        self._max_undo = 50
        
        # element定義（現在のファイルから）
        self._elements: List[ElementDefinition] = []
        # ライブラリelement定義（lib.bclから）
        self._library_elements: List[ElementDefinition] = []
        
        # BCLヘッダー（element定義、coord.defineなど配置以外の部分）
        self._bcl_header: str = ""
        
        # 元のBCLソース（ファイルから読み込んだそのまま）
        self._raw_bcl_source: str = ""
        
        # element配置履歴: [(element_name, x, y, instance_id), ...]
        self._element_placements: List[Tuple[str, int, int, int]] = []
        self._next_instance_id = 1
        
        # プレビュー用グラフィックアイテム
        self._preview_rect: Optional[QtWidgets.QGraphicsRectItem] = None
        self._preview_line: Optional[QtWidgets.QGraphicsLineItem] = None
        
        # 選択範囲（矩形選択ツール用）: (x1, y1, x2, y2)
        self._selection_rect: Optional[Tuple[int, int, int, int]] = None
        self._selection_item: Optional[QtWidgets.QGraphicsRectItem] = None
        
        # アンカーポイント（element作成用基準点）
        self._anchor_point: Optional[Tuple[int, int]] = None
        self._anchor_item: Optional[QtWidgets.QGraphicsEllipseItem] = None
        
        # element作成モード
        self._element_creation_mode = False
        
        # ソース編集中フラグ
        self._source_updating = False
        
        # Rule Editorウィンドウ参照
        self._rule_editor_window: Optional[RuleEditorWindow] = None
        
        # UI構築
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        
        # ライブラリ読み込み
        self._load_library()
        
        # 初期表示
        self._update_canvas()
        self._update_bcl_source()
        
        self.resize(1400, 900)

    def _build_ui(self):
        """UI構築"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # 左側: キャンバス
        canvas_group = QtWidgets.QGroupBox("Canvas")
        canvas_layout = QtWidgets.QVBoxLayout(canvas_group)
        
        self._scene = QtWidgets.QGraphicsScene(self)
        self._pix_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        
        self._view = CanvasView(self._scene, self)
        self._view.setMouseTracking(True)
        self._view.viewport().setMouseTracking(True)
        self._view.viewport().installEventFilter(self)
        canvas_layout.addWidget(self._view)
        
        # グリッド
        self._grid_item = QtWidgets.QGraphicsPathItem()
        pen = QtGui.QPen(QtGui.QColor(80, 80, 80, 100), 0)
        pen.setCosmetic(True)
        self._grid_item.setPen(pen)
        self._scene.addItem(self._grid_item)
        
        main_layout.addWidget(canvas_group, stretch=2)
        
        # 右側パネル
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # BCLソース表示（読み取り専用）
        source_group = QtWidgets.QGroupBox("BCL Source (Read-only)")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        
        self._source_edit = QtWidgets.QPlainTextEdit()
        self._source_edit.setFont(QtGui.QFont("Menlo", 10))
        self._source_edit.setReadOnly(True)  # 読み取り専用
        source_layout.addWidget(self._source_edit)
        
        right_layout.addWidget(source_group, stretch=2)
        
        # Element定義表示
        element_group = QtWidgets.QGroupBox("Element Definitions (Drag to Canvas)")
        element_layout = QtWidgets.QVBoxLayout(element_group)
        
        self._element_list = ElementListWidget()
        self._element_list.setDragEnabled(True)
        self._element_list.itemDoubleClicked.connect(self._on_element_double_clicked)
        element_layout.addWidget(self._element_list)
        
        # New Elementボタン
        self._new_element_btn = QtWidgets.QPushButton("New Element...")
        self._new_element_btn.clicked.connect(self._start_element_creation)
        element_layout.addWidget(self._new_element_btn)
        
        # 使い方ラベル
        hint_label = QtWidgets.QLabel("Drag element to canvas to place")
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        element_layout.addWidget(hint_label)
        
        right_layout.addWidget(element_group, stretch=1)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        # ステータスバー
        self._status = self.statusBar()
        self._tool_label = QtWidgets.QLabel("Tool: Point")
        self._zoom_label = QtWidgets.QLabel("Zoom: 100%")
        self._status.addPermanentWidget(self._tool_label)
        self._status.addPermanentWidget(self._zoom_label)
        
        # シグナル接続
        self._view.zoomChanged.connect(self._on_zoom_changed)
        self._view.elementDropped.connect(self._on_element_dropped)
        self._view.elementDragMove.connect(self._on_element_drag_move)
        self._view.elementDragLeave.connect(self._on_element_drag_leave)
        
        # elementプレビュー用グラフィックアイテム
        self._element_preview_items: List[QtWidgets.QGraphicsRectItem] = []

    def _build_menu(self):
        """メニュー構築"""
        menubar = self.menuBar()
        
        # File メニュー
        file_menu = menubar.addMenu("File")
        
        act_new = QtGui.QAction("New", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._action_new)
        file_menu.addAction(act_new)
        
        act_open = QtGui.QAction("Open BCL...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._action_open)
        file_menu.addAction(act_open)
        
        act_save = QtGui.QAction("Save BCL", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._action_save)
        file_menu.addAction(act_save)
        
        act_save_as = QtGui.QAction("Save BCL As...", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._action_save_as)
        file_menu.addAction(act_save_as)
        
        file_menu.addSeparator()
        
        act_export = QtGui.QAction("Export YAML...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._action_export_yaml)
        file_menu.addAction(act_export)
        
        file_menu.addSeparator()
        
        act_quit = QtGui.QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
        
        # Edit メニュー
        edit_menu = menubar.addMenu("Edit")
        
        self._act_undo = QtGui.QAction("Undo", self)
        self._act_undo.setShortcut("Ctrl+Z")
        self._act_undo.triggered.connect(self._action_undo)
        self._act_undo.setEnabled(False)
        edit_menu.addAction(self._act_undo)
        
        self._act_redo = QtGui.QAction("Redo", self)
        self._act_redo.setShortcut("Ctrl+Shift+Z")
        self._act_redo.triggered.connect(self._action_redo)
        self._act_redo.setEnabled(False)
        edit_menu.addAction(self._act_redo)
        
        edit_menu.addSeparator()
        
        act_resize = QtGui.QAction("Resize Canvas...", self)
        act_resize.triggered.connect(self._action_resize)
        edit_menu.addAction(act_resize)
        
        act_clear = QtGui.QAction("Clear All", self)
        act_clear.triggered.connect(self._action_clear)
        edit_menu.addAction(act_clear)
        
        edit_menu.addSeparator()
        
        act_set_anchor = QtGui.QAction("Set Anchor Point", self)
        act_set_anchor.setShortcut("A")
        act_set_anchor.triggered.connect(self._action_set_anchor_mode)
        edit_menu.addAction(act_set_anchor)
        
        act_create_element = QtGui.QAction("Create Element from Selection...", self)
        act_create_element.setShortcut("Ctrl+Shift+E")
        act_create_element.triggered.connect(self._action_create_element)
        edit_menu.addAction(act_create_element)
        
        # Tools メニュー
        tools_menu = menubar.addMenu("Tools")
        
        self._tool_group = QtGui.QActionGroup(self)
        
        act_point = QtGui.QAction("Point Tool", self, checkable=True)
        act_point.setShortcut("P")
        act_point.setChecked(True)
        act_point.triggered.connect(lambda: self._set_tool(EditTool.POINT))
        self._tool_group.addAction(act_point)
        tools_menu.addAction(act_point)
        
        act_rect = QtGui.QAction("Rectangle Tool", self, checkable=True)
        act_rect.setShortcut("R")
        act_rect.triggered.connect(lambda: self._set_tool(EditTool.RECT))
        self._tool_group.addAction(act_rect)
        tools_menu.addAction(act_rect)
        
        act_line = QtGui.QAction("Line Tool", self, checkable=True)
        act_line.setShortcut("L")
        act_line.triggered.connect(lambda: self._set_tool(EditTool.LINE))
        self._tool_group.addAction(act_line)
        tools_menu.addAction(act_line)
        
        tools_menu.addSeparator()
        
        act_rule_editor = QtGui.QAction("Rule Editor...", self)
        act_rule_editor.triggered.connect(self._open_rule_editor)
        tools_menu.addAction(act_rule_editor)

    def _build_toolbar(self):
        """ツールバー構築"""
        toolbar = self.addToolBar("Tools")
        
        # ツール選択
        toolbar.addWidget(QtWidgets.QLabel(" Tool: "))
        self._tool_combo = QtWidgets.QComboBox()
        self._tool_combo.addItems(["Point", "Rectangle", "Line"])
        self._tool_combo.currentIndexChanged.connect(self._on_tool_combo_changed)
        toolbar.addWidget(self._tool_combo)
        
        toolbar.addSeparator()
        
        # ブラシ値選択
        toolbar.addWidget(QtWidgets.QLabel(" Brush: "))
        self._brush_combo = QtWidgets.QComboBox()
        self._brush_combo.addItems([
            "0: Vacant",
            "1: Wire",
            "2: Token",
            "-1: RecycleBin",
            "3: State3",
            "4: State4",
        ])
        self._brush_combo.setCurrentIndex(1)  # Wire
        self._brush_combo.currentIndexChanged.connect(self._on_brush_changed)
        toolbar.addWidget(self._brush_combo)
        
        toolbar.addSeparator()
        
        # カスタム値入力
        toolbar.addWidget(QtWidgets.QLabel(" Custom: "))
        self._custom_spin = QtWidgets.QSpinBox()
        self._custom_spin.setRange(-128, 127)
        self._custom_spin.setValue(1)
        self._custom_spin.valueChanged.connect(self._on_custom_value_changed)
        toolbar.addWidget(self._custom_spin)

    # ========= キャンバス更新 =========
    
    def _update_canvas(self):
        """キャンバス表示を更新"""
        qimg = array_to_qimage(self._arr)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self._pix_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self._rebuild_grid()

    def _rebuild_grid(self):
        """グリッド再描画"""
        h, w = self._arr.shape
        path = QtGui.QPainterPath()
        
        scale_x = self._view.transform().m11()
        step = 1
        if scale_x < 0.5:
            step = 8
        elif scale_x < 1.0:
            step = 4
        elif scale_x < 2.0:
            step = 2
        
        for x in range(0, w + 1, step):
            path.moveTo(x, 0)
            path.lineTo(x, h)
        for y in range(0, h + 1, step):
            path.moveTo(0, y)
            path.lineTo(w, y)
        
        self._grid_item.setPath(path)

    def _update_bcl_source(self):
        """BCLソース表示を更新"""
        if self._source_updating:
            return
        self._source_updating = True
        
        if self._raw_bcl_source:
            # ファイルから読み込んだ場合は元のソースを表示
            self._source_edit.setPlainText(self._raw_bcl_source)
        else:
            # 新規作成の場合は自動生成
            full_source = self._generate_full_bcl()
            self._source_edit.setPlainText(full_source)
        
        self._source_updating = False

    def _generate_element_placements(self) -> Tuple[List[str], set]:
        """element配置からBCL構文を生成し、カバーするセル座標のセットを返す"""
        lines = []
        covered_cells = set()
        h, w = self._arr.shape
        
        for elem_name, x, y, inst_id in self._element_placements:
            elem = self._find_element_by_name(elem_name)
            if elem is None:
                continue
            
            # place.Element構文を生成
            inst_name = f"{elem_name}_{inst_id}"
            lines.append(f"place.{elem_name}({inst_name}, {elem.param}[{x}, {y}])")
            
            # このelementがカバーするセル座標を計算
            try:
                placements = self._expand_element(elem, x, y)
                for px, py, _ in placements:
                    if 0 <= py < h and 0 <= px < w:
                        covered_cells.add((px, py))
            except Exception:
                pass
        
        return lines, covered_cells

    def _array_to_placement_lines(self, exclude_cells: set = None) -> List[str]:
        """numpy配列からplace.cell行を生成（exclude_cellsは除外）"""
        if exclude_cells is None:
            exclude_cells = set()
        
        lines = []
        h, w = self._arr.shape
        for y in range(h):
            for x in range(w):
                if (x, y) in exclude_cells:
                    continue
                v = int(self._arr[y, x])
                if v != 0:  # 0は省略
                    lines.append(f"place.cell({x}, {y}, {v})")
        return lines

    # ========= イベント処理 =========

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """キーイベント処理"""
        if event.key() == QtCore.Qt.Key_Escape:
            if self._element_creation_mode:
                self._cancel_element_creation()
                self._status.showMessage("Element creation cancelled")
                return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        """マウスイベント処理"""
        if event.type() == QtCore.QEvent.MouseMove:
            if self._handle_mouse_move(event):
                return True  # イベントを消費
        elif event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                return self._handle_left_press(event)
        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            if event.button() == QtCore.Qt.LeftButton:
                return self._handle_left_release(event)
        return super().eventFilter(obj, event)

    def _handle_mouse_move(self, event) -> bool:
        """マウス移動処理。イベントを消費した場合はTrueを返す"""
        pos = self._view.mapToScene(event.position().toPoint())
        x, y = int(pos.x()), int(pos.y())
        h, w = self._arr.shape
        
        # element作成モード中のドラッグ → プレビュー表示
        if self._element_creation_mode and self._is_dragging and self._drag_start:
            sx, sy = self._drag_start
            self._update_rect_preview(sx, sy, x, y)
            return True  # イベントを消費
        
        if 0 <= y < h and 0 <= x < w:
            v = self._arr[y, x]
            self._status.showMessage(f"({x}, {y}) = {v}")
        else:
            self._status.showMessage("")
        
        # ドラッグ中のプレビュー更新
        if self._is_dragging and self._drag_start:
            sx, sy = self._drag_start
            if self._current_tool == EditTool.RECT:
                self._update_rect_preview(sx, sy, x, y)
            elif self._current_tool == EditTool.LINE:
                self._update_line_preview(sx, sy, x, y)
        
        return False  # 通常のマウス移動はイベントを消費しない

    def _handle_left_press(self, event) -> bool:
        """左クリック押下"""
        pos = self._view.mapToScene(event.position().toPoint())
        x, y = int(pos.x()), int(pos.y())
        h, w = self._arr.shape
        
        # element作成モード中の処理
        if self._element_creation_mode:
            # 選択範囲が設定済みの場合、範囲内クリックでアンカー設定
            if self._selection_rect:
                x1, y1, x2, y2 = self._selection_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._handle_anchor_click(x, y)
                    return True
            
            # 選択範囲が未設定、または範囲外クリックの場合はドラッグ開始
            self._drag_start = (x, y)
            self._is_dragging = True
            self._clear_selection()
            self._clear_anchor()
            return True
        
        if not (0 <= y < h and 0 <= x < w):
            return False
        
        # Shift+クリックでアンカー設定
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            self._handle_anchor_click(x, y)
            return True
        
        if self._current_tool == EditTool.POINT:
            before = self._arr.copy()
            old_val = self._arr[y, x]
            self._arr[y, x] = self._brush_value
            
            # 0に設定 = 消しゴム操作の場合、BCLソースからも削除
            if self._brush_value == 0 and old_val != 0:
                self._handle_cell_erase(x, y)
            # 0以外に設定する場合、place.cell行を追加
            elif self._brush_value != 0:
                self._add_place_cell(x, y, self._brush_value)
            
            self._push_undo(before, f"Paint ({x},{y})")
            self._modified = True
            self._update_canvas()
            self._update_bcl_source()
            self._status.showMessage(f"({x}, {y}): {old_val} -> {self._brush_value}")
            return True
        
        elif self._current_tool in (EditTool.RECT, EditTool.LINE):
            self._drag_start = (x, y)
            self._is_dragging = True
            return True
        
        return False

    def _handle_left_release(self, event) -> bool:
        """左クリック解放"""
        # element作成モード中のドラッグ終了 → 選択範囲を確定
        if self._element_creation_mode and self._is_dragging and self._drag_start:
            pos = self._view.mapToScene(event.position().toPoint())
            ex, ey = int(pos.x()), int(pos.y())
            sx, sy = self._drag_start
            h, w = self._arr.shape
            
            ex = max(0, min(w - 1, ex))
            ey = max(0, min(h - 1, ey))
            
            x1, x2 = min(sx, ex), max(sx, ex)
            y1, y2 = min(sy, ey), max(sy, ey)
            
            self._selection_rect = (x1, y1, x2, y2)
            self._update_selection_display()
            self._clear_preview()
            self._is_dragging = False
            self._drag_start = None
            self._status.showMessage(f"Selected ({x1},{y1})-({x2},{y2}). Click inside to set anchor.")
            return True
        
        if not self._is_dragging or not self._drag_start:
            return False
        
        pos = self._view.mapToScene(event.position().toPoint())
        ex, ey = int(pos.x()), int(pos.y())
        sx, sy = self._drag_start
        h, w = self._arr.shape
        
        ex = max(0, min(w - 1, ex))
        ey = max(0, min(h - 1, ey))
        
        before = self._arr.copy()
        
        if self._current_tool == EditTool.RECT:
            x1, x2 = min(sx, ex), max(sx, ex)
            y1, y2 = min(sy, ey), max(sy, ey)
            
            # Ctrl+ドラッグの場合は選択のみ（element作成用）
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self._selection_rect = (x1, y1, x2, y2)
                self._update_selection_display()
                self._is_dragging = False
                self._drag_start = None
                self._clear_preview()
                self._status.showMessage(f"Selected ({x1},{y1})-({x2},{y2}). Shift+Click to set anchor, Ctrl+Shift+E to create element.")
                return True
            
            # 0に設定 = 消しゴム操作の場合、BCLソースからも削除
            if self._brush_value == 0:
                for cy in range(y1, y2 + 1):
                    for cx in range(x1, x2 + 1):
                        if before[cy, cx] != 0:
                            self._handle_cell_erase(cx, cy)
            # 0以外に設定する場合、place.cell行を追加
            elif self._brush_value != 0:
                for cy in range(y1, y2 + 1):
                    for cx in range(x1, x2 + 1):
                        self._add_place_cell(cx, cy, self._brush_value)
            
            self._arr[y1:y2+1, x1:x2+1] = self._brush_value
            self._push_undo(before, f"Fill rect ({x1},{y1})-({x2},{y2})")
        
        elif self._current_tool == EditTool.LINE:
            # 0に設定 = 消しゴム操作の場合、先にBCLソースから削除
            if self._brush_value == 0:
                self._erase_line_cells(sx, sy, ex, ey, before)
            # 0以外に設定する場合、place.cell行を追加
            elif self._brush_value != 0:
                self._add_line_cells(sx, sy, ex, ey)
            
            self._draw_line(sx, sy, ex, ey, self._brush_value)
            self._push_undo(before, f"Line ({sx},{sy})-({ex},{ey})")
        
        self._is_dragging = False
        self._drag_start = None
        self._clear_preview()
        self._modified = True
        self._update_canvas()
        self._update_bcl_source()
        return True

    def _draw_line(self, x0: int, y0: int, x1: int, y1: int, value: int):
        """Bresenhamアルゴリズムで線を描画"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        h, w = self._arr.shape
        
        while True:
            if 0 <= y0 < h and 0 <= x0 < w:
                self._arr[y0, x0] = value
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _erase_line_cells(self, x0: int, y0: int, x1: int, y1: int, before: np.ndarray):
        """線上のセルに対して消しゴム処理"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        h, w = before.shape
        
        while True:
            if 0 <= y0 < h and 0 <= x0 < w:
                if before[y0, x0] != 0:
                    self._handle_cell_erase(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _handle_cell_erase(self, x: int, y: int):
        """セルを消した時の処理（element配置またはplace.cellを削除）"""
        # まずelement配置の範囲内かチェック
        elem_to_remove = self._find_element_at_cell(x, y)
        if elem_to_remove:
            # element配置全体を削除
            self._remove_element_placement(elem_to_remove)
        else:
            # place.cell行を削除
            self._remove_place_cell(x, y)

    def _find_element_at_cell(self, x: int, y: int) -> Optional[Tuple[str, int, int, int]]:
        """指定セルを含むelement配置を検索"""
        for placement in self._element_placements:
            elem_name, ex, ey, inst_id = placement
            elem = self._find_element_by_name(elem_name)
            if elem is None:
                continue
            
            # elementを展開してセル範囲を取得
            try:
                cells = self._expand_element(elem, ex, ey)
                for cx, cy, _ in cells:
                    if cx == x and cy == y:
                        return placement
            except Exception:
                pass
        return None

    def _remove_element_placement(self, placement: Tuple[str, int, int, int]):
        """element配置を削除（キャンバスとBCLソースから）"""
        elem_name, ex, ey, inst_id = placement
        
        # キャンバスからelementを消去（呼び出し元で処理するためコメントアウト）
        # elem = self._find_element_by_name(elem_name)
        # if elem:
        #     try:
        #         cells = self._expand_element(elem, ex, ey)
        #         for cx, cy, _ in cells:
        #             h, w = self._arr.shape
        #             if 0 <= cy < h and 0 <= cx < w:
        #                 self._arr[cy, cx] = 0
        #     except Exception:
        #         pass
        
        # element配置履歴から削除
        if placement in self._element_placements:
            self._element_placements.remove(placement)
        
        # BCLソースからplace.Element行を削除
        inst_name = f"{elem_name}_{inst_id}"
        if self._raw_bcl_source:
            pattern = re.compile(
                rf"^\s*place\.{re.escape(elem_name)}\s*\(\s*{re.escape(inst_name)}\s*,.*\)\s*$",
                re.MULTILINE
            )
            self._raw_bcl_source = pattern.sub("", self._raw_bcl_source)
            # 連続する空行を整理
            self._raw_bcl_source = re.sub(r"\n{3,}", "\n\n", self._raw_bcl_source)
        
        self._status.showMessage(f"Removed element: {inst_name}")

    def _remove_place_cell(self, x: int, y: int):
        """place.cell行を削除"""
        if self._raw_bcl_source:
            # place.cell(x, y, v) の行を削除
            pattern = re.compile(
                rf"^\s*place\.cell\s*\(\s*{x}\s*,\s*{y}\s*,\s*\d+\s*\)\s*$",
                re.MULTILINE
            )
            self._raw_bcl_source = pattern.sub("", self._raw_bcl_source)

    def _add_place_cell(self, x: int, y: int, value: int):
        """place.cell行を追加（既存があれば更新）"""
        # まず既存の行を削除
        self._remove_place_cell(x, y)
        
        # 新しい行を追加
        new_line = f"place.cell({x}, {y}, {value})"
        if self._raw_bcl_source:
            self._raw_bcl_source = self._raw_bcl_source.rstrip() + "\n" + new_line + "\n"
        else:
            # 新規の場合はヘッダーと共に
            header = "# BCL File generated by BCL Editor\n"
            header += f"# Canvas size: {self._arr.shape[1]}x{self._arr.shape[0]}\n\n"
            header += "# === Cell Placements ===\n"
            self._raw_bcl_source = header + new_line + "\n"

    def _add_line_cells(self, x0: int, y0: int, x1: int, y1: int):
        """線上のセルにplace.cell行を追加"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        h, w = self._arr.shape
        
        while True:
            if 0 <= y0 < h and 0 <= x0 < w:
                self._add_place_cell(x0, y0, self._brush_value)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _update_rect_preview(self, sx: int, sy: int, ex: int, ey: int):
        """矩形プレビュー更新"""
        if self._preview_rect is None:
            self._preview_rect = QtWidgets.QGraphicsRectItem()
            pen = QtGui.QPen(QtGui.QColor(0, 120, 215), 0)
            pen.setStyle(QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            self._preview_rect.setPen(pen)
            brush = QtGui.QBrush(QtGui.QColor(0, 120, 215, 50))
            self._preview_rect.setBrush(brush)
            self._scene.addItem(self._preview_rect)
        
        x1, x2 = min(sx, ex), max(sx, ex)
        y1, y2 = min(sy, ey), max(sy, ey)
        self._preview_rect.setRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    def _update_line_preview(self, sx: int, sy: int, ex: int, ey: int):
        """線プレビュー更新"""
        if self._preview_line is None:
            self._preview_line = QtWidgets.QGraphicsLineItem()
            pen = QtGui.QPen(QtGui.QColor(0, 120, 215), 0)
            pen.setStyle(QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            self._preview_line.setPen(pen)
            self._scene.addItem(self._preview_line)
        
        self._preview_line.setLine(sx + 0.5, sy + 0.5, ex + 0.5, ey + 0.5)

    def _clear_preview(self):
        """プレビューをクリア"""
        if self._preview_rect:
            self._scene.removeItem(self._preview_rect)
            self._preview_rect = None
        if self._preview_line:
            self._scene.removeItem(self._preview_line)
            self._preview_line = None

    def _on_zoom_changed(self, z: float):
        self._zoom_label.setText(f"Zoom: {int(z * 100)}%")
        self._rebuild_grid()

    def _on_brush_changed(self, index: int):
        values = [0, 1, 2, -1, 3, 4]
        self._brush_value = values[index]
        self._custom_spin.setValue(self._brush_value)

    def _on_custom_value_changed(self, value: int):
        self._brush_value = value

    def _on_tool_combo_changed(self, index: int):
        tools = [EditTool.POINT, EditTool.RECT, EditTool.LINE]
        self._set_tool(tools[index])

    def _set_tool(self, tool: EditTool):
        """ツールを設定"""
        self._current_tool = tool
        self._tool_combo.setCurrentIndex(tool.value - 1)
        self._tool_label.setText(f"Tool: {tool.name.capitalize()}")
        self._clear_preview()

    # ========= Undo/Redo =========
    
    def _push_undo(self, before: np.ndarray, description: str = ""):
        """Undo履歴に追加"""
        action = EditAction(
            before=before.copy(),
            after=self._arr.copy(),
            description=description
        )
        self._undo_stack.append(action)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_state()

    def _action_undo(self):
        """Undo実行"""
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        self._redo_stack.append(action)
        self._arr = action.before.copy()
        self._modified = True
        self._update_canvas()
        self._update_bcl_source()
        self._update_undo_redo_state()
        self._status.showMessage(f"Undo: {action.description}")

    def _action_redo(self):
        """Redo実行"""
        if not self._redo_stack:
            return
        action = self._redo_stack.pop()
        self._undo_stack.append(action)
        self._arr = action.after.copy()
        self._modified = True
        self._update_canvas()
        self._update_bcl_source()
        self._update_undo_redo_state()
        self._status.showMessage(f"Redo: {action.description}")

    def _update_undo_redo_state(self):
        """Undo/Redoボタン状態更新"""
        self._act_undo.setEnabled(len(self._undo_stack) > 0)
        self._act_redo.setEnabled(len(self._redo_stack) > 0)

    # ========= BCL双方向編集 =========
    
    def _action_apply_bcl(self):
        """BCLソースをキャンバスに適用"""
        bcl_text = self._source_edit.toPlainText()
        try:
            before = self._arr.copy()
            comp = BCLCompiler()
            comp.parse(bcl_text)
            ir = comp.lower_to_ir()
            self._arr = self._ir_to_array(ir)
            self._push_undo(before, "Apply BCL")
            self._modified = True
            self._update_canvas()
            
            # ヘッダーとelement定義を抽出・保存
            self._bcl_header = self._extract_bcl_header(bcl_text)
            self._extract_elements_from_source(bcl_text)
            
            self._status.showMessage(f"BCL applied to canvas ({len(self._elements)} elements)")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "BCL Parse Error", str(e))

    def _load_library(self):
        """ライブラリ（lib.bcl）からelement定義を読み込む"""
        # デフォルトのライブラリパス
        lib_paths = [
            Path(__file__).parent.parent.parent / "Sample" / "bclfile" / "lib.bcl",
            Path.cwd().parent / "Sample" / "bclfile" / "lib.bcl",
            Path.home() / ".bcl" / "lib.bcl",
        ]
        
        for lib_path in lib_paths:
            if lib_path.exists():
                try:
                    lib_source = lib_path.read_text(encoding="utf-8")
                    self._library_elements = self._parse_elements(lib_source)
                    self._update_element_list()
                    self._status.showMessage(f"Library loaded: {lib_path} ({len(self._library_elements)} elements)")
                    return
                except Exception as e:
                    print(f"Warning: Failed to load library {lib_path}: {e}")
        
        # ライブラリが見つからない場合
        self._library_elements = []

    def _extract_bcl_header(self, source: str) -> str:
        """BCLソースからヘッダー部分（element定義、coord.defineなど配置以外）を抽出"""
        header_lines = []
        
        re_elem_start = re.compile(
            r"^\s*element\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{\s*$"
        )
        re_elem_end = re.compile(r"^\s*\}\s*$")
        re_coord_def = re.compile(r"^\s*coord\.define\s*\(")
        re_place = re.compile(r"^\s*place\.")
        
        lines = source.splitlines()
        i, N = 0, len(lines)
        in_element = False
        
        while i < N:
            raw = lines[i]
            line = raw.split("#", 1)[0].strip()
            
            # element開始
            if re_elem_start.match(line):
                in_element = True
                header_lines.append(raw)
                i += 1
                continue
            
            # element終了
            if in_element:
                header_lines.append(raw)
                if re_elem_end.match(line):
                    in_element = False
                i += 1
                continue
            
            # coord.define
            if re_coord_def.match(line):
                header_lines.append(raw)
                i += 1
                continue
            
            # place.*は除外（ヘッダーではない）
            if re_place.match(line):
                i += 1
                continue
            
            # 空行やコメント行はヘッダーに含める
            if not line or raw.strip().startswith("#"):
                header_lines.append(raw)
            
            i += 1
        
        return "\n".join(header_lines).rstrip()

    def _parse_elements(self, source: str) -> List[ElementDefinition]:
        """BCLソースからelement定義をパースして返す"""
        elements = []
        
        re_elem_start = re.compile(
            r"^\s*element\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{\s*$"
        )
        re_elem_end = re.compile(r"^\s*\}\s*$")
        
        lines = source.splitlines()
        i, N = 0, len(lines)
        
        while i < N:
            line = lines[i].split("#", 1)[0].strip()
            m = re_elem_start.match(line)
            if m:
                name, param = m.group(1), m.group(2)
                body = []
                i += 1
                while i < N:
                    if re_elem_end.match(lines[i].split("#", 1)[0].strip()):
                        break
                    body.append(lines[i])
                    i += 1
                elements.append(ElementDefinition(name, param, body))
            i += 1
        
        return elements

    def _extract_elements_from_source(self, source: str):
        """BCLソースからelement定義を抽出"""
        self._elements = self._parse_elements(source)
        self._update_element_list()

    def _extract_element_placements_from_source(self, source: str):
        """BCLソースからplace.Element構文を解析してelement配置履歴を復元"""
        # place.ElementName(InstName, ParamName[x, y]) の形式を解析
        re_place_elem = re.compile(
            r"^\s*place\.([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]\s*\)\s*$"
        )
        
        # coord.define を収集（座標解決用）
        coord_syms = {}
        re_coord_def = re.compile(
            r"^\s*coord\.define\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*$"
        )
        
        for line in source.splitlines():
            stripped = line.split("#", 1)[0].strip()
            m = re_coord_def.match(stripped)
            if m:
                name = m.group(1)
                try:
                    x = int(m.group(2).strip())
                    y = int(m.group(3).strip())
                    coord_syms[name] = (x, y)
                except ValueError:
                    pass
        
        for line in source.splitlines():
            stripped = line.split("#", 1)[0].strip()
            m = re_place_elem.match(stripped)
            if m:
                elem_name = m.group(1)
                # inst_name = m.group(2)  # 使用しない
                # param_name = m.group(3)  # 使用しない
                x_expr = m.group(4).strip()
                y_expr = m.group(5).strip()
                
                # 座標を解決
                try:
                    x = self._eval_coord_expr(x_expr, coord_syms)
                    y = self._eval_coord_expr(y_expr, coord_syms)
                    
                    # element配置履歴に追加
                    self._element_placements.append((elem_name, x, y, self._next_instance_id))
                    self._next_instance_id += 1
                except Exception:
                    pass

    def _eval_coord_expr(self, expr: str, coord_syms: dict) -> int:
        """座標式を評価（整数 or name.x/y[+-N]）"""
        s = expr.replace(" ", "")
        
        # 整数リテラル
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        
        # name.x[+-N] or name.y[+-N]
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\.(x|y)([+-]\d+)?", s)
        if m:
            name, axis, off = m.group(1), m.group(2), m.group(3)
            if name in coord_syms:
                base = coord_syms[name][0 if axis == "x" else 1]
                delta = int(off) if off else 0
                return base + delta
        
        raise ValueError(f"Cannot evaluate: {expr}")

    def _update_element_list(self):
        """element定義リストを更新（ライブラリ + 現在ファイル、重複除外）"""
        self._element_list.clear()
        
        # ファイル内で定義されたelement名を収集
        file_element_names = {elem.name for elem in self._elements}
        
        # ライブラリelement（グレー表示、ファイル内に同名があれば除外）
        for elem in self._library_elements:
            if elem.name in file_element_names:
                continue  # ファイル内に同名があれば非表示
            item = QtWidgets.QListWidgetItem(f"[lib] {elem.name}({elem.param})")
            item.setData(QtCore.Qt.UserRole, elem)
            item.setForeground(QtGui.QColor(100, 100, 100))
            self._element_list.addItem(item)
        
        # 現在ファイルのelement
        for elem in self._elements:
            item = QtWidgets.QListWidgetItem(f"{elem.name}({elem.param})")
            item.setData(QtCore.Qt.UserRole, elem)
            self._element_list.addItem(item)

    def _find_element_by_name(self, name: str) -> Optional[ElementDefinition]:
        """名前でelement定義を検索（現在ファイル優先、次にライブラリ）"""
        # 現在ファイルから検索
        for e in self._elements:
            if e.name == name:
                return e
        # ライブラリから検索
        for e in self._library_elements:
            if e.name == name:
                return e
        return None

    def _on_element_double_clicked(self, item: QtWidgets.QListWidgetItem):
        """element定義をダブルクリックで詳細表示"""
        elem: ElementDefinition = item.data(QtCore.Qt.UserRole)
        if elem:
            body_text = "\n".join(elem.body)
            QtWidgets.QMessageBox.information(
                self, f"Element: {elem.name}",
                f"Parameter: {elem.param}\n\nBody:\n{body_text}"
            )

    def _on_element_drag_move(self, element_name: str, x: int, y: int):
        """elementドラッグ中のプレビュー表示"""
        # 既存のプレビューをクリア
        self._clear_element_preview()
        
        h, w = self._arr.shape
        if not (0 <= y < h and 0 <= x < w):
            return
        
        # element検索（現在ファイル + ライブラリ）
        elem = self._find_element_by_name(element_name)
        if elem is None:
            return
        
        # elementを展開してプレビュー表示
        try:
            placements = self._expand_element(elem, x, y)
            
            for px, py, pv in placements:
                if 0 <= py < h and 0 <= px < w:
                    # セルごとにプレビュー矩形を作成
                    rect_item = QtWidgets.QGraphicsRectItem(px, py, 1, 1)
                    
                    # 値に応じた色（半透明）
                    color = self._get_preview_color(pv)
                    rect_item.setBrush(QtGui.QBrush(color))
                    rect_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                    rect_item.setOpacity(0.6)
                    rect_item.setZValue(100)  # グリッドより上
                    
                    self._scene.addItem(rect_item)
                    self._element_preview_items.append(rect_item)
            
            self._status.showMessage(f"Preview: {elem.name} at ({x}, {y}) - {len(placements)} cells")
        except Exception:
            pass

    def _on_element_drag_leave(self):
        """elementドラッグ離脱時にプレビューをクリア"""
        self._clear_element_preview()

    def _clear_element_preview(self):
        """elementプレビューをクリア"""
        for item in self._element_preview_items:
            self._scene.removeItem(item)
        self._element_preview_items.clear()

    def _get_preview_color(self, value: int) -> QtGui.QColor:
        """値に応じたプレビュー色を返す（実際の表示色と同じ）"""
        # array_to_qimageと同じパレットを使用
        palette = {
            -1: (80, 80, 80),      # RecycleBin: グレー
            0: (0, 0, 0),          # Vacant: 黒
            1: (255, 255, 255),    # Wire: 白
            2: (255, 80, 80),      # Token: 赤
            3: (80, 255, 80),      # State3: 緑
            4: (80, 80, 255),      # State4: 青
        }
        r, g, b = palette.get(value, (200, 200, 100))
        return QtGui.QColor(r, g, b, 255)  # 完全不透明

    def _on_element_dropped(self, element_name: str, x: int, y: int):
        """elementをキャンバスにドロップして配置"""
        h, w = self._arr.shape
        if not (0 <= y < h and 0 <= x < w):
            self._status.showMessage(f"Drop position out of bounds")
            return
        
        # element検索（現在ファイル + ライブラリ）
        elem = self._find_element_by_name(element_name)
        if elem is None:
            self._status.showMessage(f"Element '{element_name}' not found")
            return
        
        # ライブラリelementを使う場合、ファイル内にelement定義がなければ追加
        is_library_element = elem not in self._elements
        if is_library_element:
            # element定義をファイルに追加
            self._add_library_element_to_file(elem)
        
        # elementを展開してセル配置
        try:
            before = self._arr.copy()
            placements = self._expand_element(elem, x, y)
            
            for px, py, pv in placements:
                if 0 <= py < h and 0 <= px < w:
                    self._arr[py, px] = pv
            
            # element配置履歴に追加
            inst_id = self._next_instance_id
            self._element_placements.append((element_name, x, y, inst_id))
            self._next_instance_id += 1
            
            # BCLソースに追記
            inst_name = f"{element_name}_{inst_id}"
            new_line = f"place.{element_name}({inst_name}, {elem.param}[{x}, {y}])"
            if self._raw_bcl_source:
                self._raw_bcl_source = self._raw_bcl_source.rstrip() + "\n" + new_line + "\n"
            
            self._push_undo(before, f"Place {elem.name} at ({x},{y})")
            self._modified = True
            self._update_canvas()
            self._update_bcl_source()
            self._status.showMessage(f"Placed {elem.name} at ({x}, {y})")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Element Expansion Error", str(e))

    def _add_library_element_to_file(self, elem: ElementDefinition):
        """ライブラリelementの定義をファイルに追加"""
        # element定義文字列を生成
        element_def = f"element {elem.name}({elem.param}){{\n"
        for line in elem.body:
            element_def += line + "\n"
        element_def += "}\n"
        
        # _elementsに追加
        self._elements.append(elem)
        self._update_element_list()
        
        # _raw_bcl_sourceに追加（先頭に挿入）
        if self._raw_bcl_source:
            self._raw_bcl_source = self._insert_element_definition(self._raw_bcl_source, element_def)
        else:
            # 新規ファイルの場合
            header = "# BCL File generated by BCL Editor\n"
            header += f"# Canvas size: {self._arr.shape[1]}x{self._arr.shape[0]}\n\n"
            self._raw_bcl_source = header + element_def + "\n"

    def _expand_element(self, elem: ElementDefinition, base_x: int, base_y: int) -> List[Tuple[int, int, int]]:
        """elementを展開して(x, y, value)のリストを返す"""
        placements = []
        
        # element本体のplace.*文を解析
        re_sig = re.compile(r"^\s*place\.signal_line\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
        re_token = re.compile(r"^\s*place\.token\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
        re_recycle = re.compile(r"^\s*place\.recycle_bin\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
        re_cell = re.compile(r"^\s*place\.cell\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
        
        # パラメータ名（io_aなど）を基準座標に置換するための辞書
        param_name = elem.param
        
        for raw in elem.body:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            
            # 各place文をマッチして座標を計算
            m = re_sig.match(line)
            if m:
                px = self._eval_element_coord(m.group(1), param_name, base_x, base_y)
                py = self._eval_element_coord(m.group(2), param_name, base_x, base_y)
                placements.append((px, py, 1))
                continue
            
            m = re_token.match(line)
            if m:
                px = self._eval_element_coord(m.group(1), param_name, base_x, base_y)
                py = self._eval_element_coord(m.group(2), param_name, base_x, base_y)
                placements.append((px, py, 2))
                continue
            
            m = re_recycle.match(line)
            if m:
                px = self._eval_element_coord(m.group(1), param_name, base_x, base_y)
                py = self._eval_element_coord(m.group(2), param_name, base_x, base_y)
                placements.append((px, py, -1))
                continue
            
            m = re_cell.match(line)
            if m:
                px = self._eval_element_coord(m.group(1), param_name, base_x, base_y)
                py = self._eval_element_coord(m.group(2), param_name, base_x, base_y)
                pv = self._eval_element_coord(m.group(3), param_name, base_x, base_y)
                placements.append((px, py, pv))
                continue
        
        return placements

    def _eval_element_coord(self, expr: str, param_name: str, base_x: int, base_y: int) -> int:
        """element内の座標式を評価（param_name.x/yを基準座標に置換）"""
        s = expr.replace(" ", "")
        
        # 整数リテラル
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        
        # param_name.x or param_name.y ([+-]N)?
        pattern = rf"^{re.escape(param_name)}\.(x|y)([+-]\d+)?$"
        m = re.fullmatch(pattern, s)
        if m:
            axis = m.group(1)
            off = int(m.group(2)) if m.group(2) else 0
            base = base_x if axis == "x" else base_y
            return base + off
        
        raise ValueError(f"Cannot evaluate expression: {expr}")

    # ========= アクション =========

    def _action_new(self):
        """新規作成"""
        dialog = ResizeDialog(50, 50, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            w, h = dialog.get_size()
            self._arr = np.zeros((h, w), dtype=np.int8)
            self._current_file = None
            self._modified = False
            # element配置履歴とソースをクリア
            self._element_placements.clear()
            self._next_instance_id = 1
            self._bcl_header = ""
            self._raw_bcl_source = ""
            self._elements.clear()
            self._update_element_list()
            self._update_canvas()
            self._update_bcl_source()
            self.setWindowTitle("BCL Editor - New")

    def _action_open(self):
        """BCLファイルを開く"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open BCL File", "", "BCL Files (*.bcl);;All Files (*)")
        if not path:
            return
        
        try:
            # BCLソースを読み込み
            bcl_source = Path(path).read_text(encoding="utf-8")
            
            # まずコンパイルを試行（エラーがあれば例外）
            comp = BCLCompiler()
            comp.read_file(path)
            comp.parse()
            ir = comp.lower_to_ir()
            
            # コンパイル成功後に状態を更新
            self._raw_bcl_source = bcl_source
            
            # IRからnumpy配列を構築
            self._arr = self._ir_to_array(ir)
            self._current_file = path
            self._modified = False
            self._update_canvas()
            
            # element定義を抽出（ファイル内のelement）
            self._extract_elements_from_source(bcl_source)
            
            # element配置履歴をクリアして復元
            self._element_placements.clear()
            self._next_instance_id = 1
            self._extract_element_placements_from_source(bcl_source)
            
            # 元のソースをそのまま表示
            self._source_updating = True
            self._source_edit.setPlainText(bcl_source)
            self._source_updating = False
            
            self.setWindowTitle(f"BCL Editor - {Path(path).name}")
            self._status.showMessage(f"Loaded: {path} ({len(self._elements)} elements)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load BCL:\n{e}")

    def _ir_to_array(self, ir: CompileResult) -> np.ndarray:
        """CompileResult(YAML dict)からnumpy配列を構築"""
        placements = ir.yaml_dict
        if not placements:
            return np.zeros((50, 50), dtype=np.int8)
        
        xs = [p["coord"]["x"] for p in placements]
        ys = [p["coord"]["y"] for p in placements]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # マージンを追加
        margin = 5
        w = max_x - min_x + 1 + margin * 2
        h = max_y - min_y + 1 + margin * 2
        
        arr = np.zeros((h, w), dtype=np.int8)
        off_x = -min_x + margin
        off_y = -min_y + margin
        
        for p in placements:
            x = p["coord"]["x"] + off_x
            y = p["coord"]["y"] + off_y
            v = p["value"]
            arr[y, x] = v
        
        return arr

    def _action_save(self):
        """保存"""
        if self._current_file:
            self._save_to_file(self._current_file)
        else:
            self._action_save_as()

    def _action_save_as(self):
        """名前を付けて保存"""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save BCL File", "", "BCL Files (*.bcl);;All Files (*)")
        if not path:
            return
        if not path.endswith(".bcl"):
            path += ".bcl"
        self._save_to_file(path)

    def _save_to_file(self, path: str):
        """BCLファイルに保存"""
        try:
            if self._raw_bcl_source:
                # ファイルから読み込んだ場合は編集済みソースを保存
                bcl_content = self._raw_bcl_source
            else:
                # 新規作成の場合は自動生成
                bcl_content = self._generate_full_bcl()
            
            Path(path).write_text(bcl_content, encoding="utf-8")
            self._current_file = path
            self._modified = False
            self.setWindowTitle(f"BCL Editor - {Path(path).name}")
            self._status.showMessage(f"Saved: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _generate_full_bcl(self) -> str:
        """完全なBCLファイル内容を生成（使用したelement定義を含む）"""
        lines = []
        lines.append("# BCL File generated by BCL Editor")
        lines.append(f"# Canvas size: {self._arr.shape[1]}x{self._arr.shape[0]}")
        lines.append("")
        
        # 使用したelementを収集
        used_elements = set()
        for elem_name, _, _, _ in self._element_placements:
            used_elements.add(elem_name)
        
        # 使用したelement定義を出力
        if used_elements:
            lines.append("# === Element Definitions ===")
            for elem_name in sorted(used_elements):
                elem = self._find_element_by_name(elem_name)
                if elem:
                    lines.append(f"element {elem.name}({elem.param}){{")
                    for body_line in elem.body:
                        lines.append(body_line)
                    lines.append("}")
                    lines.append("")
        
        # element配置
        element_lines, covered_cells = self._generate_element_placements()
        if element_lines:
            lines.append("# === Element Placements ===")
            lines.extend(element_lines)
            lines.append("")
        
        # 残りのセル配置
        cell_lines = self._array_to_placement_lines(covered_cells)
        if cell_lines:
            lines.append("# === Cell Placements ===")
            lines.extend(cell_lines)
        
        return "\n".join(lines)

    def _action_export_yaml(self):
        """YAMLにエクスポート"""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export YAML", "", "YAML Files (*.yaml);;All Files (*)")
        if not path:
            return
        if not path.endswith(".yaml"):
            path += ".yaml"
        
        try:
            # BCLソースを一時コンパイル
            bcl_text = self._source_edit.toPlainText()
            comp = BCLCompiler()
            comp.parse(bcl_text)
            ir = comp.lower_to_ir()
            comp.write_yaml(ir, path)
            self._status.showMessage(f"Exported: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def _action_resize(self):
        """キャンバスサイズ変更"""
        h, w = self._arr.shape
        dialog = ResizeDialog(w, h, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            new_w, new_h = dialog.get_size()
            new_arr = np.zeros((new_h, new_w), dtype=np.int8)
            
            # 既存データをコピー
            copy_h = min(h, new_h)
            copy_w = min(w, new_w)
            new_arr[:copy_h, :copy_w] = self._arr[:copy_h, :copy_w]
            
            self._arr = new_arr
            self._modified = True
            self._update_canvas()
            self._update_bcl_source()

    def _action_clear(self):
        """全クリア"""
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All", "Clear all cells to 0?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            before = self._arr.copy()
            self._arr.fill(0)
            # element配置履歴もクリア
            self._element_placements.clear()
            # BCLソースを再生成
            self._raw_bcl_source = ""
            self._push_undo(before, "Clear All")
            self._modified = True
            self._update_canvas()
            self._update_bcl_source()

    def _action_set_anchor_mode(self):
        """アンカーポイント設定モードに入る"""
        self._status.showMessage("Click on canvas to set anchor point (origin for element)")

    def _open_rule_editor(self):
        """Rule Editorを開く"""
        if self._rule_editor_window is None or not self._rule_editor_window.isVisible():
            self._rule_editor_window = RuleEditorWindow(self)
        self._rule_editor_window.show()
        self._rule_editor_window.raise_()
        self._rule_editor_window.activateWindow()

    def _start_element_creation(self):
        """element作成モードを開始"""
        self._element_creation_mode = True
        self._new_element_btn.setText("Finish Element")
        self._new_element_btn.clicked.disconnect()
        self._new_element_btn.clicked.connect(self._finish_element_creation)
        
        # 選択範囲をクリア（ドラッグで指定する）
        self._selection_rect = None
        self._clear_selection()
        self._clear_anchor()
        
        self._status.showMessage("Drag on canvas to select area. Then click inside to set anchor. Click 'Finish Element' when done.")

    def _finish_element_creation(self):
        """element作成を完了"""
        if self._selection_rect is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Selection area is not set.")
            self._cancel_element_creation()
            return
        
        if self._anchor_point is None:
            QtWidgets.QMessageBox.warning(
                self, "No Anchor",
                "Please click inside the selection to set anchor point first.")
            return
        
        # element作成ダイアログを表示
        self._action_create_element()
        
        # モード終了
        self._cancel_element_creation()

    def _cancel_element_creation(self):
        """element作成モードをキャンセル"""
        self._element_creation_mode = False
        self._new_element_btn.setText("New Element...")
        self._new_element_btn.clicked.disconnect()
        self._new_element_btn.clicked.connect(self._start_element_creation)
        
        self._clear_selection()
        self._clear_anchor()
        self._is_dragging = False
        self._drag_start = None

    def _action_create_element(self):
        """選択範囲とアンカーからelement定義を作成"""
        if self._selection_rect is None:
            QtWidgets.QMessageBox.warning(
                self, "No Selection",
                "Please use Rectangle Tool to select an area first.\n"
                "Then set an anchor point with 'A' key or Shift+Click.")
            return
        
        if self._anchor_point is None:
            QtWidgets.QMessageBox.warning(
                self, "No Anchor Point",
                "Please set an anchor point with 'A' key or Shift+Click.\n"
                "The anchor is the origin point for the element.")
            return
        
        x1, y1, x2, y2 = self._selection_rect
        ax, ay = self._anchor_point
        
        # アンカーが選択範囲内にあるか確認
        if not (x1 <= ax <= x2 and y1 <= ay <= y2):
            QtWidgets.QMessageBox.warning(
                self, "Anchor Outside Selection",
                "Anchor point must be inside the selection rectangle.")
            return
        
        # ダイアログで名前を取得
        dialog = CreateElementDialog(self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        
        elem_name, param_name = dialog.get_values()
        
        # element定義を生成
        body_lines = []
        h, w = self._arr.shape
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if 0 <= y < h and 0 <= x < w:
                    v = int(self._arr[y, x])
                    if v != 0:  # 0は省略
                        # アンカーからの相対座標
                        dx = x - ax
                        dy = y - ay
                        if dx == 0 and dy == 0:
                            body_lines.append(f"    place.cell({param_name}.x, {param_name}.y, {v})")
                        else:
                            dx_str = f"{param_name}.x+{dx}" if dx > 0 else (f"{param_name}.x{dx}" if dx < 0 else f"{param_name}.x")
                            dy_str = f"{param_name}.y+{dy}" if dy > 0 else (f"{param_name}.y{dy}" if dy < 0 else f"{param_name}.y")
                            body_lines.append(f"    place.cell({dx_str}, {dy_str}, {v})")
        
        if not body_lines:
            QtWidgets.QMessageBox.warning(self, "Empty Selection", "No non-zero cells in selection.")
            return
        
        # element定義文字列を生成
        element_def = f"element {elem_name}({param_name}){{\n"
        element_def += "\n".join(body_lines)
        element_def += "\n}\n"
        
        # BCLソースに追加
        if self._raw_bcl_source:
            # ヘッダー部分（element定義の後）に挿入
            self._raw_bcl_source = self._insert_element_definition(self._raw_bcl_source, element_def)
        else:
            # 新規の場合は先頭に追加
            current = self._source_edit.toPlainText()
            self._raw_bcl_source = element_def + "\n" + current
        
        # element定義リストに追加
        new_elem = ElementDefinition(elem_name, param_name, body_lines)
        self._elements.append(new_elem)
        self._update_element_list()
        
        # 表示更新
        self._update_bcl_source()
        self._modified = True
        
        # 選択とアンカーをクリア
        self._clear_selection()
        self._clear_anchor()
        
        self._status.showMessage(f"Created element: {elem_name}")

    def _insert_element_definition(self, source: str, element_def: str) -> str:
        """BCLソースのelement定義セクションにelement定義を挿入"""
        lines = source.splitlines()
        
        # 最後のelement定義の閉じ括弧を探す
        last_element_end = -1
        in_element = False
        re_elem_start = re.compile(r"^\s*element\s+")
        re_elem_end = re.compile(r"^\s*\}\s*$")
        
        for i, line in enumerate(lines):
            if re_elem_start.match(line):
                in_element = True
            elif in_element and re_elem_end.match(line):
                last_element_end = i
                in_element = False
        
        if last_element_end >= 0:
            # 最後のelement定義の後に挿入
            lines.insert(last_element_end + 1, "")
            lines.insert(last_element_end + 2, element_def.rstrip())
            return "\n".join(lines)
        else:
            # element定義がない場合は先頭に追加
            return element_def + "\n" + source

    def _handle_anchor_click(self, x: int, y: int):
        """アンカーポイントを設定"""
        h, w = self._arr.shape
        if 0 <= x < w and 0 <= y < h:
            self._anchor_point = (x, y)
            self._update_anchor_display()
            self._status.showMessage(f"Anchor set at ({x}, {y})")

    def _update_anchor_display(self):
        """アンカーポイントの表示を更新"""
        if self._anchor_item:
            self._scene.removeItem(self._anchor_item)
            self._anchor_item = None
        
        if self._anchor_point:
            x, y = self._anchor_point
            # 赤い円でアンカーを表示
            self._anchor_item = self._scene.addEllipse(
                x + 0.2, y + 0.2, 0.6, 0.6,
                QtGui.QPen(QtGui.QColor(255, 0, 0), 0.1),
                QtGui.QBrush(QtGui.QColor(255, 0, 0, 180))
            )
            self._anchor_item.setZValue(100)

    def _clear_anchor(self):
        """アンカーポイントをクリア"""
        self._anchor_point = None
        if self._anchor_item:
            self._scene.removeItem(self._anchor_item)
            self._anchor_item = None

    def _update_selection_display(self):
        """選択範囲の表示を更新"""
        if self._selection_item:
            self._scene.removeItem(self._selection_item)
            self._selection_item = None
        
        if self._selection_rect:
            x1, y1, x2, y2 = self._selection_rect
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            self._selection_item = self._scene.addRect(
                x1, y1, w, h,
                QtGui.QPen(QtGui.QColor(0, 120, 255), 0.1, QtCore.Qt.DashLine),
                QtGui.QBrush(QtGui.QColor(0, 120, 255, 50))
            )
            self._selection_item.setZValue(90)

    def _clear_selection(self):
        """選択範囲をクリア"""
        self._selection_rect = None
        if self._selection_item:
            self._scene.removeItem(self._selection_item)
            self._selection_item = None


# ========= ダイアログ =========

class ResizeDialog(QtWidgets.QDialog):
    """サイズ変更ダイアログ"""
    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canvas Size")
        self.setModal(True)
        
        layout = QtWidgets.QFormLayout(self)
        
        self._width_spin = QtWidgets.QSpinBox()
        self._width_spin.setRange(1, 1000)
        self._width_spin.setValue(width)
        layout.addRow("Width:", self._width_spin)
        
        self._height_spin = QtWidgets.QSpinBox()
        self._height_spin.setRange(1, 1000)
        self._height_spin.setValue(height)
        layout.addRow("Height:", self._height_spin)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_size(self) -> Tuple[int, int]:
        return self._width_spin.value(), self._height_spin.value()


class CreateElementDialog(QtWidgets.QDialog):
    """element作成ダイアログ"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Element")
        self.setMinimumWidth(300)
        
        layout = QtWidgets.QFormLayout(self)
        
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setPlaceholderText("e.g., MyElement")
        layout.addRow("Element Name:", self._name_edit)
        
        self._param_edit = QtWidgets.QLineEdit("io_a")
        self._param_edit.setPlaceholderText("e.g., io_a")
        layout.addRow("Parameter Name:", self._param_edit)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def _validate_and_accept(self):
        name = self._name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Error", "Element name is required")
            return
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid element name (use alphanumeric and underscore)")
            return
        param = self._param_edit.text().strip()
        if not param:
            QtWidgets.QMessageBox.warning(self, "Error", "Parameter name is required")
            return
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', param):
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid parameter name")
            return
        self.accept()
    
    def get_values(self) -> Tuple[str, str]:
        return self._name_edit.text().strip(), self._param_edit.text().strip()


# ========= エントリポイント =========

def main(argv=None):
    app = QtWidgets.QApplication(argv or sys.argv)
    win = BCLEditorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
