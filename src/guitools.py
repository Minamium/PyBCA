# src/guitools.py
# 依存: pip install PySide6 pyyaml numpy
# 用途:
#   - YAMLセル空間をNumPy配列に変換して表示
#   - ズーム/パン、グリッド表示、座標/値のステータス表示
#   - 将来の編集機能をQGraphicsScene上に拡張しやすい構成

from __future__ import annotations
import sys
import random
from typing import Dict, Tuple, Optional, List

import yaml
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from lib import load_cell_space_yaml_to_numpy, numpy_to_cell_space_yaml, load_transition_rules_yaml, TransitionRule, extract_cellspace_and_offset, has_offset_info, load_multiple_transition_rules_to_numpy, get_rule_ids_from_files

# ========= 配列 -> QImage（高速LUT） =========

def build_palette(states: List[int]) -> Dict[int, Tuple[int, int, int]]:
    """
    状態→色の簡易パレット。
    既定: 0=黒, 1=白, 2=赤。他の値はランダム色。
    """
    pal: Dict[int, Tuple[int, int, int]] = {
        0: (255, 255, 255),
        1: (180, 180, 180),
        2: (0, 0, 0),
        -1: (255, 0, 0)
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
    """
    任意の整数配列をRGB QImageへ（負値OK）。値→連番→LUTの3段で高速変換。
    """
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
    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return qimg.copy()  # バッファ寿命の独立性を確保


# ========= QGraphics ベースのビューア =========

class CellSpaceView(QtWidgets.QGraphicsView):
    """
    将来の編集モード（セル単位のペイント/矩形編集/スナップ等）を想定したビュー。
    - 右ドラッグ/左ドラッグは将来の編集に割当可能
    - 現在はホイールズーム + 中ドラッグパン
    """
    zoomChanged = QtCore.Signal(float)  # 表示倍率

    def __init__(self, scene: QtWidgets.QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self._zoom = 1.0

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        # Ctrl+Wheel でより細かく、通常は1.2倍
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 1 / 1.25
        self._zoom *= factor
        self.scale(factor, factor)
        self.zoomChanged.emit(self._zoom)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            # 中ボタンでパン開始
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            fake = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
                                     event.localPos(),
                                     QtCore.Qt.LeftButton,
                                     QtCore.Qt.LeftButton,
                                     QtCore.Qt.NoModifier)
            super().mousePressEvent(fake)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            fake = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease,
                                     event.localPos(),
                                     QtCore.Qt.LeftButton,
                                     QtCore.Qt.LeftButton,
                                     QtCore.Qt.NoModifier)
            super().mouseReleaseEvent(fake)
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            return
        super().mouseReleaseEvent(event)


class CellSpaceWindow(QtWidgets.QMainWindow):
    """
    メインウィンドウ。グリッド描画やステータス表示、YAMLロードを提供。
    画像（QGraphicsPixmapItem）ベース → 将来はセル矩形を個別アイテム化も可能。
    """
    def __init__(self, arr: np.ndarray = None):
        super().__init__()
        self.setWindowTitle("PyBCA CellSpace Viewer")
        self._arr = None
        self._view = None
        self._scene = None
        self._pix = None
        self._grid_item = None
        self._grid_visible = True
        self._status = None
        
        # 遷移規則管理 - numpy配列形式で統一
        self._loaded_rule_files: List[str] = []  # 読み込み済みファイルパス
        self._loaded_rule_array: np.ndarray = None  # (N, 2, 3, 3) 形状の統合配列
        self._loaded_rule_ids: List[int] = []  # 対応するrule_id
        self._rule_viewer = None
        
        # セル空間オフセット情報
        self._cellspace_offset = (0, 0)  # (min_x, min_y)

        # Scene / View / PixmapItem
        self._scene = QtWidgets.QGraphicsScene(self)
        self._pix = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pix)

        self._view = CellSpaceView(self._scene, self)
        self.setCentralWidget(self._view)

        # ステータスバー（座標・値・倍率）
        self._status = self.statusBar()
        self._zoom_label = QtWidgets.QLabel("zoom: 100%")
        self._status.addPermanentWidget(self._zoom_label)

        # グリッド描画レイヤ
        self._grid_item = QtWidgets.QGraphicsPathItem()
        pen = QtGui.QPen(QtGui.QColor(80, 80, 80, 80), 0)  # Cosmetic pen
        pen.setCosmetic(True)
        self._grid_item.setPen(pen)
        self._scene.addItem(self._grid_item)
        self._grid_visible = True

        # マウス追従（座標・値）
        self._view.setMouseTracking(True)
        self._view.viewport().setMouseTracking(True)
        self._view.viewport().installEventFilter(self)

        # メニュー
        self._build_menu()
        self.resize(1200, 900)

        if arr is not None:
            self.set_array(arr)

        # ズーム表示更新
        self._view.zoomChanged.connect(self._on_zoom_changed)

    # ---- public API ----
    def set_array(self, arr: np.ndarray) -> None:
        """配列をセットして表示を更新"""
        if arr is None:
            self._arr = None
            self._pix.setPixmap(QtGui.QPixmap())
            self._status.showMessage("No data")
            return

        # オフセット情報付きの場合は分離
        if has_offset_info(arr) and arr.shape[0] > 1 and arr.shape[1] > 2:
            try:
                cellspace, min_x, min_y = extract_cellspace_and_offset(arr)
                self._arr = cellspace
                self._cellspace_offset = (min_x, min_y)
                status_msg = f"Loaded: size={cellspace.shape}, offset=({min_x},{min_y})"
            except:
                # オフセット抽出に失敗した場合は通常の配列として扱う
                self._arr = arr
                status_msg = f"Loaded: size={arr.shape}"
        else:
            self._arr = arr
            status_msg = f"Loaded: size={arr.shape}"

        qimg = array_to_qimage(self._arr)
        self._pix.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self._view.setSceneRect(self._pix.boundingRect())
        
        # 初期表示時はセル空間全体がウィンドウに収まるように明示的にズーム計算
        view_rect = self._view.viewport().rect()
        scene_rect = self._pix.boundingRect()
        
        if scene_rect.width() > 0 and scene_rect.height() > 0:
            # ビューポートに対するスケール比を計算
            scale_x = view_rect.width() / scene_rect.width()
            scale_y = view_rect.height() / scene_rect.height()
            scale = min(scale_x, scale_y) * 0.9  # 少し余裕を持たせる
            
            # 変換をリセットしてスケールを適用
            self._view.resetTransform()
            self._view.scale(scale, scale)
            self._view._zoom = scale
            
            # 中央に配置
            self._view.centerOn(scene_rect.center())
        
        self._rebuild_grid()

    # ---- UI building ----
    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # File
        m_file = menubar.addMenu("&File")
        a_open = m_file.addAction("Open CellSpace YAML…")
        a_open.triggered.connect(self._action_open_yaml)
        m_file.addSeparator()
        a_open_rule = m_file.addAction("Open Rule YAML…")
        a_open_rule.triggered.connect(self._action_open_rule_yaml)
        m_file.addSeparator()
        a_save = m_file.addAction("Save YAML…")
        a_save.triggered.connect(self._action_save_yaml)
        m_file.addSeparator()
        a_quit = m_file.addAction("Quit")
        a_quit.triggered.connect(self.close)

        # View
        m_view = menubar.addMenu("&View")
        self._act_grid = m_view.addAction("Show Grid")
        self._act_grid.setCheckable(True)
        self._act_grid.setChecked(True)
        self._act_grid.triggered.connect(self._toggle_grid)
        
        # Rule
        m_rule = menubar.addMenu("&Rule")
        a_show_rule = m_rule.addAction("Show Loaded Rules")
        a_show_rule.triggered.connect(self._action_show_rule_pattern)
        m_rule.addSeparator()
        a_add_rule = m_rule.addAction("Add Rule File...")
        a_add_rule.triggered.connect(self._action_add_rule_file)
        m_rule.addSeparator()
        a_clear_rules = m_rule.addAction("Clear All Rules")
        a_clear_rules.triggered.connect(self._action_clear_rules)

    # ---- actions ----
    def _action_open_yaml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open CellSpace YAML", filter="YAML Files (*.yaml *.yml)")
        if not path:
            return
        
        # プログレスダイアログを作成
        progress = QtWidgets.QProgressDialog(
            "Loading YAML file...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(200)  # 200ms後に表示（早めに表示）
        progress.show()
        
        # プログレス更新用コールバック
        def update_progress(current, total):
            if progress.wasCanceled():
                raise InterruptedError("Loading was canceled by user")
            
            # 進捗に応じてメッセージを更新
            if current < 15:
                progress.setLabelText("Reading YAML file...")
            elif current < 85:
                progress.setLabelText("Parsing coordinates...")
            elif current < 90:
                progress.setLabelText("Calculating array size...")
            elif current < 97:
                progress.setLabelText("Creating array...")
            else:
                progress.setLabelText("Finalizing...")
            
            progress.setValue(current)
            QtWidgets.QApplication.processEvents()  # UI更新
        
        try:
            arr = load_cell_space_yaml_to_numpy(path, progress_callback=update_progress)
            progress.setValue(100)
            self.set_array(arr)
            self._status.showMessage(f"Loaded: {path}  size={arr.shape}")
        except InterruptedError:
            self._status.showMessage("Loading canceled")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load YAML file:\n{str(e)}")
        finally:
            progress.close()
    
    def load_rules_from_file(self):
        """遷移規則ファイルを読み込み"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Transition Rules", "", "YAML files (*.yaml *.yml)"
        )
        if path:
            try:
                # 新しい統合関数を使用
                self._loaded_rule_files = [path]
                self._loaded_rule_array = load_multiple_transition_rules_to_numpy([path])
                self._loaded_rule_ids = get_rule_ids_from_files([path])
                
                print(f"Loaded {len(self._loaded_rule_ids)} rules from {path}")
                print(f"Rule array shape: {self._loaded_rule_array.shape}")
                
                # ルールビューア用に一時的にTransitionRuleリストを作成
                temp_rules = load_transition_rules_yaml(path)
                self._rule_viewer = RuleViewerWindow(temp_rules)
                self._rule_viewer.show()
                
                self._status.showMessage(f"Loaded {len(self._loaded_rule_ids)} transition rules from {path}")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load rules: {str(e)}")
                print(f"Error loading rules: {e}")

    def _action_open_rule_yaml(self) -> None:
        """規則ファイルを読み込んで既存の規則を置き換え"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Transition Rules", "", "YAML files (*.yaml *.yml)"
        )
        if path:
            try:
                # 新しい統合関数を使用
                self._loaded_rule_files = [path]
                self._loaded_rule_array = load_multiple_transition_rules_to_numpy([path])
                self._loaded_rule_ids = get_rule_ids_from_files([path])
                
                print(f"Loaded {len(self._loaded_rule_ids)} rules from {path}")
                print(f"Rule array shape: {self._loaded_rule_array.shape}")
                
                # ルールビューア用に一時的にTransitionRuleリストを作成
                temp_rules = load_transition_rules_yaml(path)
                self._rule_viewer = RuleViewerWindow(temp_rules)
                self._rule_viewer.show()
                
                self._status.showMessage(f"Loaded {len(self._loaded_rule_ids)} transition rules from {path}")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load rules: {str(e)}")
                print(f"Error loading rules: {e}")

    def _action_add_rule_file(self) -> None:
        """規則ファイルを追加読み込み"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Add Rule YAML", filter="YAML Files (*.yaml *.yml)")
        if not path:
            return
        
        try:
            # 既存のファイルリストに追加
            if path not in self._loaded_rule_files:
                self._loaded_rule_files.append(path)
                
                # 全ファイルから統合配列を再生成
                self._loaded_rule_array = load_multiple_transition_rules_to_numpy(self._loaded_rule_files)
                self._loaded_rule_ids = get_rule_ids_from_files(self._loaded_rule_files)
                
                print(f"Added rule file: {path}")
                print(f"Total rule array shape: {self._loaded_rule_array.shape}")
                
                # ルールビューア用に一時的にTransitionRuleリストを作成
                all_temp_rules = []
                for file_path in self._loaded_rule_files:
                    temp_rules = load_transition_rules_yaml(file_path)
                    all_temp_rules.extend(temp_rules)
                
                # 規則ビューアウィンドウを更新
                if self._rule_viewer:
                    self._rule_viewer.close()
                self._rule_viewer = RuleViewerWindow(all_temp_rules)
                self._rule_viewer.show()
                
                self._status.showMessage(f"Added rule file (total: {len(self._loaded_rule_ids)} rules from {len(self._loaded_rule_files)} files)")
            else:
                QtWidgets.QMessageBox.information(
                    self, "Info", "This rule file is already loaded")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load rule YAML file:\n{str(e)}")
    
    def _action_clear_rules(self) -> None:
        """読み込んだ規則をすべてクリア"""
        if not self._loaded_rule_files:
            QtWidgets.QMessageBox.information(
                self, "Info", "No rules are currently loaded")
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, "Clear Rules", 
            f"Clear all {len(self._loaded_rule_ids)} loaded rules from {len(self._loaded_rule_files)} files?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._loaded_rule_files.clear()
            self._loaded_rule_array = None
            self._loaded_rule_ids.clear()
            if self._rule_viewer:
                self._rule_viewer.close()
                self._rule_viewer = None
            self._status.showMessage("All rules cleared")
    
    def _action_save_yaml(self) -> None:
        if self._arr is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No cell space data to save.")
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CellSpace YAML", filter="YAML Files (*.yaml *.yml)")
        if not path:
            return
        
        try:
            numpy_to_cell_space_yaml(self._arr, path)
            self._status.showMessage(f"Saved: {path}  size={self._arr.shape}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save YAML file:\n{str(e)}")

    def _action_show_rule_pattern(self) -> None:
        """読み込んだ規則パターンを表示"""
        if not self._loaded_rule_files:
            QtWidgets.QMessageBox.information(
                self, "Info", 
                "No transition rules loaded.\n"
                "Use 'Rule → Load Rule File' to load transition rules first."
            )
            return
        
        # 規則ビューアウィンドウを表示
        if self._rule_viewer is None or self._rule_viewer.isHidden():
            # ルールビューア用に一時的にTransitionRuleリストを作成
            all_temp_rules = []
            for file_path in self._loaded_rule_files:
                temp_rules = load_transition_rules_yaml(file_path)
                all_temp_rules.extend(temp_rules)
            
            # 新しいビューアを作成
            if self._rule_viewer:
                self._rule_viewer.close()
            self._rule_viewer = RuleViewerWindow(all_temp_rules)
            self._rule_viewer.show()
            
            self._status.showMessage(f"Showing {len(self._loaded_rule_ids)} loaded rules")

    def _toggle_grid(self, checked: bool) -> None:
        self._grid_visible = checked
        self._grid_item.setVisible(checked)

    def _rebuild_grid(self) -> None:
        """現在の配列サイズと表示倍率からグリッドを組み直す。"""
        if self._arr is None or not self._grid_visible:
            self._grid_item.setPath(QtGui.QPainterPath())
            return
        h, w = self._arr.shape
        path = QtGui.QPainterPath()
        # ピクセル境界に合わせた格子。倍率が低いときは間引き
        scale_x = self._view.transform().m11()
        scale_y = self._view.transform().m22()
        step = 1
        if scale_x < 0.6 or scale_y < 0.6:
            step = 8
        elif scale_x < 1.2 or scale_y < 1.2:
            step = 4
        # 垂直線
        for x in range(0, w+1, step):
            path.moveTo(x, 0)
            path.lineTo(x, h)
        # 水平線
        for y in range(0, h+1, step):
            path.moveTo(0, y)
            path.lineTo(w, y)
        self._grid_item.setPath(path)

    # ---- events ----
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseMove and self._arr is not None:
            pos = self._view.mapToScene(event.pos())
            x = int(pos.x()); y = int(pos.y())
            h, w = self._arr.shape
            if 0 <= x < w and 0 <= y < h:
                val = int(self._arr[y, x])
                self._status.showMessage(f"(x,y)=({x},{y})  val={val}")
        return super().eventFilter(obj, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._pix.pixmap().isNull():
            return
        self._view.fitInView(self._pix, QtCore.Qt.KeepAspectRatio)
        self._rebuild_grid()

    # ---- helpers ----
    def _on_zoom_changed(self, z: float) -> None:
        self._zoom_label.setText(f"zoom: {int(self._view.transform().m11()*100):d}%")
        # ズームに応じて細かすぎるグリッドは抑制（必要なら間引きロジックを強化）
        self._rebuild_grid()


class RuleViewerWindow(QtWidgets.QMainWindow):
    """
    遷移規則ビューア。prev→nextの変化を左右並列表示し、矢印ボタンで規則を切り替え。
    """
    def __init__(self, rules: List[TransitionRule]):
        super().__init__()
        self.setWindowTitle("PyBCA Rule Viewer")
        self._rules = rules
        self._current_index = 0
        
        if not self._rules:
            QtWidgets.QMessageBox.warning(self, "Warning", "No rules to display")
            return
        
        # 中央ウィジェット
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        # メインレイアウト
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # 規則情報表示
        self._rule_info = QtWidgets.QLabel()
        self._rule_info.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self._rule_info)
        
        # prev→next表示エリア
        display_layout = QtWidgets.QHBoxLayout()
        
        # prev表示
        prev_group = QtWidgets.QGroupBox("Previous State")
        prev_layout = QtWidgets.QVBoxLayout(prev_group)
        self._prev_scene = QtWidgets.QGraphicsScene()
        self._prev_view = CellSpaceView(self._prev_scene)
        self._prev_pix = QtWidgets.QGraphicsPixmapItem()
        self._prev_scene.addItem(self._prev_pix)
        prev_layout.addWidget(self._prev_view)
        display_layout.addWidget(prev_group)
        
        # 矢印
        arrow_layout = QtWidgets.QVBoxLayout()
        arrow_layout.addStretch()
        arrow_label = QtWidgets.QLabel("→")
        arrow_label.setAlignment(QtCore.Qt.AlignCenter)
        arrow_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        arrow_layout.addWidget(arrow_label)
        arrow_layout.addStretch()
        display_layout.addLayout(arrow_layout)
        
        # next表示
        next_group = QtWidgets.QGroupBox("Next State")
        next_layout = QtWidgets.QVBoxLayout(next_group)
        self._next_scene = QtWidgets.QGraphicsScene()
        self._next_view = CellSpaceView(self._next_scene)
        self._next_pix = QtWidgets.QGraphicsPixmapItem()
        self._next_scene.addItem(self._next_pix)
        next_layout.addWidget(self._next_view)
        display_layout.addWidget(next_group)
        
        main_layout.addLayout(display_layout)
        
        # コントロールボタン
        control_layout = QtWidgets.QHBoxLayout()
        
        self._prev_btn = QtWidgets.QPushButton("◀ Previous Rule")
        self._prev_btn.clicked.connect(self._prev_rule)
        control_layout.addWidget(self._prev_btn)
        
        self._rule_selector = QtWidgets.QComboBox()
        for i, rule in enumerate(self._rules):
            self._rule_selector.addItem(f"Rule {rule.rule_id}")
        self._rule_selector.currentIndexChanged.connect(self._rule_selected)
        control_layout.addWidget(self._rule_selector)
        
        self._next_btn = QtWidgets.QPushButton("Next Rule ▶")
        self._next_btn.clicked.connect(self._next_rule)
        control_layout.addWidget(self._next_btn)
        
        main_layout.addLayout(control_layout)
        
        # ステータスバー
        self._status = self.statusBar()
        
        self.resize(800, 600)
        self._update_display()
    
    def _update_display(self):
        """現在の規則を表示"""
        if not self._rules or self._current_index >= len(self._rules):
            return
        
        rule = self._rules[self._current_index]
        
        # 規則情報更新
        self._rule_info.setText(f"Rule ID: {rule.rule_id} ({self._current_index + 1}/{len(self._rules)})")
        
        # prev表示
        prev_qimg = array_to_qimage(rule.prev_pattern)
        self._prev_pix.setPixmap(QtGui.QPixmap.fromImage(prev_qimg))
        self._prev_view.setSceneRect(self._prev_pix.boundingRect())
        self._prev_view.fitInView(self._prev_pix, QtCore.Qt.KeepAspectRatio)
        
        # next表示
        next_qimg = array_to_qimage(rule.next_pattern)
        self._next_pix.setPixmap(QtGui.QPixmap.fromImage(next_qimg))
        self._next_view.setSceneRect(self._next_pix.boundingRect())
        self._next_view.fitInView(self._next_pix, QtCore.Qt.KeepAspectRatio)
        
        # ボタン状態更新
        self._prev_btn.setEnabled(self._current_index > 0)
        self._next_btn.setEnabled(self._current_index < len(self._rules) - 1)
        
        # コンボボックス更新
        self._rule_selector.setCurrentIndex(self._current_index)
        
        # ステータス更新
        self._status.showMessage(f"Rule {rule.rule_id}: prev={rule.prev_pattern.shape}, next={rule.next_pattern.shape}")
    
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


def main(argv=None):
    app = QtWidgets.QApplication(argv or sys.argv)
    # 初期状態は空で起動
    win = CellSpaceWindow(arr=None)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
