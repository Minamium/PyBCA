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
import torch

from PySide6 import QtCore, QtGui, QtWidgets

try:
    # パッケージとしてインポートされる場合
    from .lib import load_cell_space_yaml_to_numpy, numpy_to_cell_space_yaml, load_transition_rules_yaml, TransitionRule, extract_cellspace_and_offset, has_offset_info, load_multiple_transition_rules_to_numpy, load_multiple_transition_rules_with_probability, get_rule_ids_from_files, load_special_events_from_file, convert_events_to_array_coordinates, get_event_names_from_file
    from .cli_simClass import BCA_Simulator
except ImportError:
    # 直接実行される場合
    from lib import load_cell_space_yaml_to_numpy, numpy_to_cell_space_yaml, load_transition_rules_yaml, TransitionRule, extract_cellspace_and_offset, has_offset_info, load_multiple_transition_rules_to_numpy, load_multiple_transition_rules_with_probability, get_rule_ids_from_files, load_special_events_from_file, convert_events_to_array_coordinates, get_event_names_from_file
    from cli_simClass import BCA_Simulator

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
    
    # デバッグ: 配列形状とQImage作成パラメータを確認
    print(f"Debug array_to_qimage: arr.shape={arr.shape}, rgb.shape={rgb.shape}")
    print(f"Debug array_to_qimage: h={h}, w={w}, bytes_per_line={3*w}")
    
    # メモリ連続性を確保
    rgb_contiguous = np.ascontiguousarray(rgb)
    qimg = QtGui.QImage(rgb_contiguous.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
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
        self._loaded_rule_probs: np.ndarray = None  # (N,) 形状の確率配列
        self._loaded_rule_ids: List[int] = []  # 対応するrule_id
        self._rule_viewer = None
        
        # セル空間オフセット情報
        self._offset_x = 0
        self._offset_y = 0
        
        # 特殊イベント管理
        self._loaded_events: List[tuple] = []  # 読み込み済み特殊イベント
        self._event_array: np.ndarray = None   # 配列座標系変換済みイベント（オーバーレイ表示用）
        self._event_array_raw: np.ndarray = None  # 生座標イベント配列（シミュレーション用）
        self._event_names: List[str] = []      # イベント名リスト
        self._events_visible = False           # イベント表示フラグ
        self._event_file_path = None           # 特殊イベントファイルパス
        
        # ファイルパス保存
        self._cellspace_file_path = None
        self._rule_file_paths = []
        
        # シミュレーション設定
        self._sim_steps = 1
        self._sim_global_prob = 1.0
        self._sim_seed = 1
        self._sim_parallel_trials = 1
        self._sim_device = "cpu"  # デバイス設定
        self._current_step = 0  # 現在のステップ数
        
        # 編集モード設定
        self._edit_mode = False
        self._edit_brush_value = 1  # デフォルトの編集値
        
        # 連続実行用タイマー
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._timer_step)
        self._is_running = False
        self._step_interval = 500  # ミリ秒
        
        # BCA_Simulatorインスタンス
        self._bca_simulator = None
        
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
        
        # イベントオーバーレイレイヤ
        self._event_overlay_items = []  # イベント表示用のQGraphicsItemリスト
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
        
        # デフォルトファイルの自動読み込み
        self._load_default_files()

    # ---- public API ----
    def set_array(self, arr: np.ndarray) -> None:
        """配列をセットして表示を更新"""
        if arr is None:
            self._arr = None
            self._pix.setPixmap(QtGui.QPixmap())
            self._status.showMessage("No data")
            return

        # シミュレーション結果の場合はオフセット処理をスキップ
        # オフセット情報付きかどうかの判定を厳密化
        if (hasattr(self, '_bca_simulator') and self._bca_simulator is not None and 
            hasattr(self._bca_simulator, 'cellspace_tensor') and
            arr.shape == tuple(self._bca_simulator.cellspace_tensor.shape)):
            # シミュレーション結果の場合は直接使用
            self._arr = arr
            status_msg = f"Simulation result: size={arr.shape}"
        elif has_offset_info(arr) and arr.shape[0] > 1 and arr.shape[1] > 2:
            try:
                cellspace, min_x, min_y = extract_cellspace_and_offset(arr)
                self._arr = cellspace
                self._offset_x = min_x
                self._offset_y = min_y
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
        
        # セル空間が新しく読み込まれた場合、既存イベントの座標変換を更新
        if len(self._loaded_events) > 0 and self._arr is not None:
            # オーバーレイ表示用の座標変換済み配列を更新
            self._event_array = convert_events_to_array_coordinates(
                self._loaded_events, self._offset_x, self._offset_y)
            self._update_event_overlay()

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

        # Edit
        m_edit = menubar.addMenu("&Edit")
        # macOSシステムメニュー項目を無効化
        m_edit.menuAction().setMenuRole(QtGui.QAction.NoRole)
        self._act_edit_mode = m_edit.addAction("Edit Mode")
        self._act_edit_mode.setCheckable(True)
        self._act_edit_mode.setChecked(False)
        self._act_edit_mode.triggered.connect(self._toggle_edit_mode)
        m_edit.addSeparator()
        a_new_cellspace = m_edit.addAction("New Cell Space...")
        a_new_cellspace.triggered.connect(self._action_new_cellspace)
        m_edit.addSeparator()
        a_clear_all = m_edit.addAction("Clear All Cells")
        a_clear_all.triggered.connect(self._action_clear_all_cells)
        m_edit.addSeparator()
        a_fill_pattern = m_edit.addAction("Fill Pattern...")
        a_fill_pattern.triggered.connect(self._action_fill_pattern)

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
        
        # Event
        m_event = menubar.addMenu("&Event")
        a_load_event = m_event.addAction("Load Special Events...")
        a_load_event.triggered.connect(self._action_load_special_events)
        m_event.addSeparator()
        self._act_show_events = m_event.addAction("Show Events")
        self._act_show_events.setCheckable(True)
        self._act_show_events.setChecked(False)
        self._act_show_events.triggered.connect(self._toggle_events_visibility)
        m_event.addSeparator()
        a_clear_events = m_event.addAction("Clear Events")
        a_clear_events.triggered.connect(self._action_clear_events)
        
        # Simulation
        m_sim = menubar.addMenu("&Simulation")
        a_config = m_sim.addAction("Simulation Config...")
        a_config.triggered.connect(self._action_configure_simulation)
        m_sim.addSeparator()
        a_step = m_sim.addAction("Step Forward")
        a_step.triggered.connect(self._action_step_simulation)
        m_sim.addSeparator()
        self._a_run_continuous = m_sim.addAction("Start Continuous Run")
        self._a_run_continuous.triggered.connect(self._action_toggle_continuous_run)
        a_speed = m_sim.addAction("Set Speed...")
        a_speed.triggered.connect(self._action_set_speed)
        m_sim.addSeparator()
        a_reset = m_sim.addAction("Reset Simulation")
        a_reset.triggered.connect(self._action_reset_simulation)

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
            self._cellspace_file_path = path  # ファイルパスを保存
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
                # 新しい統合関数を使用（確率情報付き）
                self._loaded_rule_files = [path]
                self._loaded_rule_array, self._loaded_rule_probs = load_multiple_transition_rules_with_probability([path])
                self._loaded_rule_ids = get_rule_ids_from_files([path])
                
                print(f"Loaded {len(self._loaded_rule_ids)} rules from {path}")
                print(f"Rule array shape: {self._loaded_rule_array.shape}")
                
                # ルールビューア用に一時的にTransitionRuleリストを作成
                temp_rules = load_transition_rules_yaml(path)
                self._rule_viewer = RuleViewerWindow(temp_rules, self._loaded_rule_probs)
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
                # 新しい統合関数を使用（確率情報付き）
                self._loaded_rule_files = [path]
                self._rule_file_paths = [path]  # ファイルパスを保存
                self._loaded_rule_array, self._loaded_rule_probs = load_multiple_transition_rules_with_probability([path])
                self._loaded_rule_ids = get_rule_ids_from_files([path])
                
                print(f"Loaded {len(self._loaded_rule_ids)} rules from {path}")
                print(f"Rule array shape: {self._loaded_rule_array.shape}")
                
                # ルールビューア用に一時的にTransitionRuleリストを作成
                temp_rules = load_transition_rules_yaml(path)
                self._rule_viewer = RuleViewerWindow(temp_rules, self._loaded_rule_probs)
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
                self._rule_file_paths.append(path)  # ファイルパスを保存
                
                # 全ファイルから統合配列を再生成（確率情報付き）
                self._loaded_rule_array, self._loaded_rule_probs = load_multiple_transition_rules_with_probability(self._loaded_rule_files)
                self._loaded_rule_ids = get_rule_ids_from_files(self._loaded_rule_files)
                
                print(f"Added rule file: {path}")
                print(f"Total rule array shape: {self._loaded_rule_array.shape}")
                
                # ルールビューア用に一時的にTransitionRuleリストを作成
                all_temp_rules = []
                for file_path in self._loaded_rule_files:
                    temp_rules = load_transition_rules_yaml(file_path)
                    all_temp_rules.extend(temp_rules)
                self._rule_viewer = RuleViewerWindow(all_temp_rules, self._loaded_rule_probs)
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

    def _action_load_special_events(self) -> None:
        """特殊イベントファイルを読み込む"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Special Events", filter="Python Files (*.py)")
        if not path:
            return
        
        try:
            # 特殊イベントを読み込み
            events = load_special_events_from_file(path)
            event_names = get_event_names_from_file(path)
            
            self._loaded_events = events
            self._event_names = event_names
            self._event_file_path = path  # ファイルパスを保存
            
            # 生座標配列（シミュレーション用）を保存
            self._event_array_raw = events.copy() if events is not None else None
            
            # オーバーレイ表示用の座標変換済み配列を作成
            if self._arr is not None:
                self._event_array = convert_events_to_array_coordinates(
                    events, self._offset_x, self._offset_y)
            
            # イベントオーバーレイを更新
            self._update_event_overlay()

            self._status.showMessage(
                f"Loaded {len(events)} special events from {path}")
            
            QtWidgets.QMessageBox.information(
                self, "Events Loaded", 
                f"Successfully loaded {len(events)} special events.\n"
                f"Use 'Event → Show Events' to display them on the cell space."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load special events:\n{str(e)}")

    def _toggle_events_visibility(self, checked: bool) -> None:
        """特殊イベントの表示/非表示を切り替え"""
        self._events_visible = checked
        self._update_event_overlay()
        
        if checked and len(self._loaded_events) > 0:
            self._status.showMessage(f"Showing {len(self._loaded_events)} special events")
        else:
            self._status.showMessage("Events hidden")

    def _action_clear_events(self) -> None:
        """読み込み済み特殊イベントをクリア"""
        self._loaded_events = []
        self._event_array = None
        self._event_array_raw = None
        self._event_names = []
        self._events_visible = False
        self._event_file_path = None
        self._act_show_events.setChecked(False)
        
        # イベントオーバーレイをクリア
        self._update_event_overlay()
        
        self._status.showMessage("Special events cleared")

    def _update_event_overlay(self) -> None:
        """特殊イベントオーバーレイを更新"""
        # 既存のオーバーレイアイテムをクリア
        for item in self._event_overlay_items:
            self._scene.removeItem(item)
        self._event_overlay_items.clear()
        
        # イベントが非表示または存在しない場合は終了
        if not self._events_visible or self._event_array is None or len(self._event_array) == 0:
            return
        
        # セル空間が読み込まれていない場合は終了
        if self._arr is None:
            return
        
        # 各イベントに対してオーバーレイアイテムを作成
        for i, event in enumerate(self._event_array):
            x, y = int(event[0]), int(event[1])
            
            # 配列範囲内チェック
            if 0 <= y < self._arr.shape[0] and 0 <= x < self._arr.shape[1]:
                # 円形のオーバーレイアイテムを作成
                overlay_item = QtWidgets.QGraphicsEllipseItem(x - 0.4, y - 0.4, 0.8, 0.8)
                
                # 半透明の赤色で表示
                pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 200))
                pen.setWidth(0.1)
                brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 100))
                overlay_item.setPen(pen)
                overlay_item.setBrush(brush)
                
                # ツールチップを設定（イベント名があれば表示）
                if i < len(self._event_names):
                    overlay_item.setToolTip(f"Event: {self._event_names[i]} at ({x}, {y})")
                else:
                    overlay_item.setToolTip(f"Special Event at ({x}, {y})")
                
                # シーンに追加
                self._scene.addItem(overlay_item)
                self._event_overlay_items.append(overlay_item)

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

    # ---- Simulation ----
    def _action_configure_simulation(self) -> None:
        """シミュレーション設定ダイアログを表示してBCA_Simulatorインスタンスを作成"""
        # 必要なファイルが読み込まれているかチェック
        if self._cellspace_file_path is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No cell space file loaded. Please load a cell space file first.")
            return
        
        if len(self._rule_file_paths) == 0:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No transition rules loaded. Please load rule files first.")
            return
        
        dialog = SimulationConfigDialog(
            self._sim_steps, self._sim_global_prob, self._sim_seed, self._sim_parallel_trials, self._sim_device, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self._sim_steps, self._sim_global_prob, self._sim_seed, self._sim_parallel_trials, self._sim_device = dialog.get_values()
            
            # BCA_Simulatorインスタンスを作成
            try:
                self._bca_simulator = BCA_Simulator(
                    cellspace_path=self._cellspace_file_path,
                    rule_paths=self._rule_file_paths,
                    device=self._sim_device,
                    spatial_event_filePath=self._event_file_path
                )
                
                # テンソル割り当て
                self._bca_simulator.Allocate_torch_Tensors_on_Device()
                
                # デバッグ情報を収集
                debug_info = self._collect_simulator_debug_info()
                
                self._status.showMessage(
                    f"Simulation configured: steps={self._sim_steps}, prob={self._sim_global_prob}, "
                    f"seed={self._sim_seed}, device={self._sim_device}, BCA_Simulator ready")
                
                QtWidgets.QMessageBox.information(
                    self, "Simulation Configured", 
                    f"BCA_Simulator instance created successfully!\n\n"
                    f"Cell space: {self._cellspace_file_path}\n"
                    f"Rules: {len(self._rule_file_paths)} files\n"
                    f"Events: {'Yes' if self._event_file_path else 'None'}\n"
                    f"Device: {self._sim_device}\n"
                    f"Steps: {self._sim_steps}\n"
                    f"Global Probability: {self._sim_global_prob}\n"
                    f"Seed: {self._sim_seed}\n\n"
                    f"Debug Info:\n{debug_info}")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to create BCA_Simulator instance:\n{str(e)}")
                self._bca_simulator = None

    def _collect_simulator_debug_info(self) -> str:
        """BCA_Simulatorインスタンスのデバッグ情報を収集"""
        if self._bca_simulator is None:
            return "No simulator instance"
        
        info_lines = []
        
        # 基本情報
        info_lines.append(f"Device: {self._bca_simulator.device}")
        
        # セル空間テンソル
        if hasattr(self._bca_simulator, 'cellspace_tensor') and self._bca_simulator.cellspace_tensor is not None:
            tensor = self._bca_simulator.cellspace_tensor
            info_lines.append(f"Cellspace tensor: {tensor.shape} {tensor.dtype} on {tensor.device}")
        else:
            info_lines.append("Cellspace tensor: Not allocated")
        
        # 遷移規則テンソル
        if hasattr(self._bca_simulator, 'rule_arrays_tensor') and self._bca_simulator.rule_arrays_tensor is not None:
            tensor = self._bca_simulator.rule_arrays_tensor
            info_lines.append(f"Rule arrays tensor: {tensor.shape} {tensor.dtype} on {tensor.device}")
        else:
            info_lines.append("Rule arrays tensor: Not allocated")
        
        # 確率テンソル
        if hasattr(self._bca_simulator, 'rule_probs_tensor') and self._bca_simulator.rule_probs_tensor is not None:
            tensor = self._bca_simulator.rule_probs_tensor
            info_lines.append(f"Rule probs tensor: {tensor.shape} {tensor.dtype} on {tensor.device}")
        else:
            info_lines.append("Rule probs tensor: Not allocated")
        
        # 特殊イベントテンソル
        if hasattr(self._bca_simulator, 'spatial_event_arrays_tensor') and self._bca_simulator.spatial_event_arrays_tensor is not None:
            tensor = self._bca_simulator.spatial_event_arrays_tensor
            info_lines.append(f"Event arrays tensor: {tensor.shape} {tensor.dtype} on {tensor.device}")
        else:
            info_lines.append("Event arrays tensor: None")
        
        return "\n".join(info_lines)

    def _action_step_simulation(self) -> None:
        """1ステップだけシミュレーションを実行"""
        if self._bca_simulator is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No BCA_Simulator instance. Please configure simulation first.")
            return
        
        try:
            # 初回実行時のみ平行試行数を設定
            if self._current_step == 0:
                self._bca_simulator.set_ParallelTrial(self._sim_parallel_trials)
            
            # 1ステップだけ実行
            self._bca_simulator.run_steps(
                steps=1,
                global_prob=self._sim_global_prob,
                seed=self._sim_seed + self._current_step,  # ステップごとに異なるシード
                debug=False,
                debug_per_trial=False
            )
            
            self._current_step += 1
            
            # シミュレーション結果をGUIに反映
            # 最初のトライアルの結果を表示用に取得
            result_tensor = self._bca_simulator.TCHW[0, 0]  # [H, W]
            result_array = result_tensor.cpu().numpy()
            
            # デバッグ: テンソル形状を確認
            original_shape = self._bca_simulator.cellspace_tensor.shape
            result_shape = result_array.shape
            print(f"Debug: Original cellspace shape: {original_shape}")
            print(f"Debug: Result array shape: {result_shape}")
            print(f"Debug: Shape match: {original_shape == result_shape}")
            
            # 表示を更新
            self.set_array(result_array)
            
            self._status.showMessage(
                f"Step {self._current_step} completed (prob={self._sim_global_prob}, "
                f"seed={self._sim_seed + self._current_step - 1}, device={self._sim_device})")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Simulation Error", f"Failed to run simulation step:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _action_toggle_continuous_run(self) -> None:
        """連続実行の開始・停止を切り替え"""
        if self._bca_simulator is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No BCA_Simulator instance. Please configure simulation first.")
            return
        
        if self._is_running:
            # 停止
            self._timer.stop()
            self._is_running = False
            self._a_run_continuous.setText("Start Continuous Run")
            self._status.showMessage(f"Continuous run stopped at step {self._current_step}")
        else:
            # 開始
            # 初回実行時のみ平行試行数を設定
            if self._current_step == 0:
                self._bca_simulator.set_ParallelTrial(self._sim_parallel_trials)
            
            self._timer.start(self._step_interval)
            self._is_running = True
            self._a_run_continuous.setText("Stop Continuous Run")
            self._status.showMessage(f"Continuous run started (interval: {self._step_interval}ms)")
    
    def _action_set_speed(self) -> None:
        """実行速度を設定"""
        interval, ok = QtWidgets.QInputDialog.getInt(
            self, "Set Speed", "Step interval (milliseconds):", 
            self._step_interval, 50, 5000, 50)
        
        if ok:
            self._step_interval = interval
            if self._is_running:
                self._timer.setInterval(self._step_interval)
            self._status.showMessage(f"Step interval set to {self._step_interval}ms")
    
    def _timer_step(self) -> None:
        """タイマーによる自動ステップ実行"""
        try:
            # 1ステップだけ実行
            self._bca_simulator.run_steps(
                steps=1,
                global_prob=self._sim_global_prob,
                seed=self._sim_seed + self._current_step,
                debug=False,
                debug_per_trial=False
            )
            
            self._current_step += 1
            
            # シミュレーション結果をGUIに反映
            result_tensor = self._bca_simulator.TCHW[0, 0]  # [H, W]
            result_array = result_tensor.cpu().numpy()
            
            # デバッグ: テンソル形状を確認
            original_shape = self._bca_simulator.cellspace_tensor.shape
            result_shape = result_array.shape
            print(f"Debug: Original cellspace shape: {original_shape}")
            print(f"Debug: Result array shape: {result_shape}")
            print(f"Debug: Shape match: {original_shape == result_shape}")
            
            # 表示を更新
            self.set_array(result_array)
            
            self._status.showMessage(
                f"Step {self._current_step} (continuous, interval: {self._step_interval}ms)")
                
        except Exception as e:
            # エラー時は自動停止
            self._timer.stop()
            self._is_running = False
            self._a_run_continuous.setText("Start Continuous Run")
            QtWidgets.QMessageBox.critical(
                self, "Simulation Error", f"Failed to run simulation step:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _action_reset_simulation(self) -> None:
        """シミュレーション状態をリセット"""
        if self._bca_simulator is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No BCA_Simulator instance. Please configure simulation first.")
            return
        
        try:
            # セル空間を初期状態に戻す
            self._bca_simulator.cellspace_tensor = torch.from_numpy(self._bca_simulator.cellspace).to(self._bca_simulator.device)
            
            # ステップカウンタをリセット
            self._current_step = 0
            
            # 初期状態を表示
            initial_array = self._bca_simulator.cellspace_tensor.cpu().numpy()
            self.set_array(initial_array)
            
            # 連続実行中の場合は停止
            if self._is_running:
                self._timer.stop()
                self._is_running = False
                self._a_run_continuous.setText("Start Continuous Run")
            
            self._status.showMessage("Simulation reset to initial state")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Reset Error", f"Failed to reset simulation:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _load_default_files(self) -> None:
        """デフォルトファイルを自動読み込み"""
        import os
        
        # デフォルトファイルパス
        default_cellspace = "SampleCP/test.yaml"
        default_rules = "src/PyBCA/rule/base-rule.yaml"
        
        try:
            # セル空間の読み込み
            if os.path.exists(default_cellspace):
                arr = load_cell_space_yaml_to_numpy(default_cellspace)
                self.set_array(arr)
                self._cellspace_file_path = default_cellspace  # ファイルパスを保存
                self._status.showMessage(f"Auto-loaded: {default_cellspace}")
            
            # 遷移規則の読み込み
            if os.path.exists(default_rules):
                self._loaded_rule_files = [default_rules]
                self._rule_file_paths = [default_rules]  # ファイルパスを保存
                self._loaded_rule_array, self._loaded_rule_probs = load_multiple_transition_rules_with_probability([default_rules])
                self._loaded_rule_ids = get_rule_ids_from_files([default_rules])
                
                # ルールビューア用に一時的にTransitionRuleリストを作成
                temp_rules = load_transition_rules_yaml(default_rules)
                self._rule_viewer = RuleViewerWindow(temp_rules, self._loaded_rule_probs)
                
                self._status.showMessage(f"Auto-loaded: {default_cellspace} and {default_rules}")
                
        except Exception as e:
            print(f"Failed to load default files: {e}")
            self._status.showMessage("Failed to load default files")

    # ---- Edit Mode Actions ----
    def _toggle_edit_mode(self) -> None:
        """編集モードの切り替え"""
        self._edit_mode = self._act_edit_mode.isChecked()
        
        if self._edit_mode:
            self._view.setCursor(QtCore.Qt.CrossCursor)
            self._status.showMessage(f"Edit Mode ON - Click to paint value {self._edit_brush_value}")
        else:
            self._view.setCursor(QtCore.Qt.ArrowCursor)
            self._status.showMessage("Edit Mode OFF")
    
    def _action_new_cellspace(self) -> None:
        """新しいセル空間を作成"""
        dialog = NewCellSpaceDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            width, height = dialog.get_size()
            
            # 新しい0で埋められたセル空間を作成
            new_array = np.zeros((height, width), dtype=np.int8)
            
            # 現在のセル空間を置き換え
            self.set_array(new_array)
            
            # ステータス更新
            self._status.showMessage(f"New cell space created: {width}x{height}")
            
            # 編集モードを有効にする
            self._act_edit_mode.setChecked(True)
            self._toggle_edit_mode()
    
    def _action_clear_all_cells(self) -> None:
        """全セルをクリア（値0に設定）"""
        if self._arr is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No cell space loaded.")
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All Cells", 
            "Are you sure you want to clear all cells to 0?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        
        if reply == QtWidgets.QMessageBox.Yes:
            self._arr.fill(0)
            self._update_display()
            self._status.showMessage("All cells cleared to 0")
    
    def _action_fill_pattern(self) -> None:
        """パターン塗りつぶしダイアログ"""
        if self._arr is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No cell space loaded.")
            return
        
        dialog = FillPatternDialog(self._edit_brush_value, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self._edit_brush_value = dialog.get_value()
            self._status.showMessage(f"Brush value set to {self._edit_brush_value}")

    def _update_display(self) -> None:
        """表示を更新する"""
        if self._arr is None:
            return
        
        qimg = array_to_qimage(self._arr)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self._pix.setPixmap(pixmap)
        
        # シーンのサイズを更新
        self._scene.setSceneRect(pixmap.rect())
        
        # グリッドを再描画
        self._rebuild_grid()

    # ---- events ----
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseMove and self._arr is not None:
            # マウス座標→シーン座標→配列インデックス
            pos = self._view.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            h, w = self._arr.shape
            if 0 <= y < h and 0 <= x < w:
                val = self._arr[y, x]
                self._status.showMessage(f"({x},{y}) = {val}")
            else:
                self._status.showMessage("")
        elif event.type() == QtCore.QEvent.MouseButtonPress and self._edit_mode and self._arr is not None:
            # 編集モードでのマウスクリック処理
            if event.button() == QtCore.Qt.LeftButton:
                pos = self._view.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())
                h, w = self._arr.shape
                if 0 <= y < h and 0 <= x < w:
                    old_val = self._arr[y, x]
                    self._arr[y, x] = self._edit_brush_value
                    self._update_display()
                    self._status.showMessage(f"Cell ({x},{y}) changed: {old_val} → {self._edit_brush_value}")
                    return True  # イベントを消費
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


class NewCellSpaceDialog(QtWidgets.QDialog):
    """新しいセル空間作成ダイアログ"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Cell Space")
        self.setModal(True)
        
        layout = QtWidgets.QFormLayout(self)
        
        # 幅設定
        self._width_spin = QtWidgets.QSpinBox()
        self._width_spin.setRange(1, 10000)
        self._width_spin.setValue(100)
        layout.addRow("Width:", self._width_spin)
        
        # 高さ設定
        self._height_spin = QtWidgets.QSpinBox()
        self._height_spin.setRange(1, 10000)
        self._height_spin.setValue(100)
        layout.addRow("Height:", self._height_spin)
        
        # ボタン
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.resize(250, 120)
    
    def get_size(self):
        return self._width_spin.value(), self._height_spin.value()


class FillPatternDialog(QtWidgets.QDialog):
    """セル値設定ダイアログ"""
    def __init__(self, current_value, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fill Pattern")
        self.setModal(True)
        
        layout = QtWidgets.QFormLayout(self)
        
        # セル値設定
        self._value_spin = QtWidgets.QSpinBox()
        self._value_spin.setRange(-128, 127)  # int8の範囲
        self._value_spin.setValue(current_value)
        layout.addRow("Cell Value:", self._value_spin)
        
        # ボタン
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.resize(200, 100)
    
    def get_value(self):
        return self._value_spin.value()


class RuleViewerWindow(QtWidgets.QMainWindow):
    """
    遷移規則ビューア。prev→nextの変化を左右並列表示し、矢印ボタンで規則を切り替え。
    """
    def __init__(self, rules: List[TransitionRule], probabilities: np.ndarray = None):
        super().__init__()
        self.setWindowTitle("PyBCA Rule Viewer")
        self._rules = rules
        self._probabilities = probabilities if probabilities is not None else np.ones(len(rules))
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
            prob_text = f" (p={self._probabilities[i]:.3f})" if self._probabilities is not None else ""
            self._rule_selector.addItem(f"Rule {rule.rule_id}{prob_text}")
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
        prob_text = f", Probability: {self._probabilities[self._current_index]:.3f}" if self._probabilities is not None else ""
        self._rule_info.setText(f"Rule ID: {rule.rule_id} ({self._current_index + 1}/{len(self._rules)}){prob_text}")
        
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
        prob_text = f", prob={self._probabilities[self._current_index]:.3f}" if self._probabilities is not None else ""
        self._status.showMessage(f"Rule {rule.rule_id}: prev={rule.prev_pattern.shape}, next={rule.next_pattern.shape}{prob_text}")
    
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


class SimulationConfigDialog(QtWidgets.QDialog):
    """シミュレーション設定ダイアログ"""
    def __init__(self, steps, global_prob, seed, parallel_trials, device, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Configuration")
        self.setModal(True)
        
        layout = QtWidgets.QFormLayout(self)
        
        # Steps
        self._steps_spin = QtWidgets.QSpinBox()
        self._steps_spin.setRange(1, 10000)
        self._steps_spin.setValue(steps)
        layout.addRow("Steps:", self._steps_spin)
        
        # Global Probability
        self._prob_spin = QtWidgets.QDoubleSpinBox()
        self._prob_spin.setRange(0.0, 1.0)
        self._prob_spin.setSingleStep(0.1)
        self._prob_spin.setDecimals(2)
        self._prob_spin.setValue(global_prob)
        layout.addRow("Global Probability:", self._prob_spin)
        
        # Seed
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(1, 999999)
        self._seed_spin.setValue(seed)
        layout.addRow("Seed:", self._seed_spin)
        
        # Parallel Trials
        self._trials_spin = QtWidgets.QSpinBox()
        self._trials_spin.setRange(1, 10)
        self._trials_spin.setValue(parallel_trials)
        layout.addRow("Parallel Trials:", self._trials_spin)
        
        # Device
        self._device_combo = QtWidgets.QComboBox()
        self._device_combo.addItems(["cpu", "cuda"])
        self._device_combo.setCurrentText(device)
        layout.addRow("Device:", self._device_combo)
        
        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.resize(300, 180)
    
    def get_values(self):
        return (
            self._steps_spin.value(),
            self._prob_spin.value(),
            self._seed_spin.value(),
            self._trials_spin.value(),
            self._device_combo.currentText()
        )


def main(argv=None):
    app = QtWidgets.QApplication(argv or sys.argv)
    # 初期状態は空で起動
    win = CellSpaceWindow(arr=None)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
