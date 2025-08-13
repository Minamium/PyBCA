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

from lib import load_cell_space_yaml_to_numpy

# ========= 配列 -> QImage（高速LUT） =========

def build_palette(states: List[int]) -> Dict[int, Tuple[int, int, int]]:
    """
    状態→色の簡易パレット。
    既定: 0=黒, 1=白, 2=赤。他の値はランダム色。
    """
    pal: Dict[int, Tuple[int, int, int]] = {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (220, 40, 40),
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
    def __init__(self, arr: Optional[np.ndarray] = None,
                 palette: Optional[Dict[int, Tuple[int, int, int]]] = None):
        super().__init__()
        self.setWindowTitle("PyBCA CellSpace Viewer")
        self._palette = palette
        self._arr: Optional[np.ndarray] = None

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
        self._arr = arr
        qimg = array_to_qimage(arr, self._palette)
        self._pix.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self._view.setSceneRect(self._pix.boundingRect())
        self._view.fitInView(self._pix, QtCore.Qt.KeepAspectRatio)
        self._rebuild_grid()

    # ---- UI building ----
    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # File
        m_file = menubar.addMenu("&File")
        a_open = m_file.addAction("Open YAML…")
        a_open.triggered.connect(self._action_open_yaml)
        a_quit = m_file.addAction("Quit")
        a_quit.triggered.connect(self.close)

        # View
        m_view = menubar.addMenu("&View")
        self._act_grid = m_view.addAction("Show Grid")
        self._act_grid.setCheckable(True)
        self._act_grid.setChecked(True)
        self._act_grid.triggered.connect(self._toggle_grid)

    # ---- actions ----
    def _action_open_yaml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open CellSpace YAML", filter="YAML Files (*.yaml *.yml)")
        if not path:
            return
        arr = load_cell_space_yaml_to_numpy(path)
        self.set_array(arr)
        self._status.showMessage(f"Loaded: {path}  size={arr.shape}")

    def _toggle_grid(self, checked: bool) -> None:
        self._grid_visible = checked
        self._grid_item.setVisible(checked)

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

    def _rebuild_grid(self) -> None:
        """現在の配列サイズと表示倍率からグリッドを組み直す。"""
        if self._arr is None or not self._grid_visible:
            self._grid_item.setPath(QtGui.QPainterPath())
            return
        h, w = self._arr.shape
        path = QtGui.QPainterPath()
        # ピクセル境界に合わせた格子。倍率が低いときは間引き
        # ここでは単純に 1px あたりのスケールで閾値を決める
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


def main(argv=None):
    app = QtWidgets.QApplication(argv or sys.argv)
    arr = None
    if len(sys.argv) > 1:
        arr = load_cell_space_yaml_to_numpy(sys.argv[1])
    win = CellSpaceWindow(arr=arr)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
