# src/BCL/compiler.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import re
import yaml

@dataclass
class CompileResult:
    yaml_dict: dict | list  # 今回はトップがリスト

class BCLCompiler:
    def __init__(self) -> None:
        self.source_text: str | None = None
        # 収集結果（順序保持）
        self._placements: list[dict] = []
        self._grid_decl: tuple[int, int] | None = None  # (W, H) 明示グリッド（今回は参照のみ）
        self._coord_syms: dict[str, tuple[int, int]] = {}

    def read_file(self, path: str) -> str:
        text = Path(path).read_text(encoding="utf-8")
        self.source_text = text
        return text

    def _eval_value(self, expr: str) -> int:
        """
        expr は以下のいずれか:
          - 整数:         "-3", "10"
          - シンボル軸:   "name.x", "name.y"
          - 軸+オフセット: "name.x+2", "name.y-1"
        空白は無視。未定義なら SyntaxError。
        """
        s = expr.replace(" ", "")
        # 整数
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        # name.(x|y)([+-]\d+)?
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\.(x|y)([+-]\d+)?", s)
        if not m:
            raise SyntaxError(f"Invalid coordinate expression: {expr}")
        name, axis, off = m.group(1), m.group(2), m.group(3)
        if name not in self._coord_syms:
            raise SyntaxError(f"Undefined coord: {name}")
        base = self._coord_syms[name][0 if axis == "x" else 1]
        delta = int(off) if off else 0
        return int(base + delta)


    def parse(self, source_text: str | None = None) -> None:
        """
        最小構文：
          - grid WxH
          - coord.define(name,x,y)
          - place.signal_line(ax, ay)   # ax, ay は整数 or name.{x|y}[±N]
          - place.token(ax, ay)
        """
        if source_text is not None:
            self.source_text = source_text
        if not self.source_text:
            raise ValueError("source is empty")

        self._placements.clear()
        self._coord_syms.clear()
        self._grid_decl = None

        re_grid   = re.compile(r"^\s*grid\s+(-?\d+)[xX](-?\d+)\s*$")
        re_def    = re.compile(
            r"^\s*coord\.define\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*$"
        )
        re_sig    = re.compile(r"^\s*place\.signal_line\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
        re_token  = re.compile(r"^\s*place\.token\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")

        for raw in self.source_text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            m = re_grid.match(line)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
                self._grid_decl = (w, h)
                continue

            m = re_def.match(line)
            if m:
                name, xexpr, yexpr = m.group(1), m.group(2), m.group(3)
                x = self._eval_value(xexpr)   # ← 式を評価（整数 or name.x±k / name.y±k）
                y = self._eval_value(yexpr)
                self._coord_syms[name] = (x, y)
                continue

            m = re_sig.match(line)
            if m:
                ax, ay = m.group(1), m.group(2)
                x = self._eval_value(ax)
                y = self._eval_value(ay)
                self._placements.append({"coord": {"x": x, "y": y}, "value": 1})
                continue

            m = re_token.match(line)
            if m:
                ax, ay = m.group(1), m.group(2)
                x = self._eval_value(ax)
                y = self._eval_value(ay)
                self._placements.append({"coord": {"x": x, "y": y}, "value": 2})
                continue

            raise SyntaxError(f"Unsupported line: {raw}")

    def lower_to_ir(self) -> CompileResult:
        """
        今回は “座標と値の列” をそのまま YAML のトップにリストとして出す。
        グリッド宣言はまだメタ扱い（必要なら dict の先頭に meta を追加する）。
        """
        # 仕様どおり、トップが配列のYAMLを出す
        return CompileResult(yaml_dict=list(self._placements))

    def write_yaml(self, result: CompileResult, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            # sort_keys=False でキー順を固定、リストトップ対応
            yaml.safe_dump(result.yaml_dict, f, sort_keys=False, allow_unicode=True)

    def compile_file(self, input_path: str, output_path: str) -> None:
        _ = self.read_file(input_path)
        self.parse()
        ir = self.lower_to_ir()
        self.write_yaml(ir, output_path)
