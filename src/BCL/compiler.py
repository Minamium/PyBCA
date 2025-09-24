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
        self._coord_syms: dict[str, tuple[int, int]] = {}

    def read_file(self, path: str) -> str:
        text = Path(path).read_text(encoding="utf-8")
        self.source_text = text
        return text

    # ========= 式評価 =========
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

    # ========= 前処理：element 定義と展開 =========
    def _preprocess_elements(self, src_text: str) -> list[str]:
        """
        element <Name>(Param) { ... }
        place.<Name>( Inst, Param[ex,ey] )

        を “coord.define / place.*” のフラット行列に展開して返す。
        """
        lines = src_text.splitlines()

        re_elem_start = re.compile(
            r"^\s*element\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{\s*$"
        )
        re_elem_end   = re.compile(r"^\s*\}\s*$")
        re_place_elem = re.compile(
            r"^\s*place\.([A-Za-z_][A-Za-z0-9_]*)\s*"
            r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]\s*\)\s*$"
        )

        elements: dict[str, dict] = {}
        top_lines: list[str] = []

        # 1) element ブロック収集
        i, N = 0, len(lines)
        while i < N:
            raw = lines[i]
            line = raw.split("#", 1)[0].strip()
            m = re_elem_start.match(line)
            if m:
                name = m.group(1)
                param = m.group(2)
                body: list[str] = []
                i += 1
                found_end = False
                while i < N:
                    raw2 = lines[i]
                    if re_elem_end.match(raw2.split("#", 1)[0].strip()):
                        found_end = True
                        break
                    body.append(raw2)
                    i += 1
                if not found_end:
                    raise SyntaxError(f"element '{name}': missing closing '}}'")
                elements[name] = {"param": param, "body": body}
                i += 1  # '}' の次へ
                continue
            else:
                top_lines.append(raw)
                i += 1

        # ローカル名抽出（coord.define の第1引数）
        re_local_def = re.compile(r"^\s*coord\.define\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")
        def _replace_ref(line: str, sym: str, repl_prefix: str) -> str:
            line = re.sub(rf"\b{re.escape(sym)}\.x\b", repl_prefix + ".x", line)
            line = re.sub(rf"\b{re.escape(sym)}\.y\b", repl_prefix + ".y", line)
            return line

        # 2) place.Element を展開
        expanded: list[str] = []
        for raw in top_lines:
            stripped = raw.split("#", 1)[0].strip()
            m = re_place_elem.match(stripped)
            if not m:
                expanded.append(raw)
                continue

            elem_name = m.group(1)
            inst_name = m.group(2)
            bind_name = m.group(3)
            ex, ey    = m.group(4).strip(), m.group(5).strip()

            if elem_name not in elements:
                raise SyntaxError(f"Undefined element: {elem_name}")
            declared_param: str = elements[elem_name]["param"]
            body: list[str] = elements[elem_name]["body"]

            # パラメータ名一致チェック（v0は1つだけ、名前で対応付け）
            if bind_name != declared_param:
                raise SyntaxError(
                    f"element '{elem_name}': expected param '{declared_param}', got '{bind_name}'"
                )

            # Param 束縛を coord に起こす（__Inst__Param）
            param_alias = f"__{inst_name}__{declared_param}"
            expanded.append(f"coord.define({param_alias}, {ex}, {ey})")

            # 本文中のローカル座標名を収集（後で参照置換用）
            locals_in_body: set[str] = set()
            for b in body:
                mm = re_local_def.match(b.split("#", 1)[0].strip())
                if mm:
                    locals_in_body.add(mm.group(1))
            locals_sorted = sorted(list(locals_in_body), key=len, reverse=True)

            # 本文展開：
            #  - coord.define(local, ...) の local を Inst__local にリネーム
            #  - Param.x/y         → __Inst__Param.x/y
            #  - local.x/y 参照    → Inst__local.x/y
            for b in body:
                line = b
                # local 定義の第1引数を Inst__local に
                line_no_comment = line.split("#", 1)[0]
                if re_local_def.match(line_no_comment.strip()):
                    line = re.sub(
                        r"(^\s*coord\.define\s*\(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*,)",
                        lambda mo: mo.group(1) + f"{inst_name}__{mo.group(2)}" + mo.group(3),
                        line,
                        count=1,
                    )
                # Param 参照置換
                line = _replace_ref(line, declared_param, param_alias)
                # ローカル参照置換
                for loc in locals_sorted:
                    line = _replace_ref(line, loc, f"{inst_name}__{loc}")

                expanded.append(line)

        return expanded

    # ========= 構文解析（element 前処理 → 既存プレーン構文） =========
    def parse(self, source_text: str | None = None) -> None:
        """
        最小構文：
          - coord.define(name,x,y)
          - place.signal_line(ax, ay)   # ax, ay は整数 or name.{x|y}[±N]
          - place.token(ax, ay)
          - place.recycle_bin(ax, ay)   # value=-1
          - element <Name>(Param) { ... }                              # ← 追加
          - place.<Name>(Inst, Param[expr_x, expr_y])                  # ← 追加
        """
        if source_text is not None:
            self.source_text = source_text
        if not self.source_text:
            raise ValueError("source is empty")

        # 1) element を前処理で展開 → フラットな行列へ
        flat_lines = self._preprocess_elements(self.source_text)

        # 2) 既存のプレーン構文をパース
        self._placements.clear()
        self._coord_syms.clear()

        re_def     = re.compile(
            r"^\s*coord\.define\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*$"
        )
        re_sig     = re.compile(r"^\s*place\.signal_line\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
        re_token   = re.compile(r"^\s*place\.token\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
        re_recycle = re.compile(r"^\s*place\.recycle_bin\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")

        for raw in flat_lines:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            m = re_def.match(line)
            if m:
                name, xexpr, yexpr = m.group(1), m.group(2), m.group(3)
                x = self._eval_value(xexpr)
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

            m = re_recycle.match(line)
            if m:
                ax, ay = m.group(1), m.group(2)
                x = self._eval_value(ax)
                y = self._eval_value(ay)
                self._placements.append({"coord": {"x": x, "y": y}, "value": -1})
                continue

            raise SyntaxError(f"Unsupported line: {raw}")

    # ========= 出力 =========
    def lower_to_ir(self) -> CompileResult:
        """“座標と値の列” をそのまま YAML のトップにリストとして出す。"""
        return CompileResult(yaml_dict=list(self._placements))

    def write_yaml(self, result: CompileResult, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(result.yaml_dict, f, sort_keys=False, allow_unicode=True)

    def compile_file(self, input_path: str, output_path: str) -> None:
        _ = self.read_file(input_path)
        self.parse()
        ir = self.lower_to_ir()
        self.write_yaml(ir, output_path)
