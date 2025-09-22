# src/BCL/compiler.py
from __future__ import annotations
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class CompileResult:
    yaml_dict: dict

class BCLCompiler:
    def __init__(self) -> None:
        self.source_text: str | None = None
        self.ast: dict | None = None

    def read_file(self, path: str) -> str:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        self.source_text = text
        return text

    def parse(self, source_text: str | None = None) -> None:
        """
        最小ダミーのパーサ。
        - いまは source_text を保持するだけ（後で構文解析を実装）
        """
        if source_text is not None:
            self.source_text = source_text
        if self.source_text is None:
            raise ValueError("source is empty")
        # TODO: ここに字句解析→構文解析→AST生成を実装
        self.ast = {"_raw": self.source_text}

    def lower_to_ir(self) -> CompileResult:
        """
        最小のIR雛形を返す。
        PyBCA 側の読み込み仕様に合わせて後で keys を揃える。
        """
        if self.ast is None:
            raise RuntimeError("parse() を先に呼んでください")
        # まずは最小の YAML 雛形（後で本実装に差し替え）
        y = {
            "version": "0.1",
            "meta": {
                "generator": "BCLCompiler",
                "note": "WIP skeleton; replace with real IR later",
            },
            # ↓ ここに最終的な PyBCA 用の cellspace / events / tokens 等を入れる
            "grid": {"width": 0, "height": 0},
            "wires": [],
            "events": [],
            "initial_tokens": [],
        }
        return CompileResult(yaml_dict=y)

    def write_yaml(self, result: CompileResult, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(result.yaml_dict, f, sort_keys=False, allow_unicode=True)

    # ワンショット（CLI から使う）
    def compile_file(self, input_path: str, output_path: str) -> None:
        _ = self.read_file(input_path)
        self.parse()
        ir = self.lower_to_ir()
        self.write_yaml(ir, output_path)
