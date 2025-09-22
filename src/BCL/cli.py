# src/BCL/cli.py
from __future__ import annotations
import argparse
from BCL.compiler import BCLCompiler

def main() -> None:
    p = argparse.ArgumentParser(prog="bcl", description="BCL → YAML/npz compiler", usage="bcl INPUT.bcl -o OUTPUT.yaml")
    p.add_argument("input", help="input .bcl file")
    p.add_argument("-o", "--output", required=True, help="output YAML or npz path")
    args = p.parse_args()

    comp = BCLCompiler()
    comp.compile_file(args.input, args.output)
    print(f"[bcl] wrote YAML: {args.output}")
