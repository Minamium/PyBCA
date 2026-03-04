"""Compatibility runner for the legacy test script."""

from runpy import run_module


if __name__ == "__main__":
    run_module("PyBCA._legacy.TEST", run_name="__main__")
