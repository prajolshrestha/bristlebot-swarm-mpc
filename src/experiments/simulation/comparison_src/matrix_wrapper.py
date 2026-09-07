#!/usr/bin/env python3
"""Runs one obstacle-matrix simulation and files its output."""
import os
import sys
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
DRIVER = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "behaviors_src",
                                      "steady_state_matrix.py"))

os.environ.setdefault("MATRIX_OUT_DIR",
                      os.path.join(SCRIPT_DIR, "..", "results", "csv", "matrix"))
os.environ.setdefault("MATRIX_RAW_DIR",
                      os.path.join(SCRIPT_DIR, "..", "results", "raw", "matrix_npz"))


def load_driver():
    if not os.path.exists(DRIVER):
        sys.exit(f"matrix driver not found: {DRIVER}")
    spec = importlib.util.spec_from_file_location("steady_state_matrix", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["steady_state_matrix"] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    load_driver().main()
