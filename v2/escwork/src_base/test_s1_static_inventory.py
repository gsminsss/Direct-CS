#!/usr/bin/env python3
"""Static arithmetic regression independent of Stim availability."""
from pathlib import Path
import importlib.util

p = Path(__file__).resolve().parents[1] / "stim_pipeline" / "zxz" / "hirano_zxz_generator.py"
spec = importlib.util.spec_from_file_location("hz", p)
hz = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = hz
spec.loader.exec_module(hz)


def test_canonical_inventory_math() -> None:
    terminal = dict(hz.CANONICAL_PRE_READOUT)
    terminal["MX"] += hz.CANONICAL_TERMINAL_MEASUREMENTS
    loc, singles, pairs = hz._fault_inventory(terminal)
    assert loc == 797
    assert singles == 7279
    assert pairs == 26_439_666


def test_expected_raw_x_counts() -> None:
    terminal = dict(hz.CANONICAL_PRE_READOUT)
    terminal["MX"] += 25
    raw = {k: terminal[k] + hz.IDEAL_STEANE_INIT["X"].get(k, 0) for k in terminal}
    assert raw == {"CX": 474, "RX": 91, "RZ": 80, "MX": 95, "MZ": 75}


def test_expected_raw_z_counts() -> None:
    terminal = dict(hz.CANONICAL_PRE_READOUT)
    terminal["MZ"] += 25
    raw = {k: terminal[k] + hz.IDEAL_STEANE_INIT["Z"].get(k, 0) for k in terminal}
    assert raw == {"CX": 474, "RX": 90, "RZ": 81, "MX": 70, "MZ": 100}


if __name__ == "__main__":
    test_canonical_inventory_math()
    test_expected_raw_x_counts()
    test_expected_raw_z_counts()
    print("PASS: S1 static canonical inventory")
