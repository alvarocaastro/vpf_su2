"""Unit tests for pitch map computation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from su2_analysis.stage2_su2_simulations.pitch_map import compute_pitch_map


def _make_polar(alpha_opt: float = 4.0) -> pd.DataFrame:
    alphas = np.linspace(-5, 20, 26)
    cl = np.clip(0.1 * (alphas + 2), 0, 1.4)
    cd = 0.01 + 0.001 * alphas**2
    return pd.DataFrame({
        "alpha":     alphas,
        "cl":        cl,
        "cd":        cd,
        "ld":        cl / cd,
        "converged": True,
    })


def test_pitch_map_has_all_keys():
    polars = {
        "takeoff_root": _make_polar(),
        "cruise_mid":   _make_polar(),
        "descent_tip":  _make_polar(),
    }
    pm = compute_pitch_map(polars)
    assert set(pm["condition"]) >= {"takeoff", "cruise", "descent"}


def test_pitch_map_alpha_opt_in_range():
    polars = {"cruise_mid": _make_polar()}
    pm = compute_pitch_map(polars)
    assert all(pm["alpha_opt"] >= 1.0), "alpha_opt should be >= 1° (second peak)"
