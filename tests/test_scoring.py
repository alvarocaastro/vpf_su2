"""Unit tests for Stage 1 airfoil scoring."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from su2_analysis.stage1_airfoil_selection.scoring import (
    score_airfoils,
    _second_peak_ld,
    _stall_margin,
    _cl_max,
)


def _make_polar(cl_values: list, cd: float = 0.01) -> pd.DataFrame:
    alphas = np.linspace(-5, 20, len(cl_values))
    ld     = [cl / cd for cl in cl_values]
    return pd.DataFrame({
        "alpha":     alphas,
        "cl":        cl_values,
        "cd":        [cd] * len(cl_values),
        "ld":        ld,
        "converged": [True] * len(cl_values),
    })


def test_second_peak_ld_basic():
    polar = _make_polar([0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 0.9, 0.5])
    ld    = _second_peak_ld(polar)
    assert ld > 0, "CL/CD peak should be positive"


def test_stall_margin_positive():
    polar = _make_polar([0.3, 0.5, 0.7, 0.9, 1.1, 1.0, 0.8])
    sm    = _stall_margin(polar)
    assert sm >= 0 or np.isnan(sm), "Stall margin should be non-negative"


def test_score_airfoils_ranking():
    polars = {
        "naca_0012":   _make_polar([0.3, 0.5, 0.7, 0.9, 1.0], cd=0.015),
        "naca_65-410": _make_polar([0.4, 0.7, 1.0, 1.2, 1.3], cd=0.008),
    }
    ranking = score_airfoils(polars)
    assert ranking.iloc[0]["airfoil"] == "naca_65-410", "Higher CL/CD airfoil should rank first"


def test_score_airfoils_all_columns():
    polars = {"foil_a": _make_polar([0.5, 0.8, 1.0, 0.9])}
    ranking = score_airfoils(polars)
    for col in ("airfoil", "ld_2nd_peak", "alpha_opt", "cl_max", "stall_margin", "score", "rank"):
        assert col in ranking.columns, f"Missing column: {col}"
