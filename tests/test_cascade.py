"""Unit tests for cascade corrections."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from su2_analysis.stage5_pitch_kinematics.core.services.cascade_correction_service import (
    weinig_factor, carter_deviation,
)


def test_weinig_factor_range():
    for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
        K = weinig_factor(sigma)
        assert 0 < K <= 1.0, f"Weinig factor should be in (0, 1], got {K} for σ={sigma}"


def test_weinig_factor_approaches_one_at_low_solidity():
    K_low  = weinig_factor(0.5)
    K_high = weinig_factor(2.5)
    assert K_low > K_high, "Lower solidity should give larger Weinig factor (less cascade interference)"


def test_carter_deviation_positive():
    delta = carter_deviation(20.0, 1.0)
    assert delta > 0, "Carter deviation should be positive for positive camber"


def test_carter_deviation_decreases_with_solidity():
    d1 = carter_deviation(20.0, 0.5)
    d2 = carter_deviation(20.0, 2.0)
    assert d1 > d2, "Carter deviation decreases with solidity (1/√σ)"
