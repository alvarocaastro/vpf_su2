"""Unit tests for ISA atmosphere model."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from su2_analysis.shared.atmosphere import (
    isa_temperature, isa_density, isa_pressure,
    reynolds_number, speed_of_sound,
    blade_velocity, relative_velocity,
)


def test_sea_level_temperature():
    T = isa_temperature(0.0)
    assert abs(T - 288.15) < 0.01, f"ISA SL temperature should be 288.15 K, got {T}"


def test_temperature_decreases_with_altitude():
    assert isa_temperature(10000) < isa_temperature(0), "Temperature decreases with altitude"


def test_density_positive():
    assert isa_density(0.0) > 0
    assert isa_density(10000) > 0


def test_density_decreases_with_altitude():
    assert isa_density(10000) < isa_density(0)


def test_speed_of_sound_decreases_with_altitude():
    assert speed_of_sound(10000) < speed_of_sound(0)


def test_reynolds_number_positive():
    Re = reynolds_number(100.0, 0.46, 0.0)
    assert Re > 0
    assert Re > 1e5, "Reynolds should be in realistic range for fan blade"


def test_blade_velocity():
    U = blade_velocity(1.0, 2200)
    assert abs(U - 2 * 3.14159 * 2200 / 60) < 1.0, "Blade velocity at r=1m, 2200 RPM"


def test_relative_velocity_pythagorean():
    Va, U = 100.0, 150.0
    W = relative_velocity(Va, U)
    import math
    assert abs(W - math.sqrt(Va**2 + U**2)) < 1e-9
