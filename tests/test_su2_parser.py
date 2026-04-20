"""Unit tests for SU2 output file parser."""
import sys
from pathlib import Path
import tempfile
import textwrap
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from su2_analysis.adapters.su2.su2_parser import (
    parse_history, build_polar, SU2RunResult, _read_csv_flexible,
)


_HISTORY_CONTENT = textwrap.dedent("""\
    Inner_Iter,Rho,RhoE,LIFT,DRAG,MOMENT_Z
    0,1e-1,1e-1,0.50,0.020,0.010
    100,1e-4,1e-4,0.82,0.015,0.005
    500,1e-8,1e-8,0.85,0.014,0.004
""")


def test_parse_history_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(_HISTORY_CONTENT)
        path = Path(f.name)
    result = parse_history(path, aoa=4.0)
    assert result.converged
    assert abs(result.cl - 0.85) < 0.01
    assert abs(result.cd - 0.014) < 0.001
    path.unlink()


def test_build_polar_sorted():
    results = [
        SU2RunResult(aoa=5.0,  cl=0.9, cd=0.015, cm=0.0, converged=True, n_iter=500),
        SU2RunResult(aoa=-2.0, cl=0.1, cd=0.012, cm=0.0, converged=True, n_iter=500),
        SU2RunResult(aoa=2.0,  cl=0.5, cd=0.013, cm=0.0, converged=True, n_iter=500),
    ]
    polar = build_polar(results)
    assert list(polar["alpha"]) == sorted(polar["alpha"]), "Polar should be sorted by alpha"
    assert "ld" in polar.columns
    assert polar.loc[polar["alpha"] == 5.0, "ld"].iloc[0] > 0
