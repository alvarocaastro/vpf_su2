"""Reverse thrust via negative blade pitch (VPF advantage — no cascade doors needed)."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig, EngineParameters
from su2_analysis.shared.atmosphere import isa_density, blade_velocity


def _reverse_thrust_at_delta_beta(
    delta_beta: float,
    Va: float,
    rho: float,
    rpm_fraction: float,
    cfg: AnalysisConfig,
    cl_ref: float = 0.8,
    cd_ref: float = 0.015,
) -> dict:
    """Estimate reverse thrust [N] for a given pitch change Δβ.

    Physical model
    --------------
    When blade pitch is reversed by |Δβ|, the effective AoA becomes negative
    → lift component points upstream (reverse thrust direction).

    T_rev = Z · ρ · Va · Ω·r · c · (CL·sin|β| − CD·cos|β|)   [per section]
    Simplified using mid-span representative section.

    Returns
    -------
    dict with thrust, stall_margin, and key kinematic quantities.
    """
    rpm = cfg.fan_geometry.rpm * rpm_fraction
    sec = cfg.fan_geometry.sections["mid"]
    U   = blade_velocity(sec.radius, rpm)
    Z   = cfg.fan_geometry.n_blades
    c   = sec.chord

    W       = math.sqrt(Va**2 + U**2)
    beta_in = math.degrees(math.atan2(Va, U))     # inlet flow angle [°]
    alpha   = beta_in + delta_beta                 # effective incidence

    # Approximate CL and CD: linear lift slope around cl_ref at alpha_opt≈4°
    # At large negative alpha, stall occurs.  Use a simple sinusoidal model.
    CL = cl_ref * math.sin(math.radians(2.0 * alpha)) / math.sin(math.radians(8.0))
    CD = cd_ref + 0.05 * alpha**2 / 100.0          # induced drag estimate

    # Thrust component (axial direction, negative = reverse)
    beta_rad = math.radians(abs(beta_in + delta_beta))
    T_rev = Z * rho * Va * U * c * (CL * math.sin(beta_rad) - CD * math.cos(beta_rad))

    # Stall margin: alpha_stall ≈ -16° for most cambered airfoils in reverse
    stall_margin = abs(alpha) - 16.0    # positive → stalled

    return {
        "delta_beta_deg": delta_beta,
        "alpha_eff_deg":  alpha,
        "CL":             CL,
        "CD":             CD,
        "T_reverse_N":    -T_rev,        # convention: positive = braking force
        "stall_margin":   stall_margin,
        "U_ms":           U,
        "W_ms":           W,
    }


def sweep_reverse_thrust(
    cfg: AnalysisConfig,
    engine: EngineParameters,
) -> pd.DataFrame:
    """Sweep Δβ and compute reverse thrust across the full range."""
    rt  = engine.reverse_thrust
    rho = isa_density(0.0)
    Va  = rt.axial_velocity
    rpm_frac = rt.rpm_fraction

    delta_betas = np.linspace(
        rt.delta_beta_sweep_start,
        rt.delta_beta_sweep_end,
        rt.delta_beta_sweep_points,
    )
    rows = [_reverse_thrust_at_delta_beta(db, Va, rho, rpm_frac, cfg)
            for db in delta_betas]
    return pd.DataFrame(rows)


def find_optimal_reverse(
    sweep: pd.DataFrame,
    min_stall_margin: float = 0.0,
) -> pd.DataFrame:
    """Return the Δβ that maximises reverse thrust with acceptable stall margin."""
    feasible = sweep[sweep["stall_margin"] <= min_stall_margin]
    if feasible.empty:
        feasible = sweep
    idx = feasible["T_reverse_N"].idxmax()
    return feasible.loc[[idx]].copy()
