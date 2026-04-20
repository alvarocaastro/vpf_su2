"""Stage 3 — CFD Post-Processing: Cp distributions, Mach surfaces, shock detection."""
from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from su2_analysis.adapters.su2.su2_parser import parse_surface_flow
from su2_analysis.config import STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.pipeline.contracts import Stage2Result, Stage3Result
from su2_analysis.settings import KORN_KAPPA_AIRFOIL, DPI, FIGURE_FORMAT
from su2_analysis.shared.plot_style import apply_style, CONDITION_COLORS, SECTION_LINESTYLES

log = logging.getLogger(__name__)

CONDITIONS = ["takeoff", "climb", "cruise", "descent"]
SECTIONS   = ["root", "mid", "tip"]


def _korn_critical_mach(cl: float, thickness_ratio: float, sweep_deg: float = 0.0) -> float:
    """Korn equation for drag-divergence Mach number."""
    cos_l = math.cos(math.radians(sweep_deg))
    m_dd = (KORN_KAPPA_AIRFOIL / cos_l
            - thickness_ratio / cos_l**2
            - cl / (10.0 * cos_l**3))
    return max(m_dd, 0.5)


def run_stage3(cfg: AnalysisConfig, stage2: Stage2Result) -> Stage3Result:
    apply_style()
    out_dir = STAGE_DIRS["stage3"]
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2_dir = STAGE_DIRS["stage2"]
    cp_data: Dict[str, pd.DataFrame] = {}
    mach_rows = []

    # Approximate thickness ratio from airfoil name (e.g. naca_65-410 → 0.10)
    airfoil_name = stage2.selected_airfoil
    t_c = _guess_thickness_ratio(airfoil_name)

    for cond_name, fc in cfg.flight_conditions.items():
        pitch_map = stage2.pitch_map
        for section_name in SECTIONS:
            key = f"{cond_name}_{section_name}"
            polar = stage2.polars.get(key)
            if polar is None:
                continue

            pm_row = pitch_map[
                (pitch_map["condition"] == cond_name) &
                (pitch_map["section"] == section_name)
            ]
            alpha_opt = float(pm_row["alpha_opt"].iloc[0]) if not pm_row.empty else 4.0
            mach_rel  = float(polar["mach"].iloc[0]) if "mach" in polar.columns else 0.85

            # Find the alpha run directory closest to alpha_opt
            alpha_dir = _find_alpha_dir(stage2_dir / cond_name / section_name, alpha_opt)
            surface_csv = alpha_dir / "surface_flow.csv" if alpha_dir else None

            if surface_csv and surface_csv.exists():
                df_surf = parse_surface_flow(surface_csv)
            else:
                df_surf = pd.DataFrame()
                log.debug("No surface_flow.csv for %s at α=%.1f°", key, alpha_opt)

            cp_data[key] = df_surf

            # CL at alpha_opt
            pm_cl = float(pm_row["cl_opt"].iloc[0]) if not pm_row.empty else 0.8
            m_crit = _korn_critical_mach(pm_cl, t_c)
            wave_drag = mach_rel > m_crit

            mach_rows.append({
                "condition":   cond_name,
                "section":     section_name,
                "mach_rel":    mach_rel,
                "alpha_opt":   alpha_opt,
                "m_crit_korn": m_crit,
                "wave_drag":   wave_drag,
            })

            _plot_cp(df_surf, key, alpha_opt, mach_rel, m_crit, out_dir)

    mach_summary = pd.DataFrame(mach_rows)
    mach_summary.to_csv(out_dir / "mach_summary.csv", index=False)

    _plot_mach_summary(mach_summary, out_dir)

    return Stage3Result(
        cp_data=cp_data,
        mach_summary=mach_summary,
        output_dir=out_dir,
    )


def _find_alpha_dir(base: Path, alpha_opt: float) -> Path | None:
    """Return the run directory whose AoA label is closest to alpha_opt."""
    if not base.exists():
        return None
    candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("aoa_")]
    if not candidates:
        return None
    def _aoa(d: Path) -> float:
        try:
            return float(d.name.replace("aoa_", "").replace("p", "."))
        except ValueError:
            return float("inf")
    return min(candidates, key=lambda d: abs(_aoa(d) - alpha_opt))


def _guess_thickness_ratio(name: str) -> float:
    """Heuristic: extract thickness ratio from NACA airfoil name."""
    # e.g. naca_65-410 → last two digits = 10% → 0.10
    try:
        digits = name.split("-")[-1]
        return int(digits[-2:]) / 100.0
    except Exception:
        return 0.12


def _plot_cp(
    df: pd.DataFrame,
    key: str,
    alpha_opt: float,
    mach_rel: float,
    m_crit: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not df.empty and "cp" in df.columns and "x" in df.columns:
        # Normalise x by chord
        x_norm = df["x"] / df["x"].max() if df["x"].max() > 0 else df["x"]
        # Separate upper (negative Cp) and lower
        ax.plot(x_norm, -df["cp"], color="#4477AA", lw=1.8, label="−Cp (SU2 RANS)")
        ax.axhline(0, color="k", lw=0.5, ls=":")
    else:
        ax.text(0.5, 0.5, "Surface data not available\n(SU2 surface_flow.csv missing)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    # Critical Cp reference (Cp* at given Mach)
    if mach_rel < 1.0:
        cp_star = (2.0 / (1.4 * mach_rel**2)) * (
            ((2.0 / (1.4 + 1.0)) * (1.0 + 0.5 * (1.4 - 1.0) * mach_rel**2))
            ** (1.4 / (1.4 - 1.0)) - 1.0
        )
        ax.axhline(-cp_star, color="red", lw=1.0, ls="--",
                   label=f"Cp* (M={mach_rel:.2f})")

    shock_label = " ⚠ WAVE DRAG" if mach_rel > m_crit else ""
    ax.set(
        xlabel="x/c",
        ylabel="−Cp",
        title=f"{key.replace('_', ' / ')} | α={alpha_opt:.1f}° M={mach_rel:.2f}{shock_label}",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"cp_{key}.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_mach_summary(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    labels = [f"{r.condition}/{r.section}" for _, r in df.iterrows()]
    bars = ax.bar(x, df["mach_rel"], label="M_rel", color="#4477AA", width=0.4)
    ax.bar(x + 0.4, df["m_crit_korn"], label="M_crit (Korn)", color="#EE6677", width=0.4)
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="Mach", title="Stage 3 — Relative Mach vs Critical Mach (Korn)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"mach_summary.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)
