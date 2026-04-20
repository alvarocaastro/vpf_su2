"""Stage 1 — Airfoil selection via restart-chained SU2 RANS polar sweep."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from su2_analysis.adapters.su2.config_writer import write_su2_config
from su2_analysis.adapters.su2.mesh_generator import generate_cgrid_mesh
from su2_analysis.adapters.su2.su2_parser import build_polar, parse_history, SU2RunResult
from su2_analysis.adapters.su2.su2_runner import run_polar_sweep
from su2_analysis.config import AIRFOIL_DIR, STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.pipeline.contracts import Stage1Result
from su2_analysis.settings import TARGET_YPLUS, DPI, FIGURE_FORMAT
from su2_analysis.shared.atmosphere import (
    isa_density, isa_temperature, sutherland_viscosity, wall_spacing_for_yplus,
)
from su2_analysis.shared.plot_style import apply_style, PALETTE
from su2_analysis.stage1_airfoil_selection.scoring import score_airfoils

log = logging.getLogger(__name__)


def run_stage1(cfg: AnalysisConfig) -> Stage1Result:
    apply_style()
    out_dir = STAGE_DIRS["stage1"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ref      = cfg.reference_condition
    mach     = float(ref["mach"])
    reynolds = float(ref["reynolds"])
    # 2° step for selection — enough to rank airfoils, 5× fewer runs than 1°
    alpha_list = list(np.arange(
        float(ref["alpha_sweep"][0]),
        float(ref["alpha_sweep"][1]) + 0.01,
        2.0,
    ))

    T_inf = isa_temperature(0.0)
    chord = cfg.fan_geometry.sections["mid"].chord
    rho   = isa_density(0.0)
    mu    = sutherland_viscosity(T_inf)
    vel   = reynolds * mu / (rho * chord)
    ws    = wall_spacing_for_yplus(TARGET_YPLUS, vel, chord, 0.0)

    polars: Dict[str, pd.DataFrame] = {}

    for airfoil_name in cfg.airfoil_candidates:
        log.info("Stage 1 — %s (%d AoA points, restart-chained)", airfoil_name, len(alpha_list))
        dat_file = AIRFOIL_DIR / f"{airfoil_name}.dat"
        if not dat_file.exists():
            log.warning("Airfoil file not found: %s — skipping", dat_file)
            continue

        foil_dir  = out_dir / airfoil_name
        mesh_dir  = foil_dir / "mesh"
        sweep_dir = foil_dir / "sweep"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        mesh_file = mesh_dir / f"{airfoil_name}.su2"
        if not mesh_file.exists():
            log.info("  Generating C-grid mesh → %s", mesh_file.name)
            generate_cgrid_mesh(
                airfoil_dat=dat_file,
                output_mesh=mesh_file,
                chord=chord,
                wall_spacing=ws,
                farfield_radius_chords=cfg.su2.mesh_farfield_radius,
                n_airfoil_points=cfg.su2.mesh_airfoil_points,
                n_radial_layers=cfg.su2.mesh_radial_layers,
                growth_rate=cfg.su2.mesh_growth_rate,
            )

        # Base config — AOA and RESTART_SOL are patched per alpha by run_polar_sweep
        base_cfg = foil_dir / "base_config.cfg"
        write_su2_config(
            output_path=base_cfg,
            mesh_file=mesh_file,
            mach=mach,
            aoa=alpha_list[0],
            reynolds=reynolds,
            chord=chord,
            T_inf=T_inf,
            max_iter=cfg.su2.max_iter,
            conv_residual=cfg.su2.convergence_residual,
            cfl=cfg.su2.cfl_number,
            turb_model=cfg.su2.turbulence_model,
        )

        history_map = run_polar_sweep(
            su2_exe=cfg.su2.executable,
            mesh_file=mesh_file,
            base_cfg_file=base_cfg,
            alpha_list=alpha_list,
            sweep_dir=sweep_dir,
            timeout_per_alpha=cfg.su2.timeout_seconds,
            max_retries=cfg.su2.max_retries,
        )

        results = []
        for alpha in sorted(alpha_list):
            h = history_map.get(alpha)
            if h and h.exists():
                try:
                    results.append(parse_history(h, alpha))
                except Exception as exc:
                    log.warning("  Parse failed α=%+.1f°: %s", alpha, exc)
                    results.append(SU2RunResult(alpha, float("nan"), float("nan"),
                                                float("nan"), False, 0))
            else:
                results.append(SU2RunResult(alpha, float("nan"), float("nan"),
                                            float("nan"), False, 0))

        polar_df = build_polar(results)
        polar_df.to_csv(foil_dir / "polar.csv", index=False)
        polars[airfoil_name] = polar_df
        log.info("  %s: %d/%d points converged",
                 airfoil_name, int(polar_df["converged"].sum()), len(polar_df))

    ranking = score_airfoils(polars)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    selected = ranking.iloc[0]["airfoil"]
    log.info("Stage 1 complete — selected: %s", selected)

    _plot_polar_comparison(polars, selected, out_dir)

    return Stage1Result(
        selected_airfoil=selected,
        ranking=ranking,
        polars=polars,
        output_dir=out_dir,
    )


def _plot_polar_comparison(
    polars: Dict[str, pd.DataFrame],
    selected: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, polar) in enumerate(polars.items()):
        color = PALETTE[i % len(PALETTE)]
        lw    = 2.5 if name == selected else 1.2
        ls    = "-"  if name == selected else "--"
        label = f"{name} ★" if name == selected else name
        df = polar[polar["converged"]]
        if df.empty:
            continue
        axes[0].plot(df["alpha"], df["cl"],     color=color, lw=lw, ls=ls, label=label)
        axes[1].plot(df["alpha"], df["cd"]*1e4, color=color, lw=lw, ls=ls)
        axes[2].plot(df["alpha"], df["ld"],     color=color, lw=lw, ls=ls)

    axes[0].set(xlabel="α [°]", ylabel="CL",       title="Lift Curve")
    axes[1].set(xlabel="α [°]", ylabel="CD × 10⁴", title="Drag Polar")
    axes[2].set(xlabel="α [°]", ylabel="CL/CD",     title="Glide Ratio")
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 1 — Airfoil Comparison (SU2 RANS, M=0.3, Re=3M)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"polar_comparison.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)
