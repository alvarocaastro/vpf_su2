"""Stage 6 — Reverse Thrust orchestrator."""
from __future__ import annotations
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from su2_analysis.config import STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig, EngineParameters
from su2_analysis.pipeline.contracts import Stage5Result, Stage6Result
from su2_analysis.settings import DPI, FIGURE_FORMAT
from su2_analysis.shared.plot_style import apply_style, PALETTE
from su2_analysis.stage6_reverse_thrust.core.services.reverse_thrust_service import (
    find_optimal_reverse, sweep_reverse_thrust,
)
from su2_analysis.stage6_reverse_thrust.core.services.mechanism_weight_service import (
    compute_mechanism_weight,
)

log = logging.getLogger(__name__)


def run_stage6(
    cfg: AnalysisConfig,
    engine: EngineParameters,
    stage5: Stage5Result,
) -> Stage6Result:
    apply_style()
    out_dir = STAGE_DIRS["stage6"]
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    sweep   = sweep_reverse_thrust(cfg, engine)
    optimal = find_optimal_reverse(sweep, engine.reverse_thrust.min_stall_margin_deg)
    weight  = compute_mechanism_weight(engine)

    sweep.to_csv(out_dir / "tables" / "reverse_thrust_sweep.csv", index=False)
    optimal.to_csv(out_dir / "tables" / "reverse_thrust_optimal.csv", index=False)
    weight.to_csv(out_dir / "tables" / "mechanism_weight.csv", index=False)

    # Kinematics at optimal Δβ
    kinematics_row = optimal.copy()
    kinematics_row.to_csv(out_dir / "tables" / "reverse_kinematics.csv", index=False)

    _plot_sweep(sweep, optimal, out_dir)
    _plot_weight(weight, out_dir)

    summary = _build_summary(sweep, optimal, weight)
    (out_dir / "reverse_thrust_summary.txt").write_text(summary)

    return Stage6Result(
        sweep_table=sweep,
        optimal_table=optimal,
        kinematics_table=kinematics_row,
        weight_table=weight,
        summary_text=summary,
        output_dir=out_dir,
    )


def _plot_sweep(sweep: pd.DataFrame, optimal: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sweep["delta_beta_deg"], sweep["T_reverse_N"] / 1000, color=PALETTE[0], lw=2)
    if not optimal.empty:
        axes[0].axvline(optimal["delta_beta_deg"].iloc[0], color="red", lw=1, ls="--",
                        label=f'Δβ_opt = {optimal["delta_beta_deg"].iloc[0]:.1f}°')
    axes[0].set(xlabel="Δβ [°]", ylabel="Reverse Thrust [kN]",
                title="Reverse Thrust vs Pitch Setting")
    axes[0].legend()

    axes[1].plot(sweep["delta_beta_deg"], sweep["stall_margin"], color=PALETTE[4], lw=2)
    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set(xlabel="Δβ [°]", ylabel="Stall margin [°]",
                title="Stall Margin vs Pitch Setting")
    fig.suptitle("Stage 6 — Reverse Thrust Analysis")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"reverse_thrust_sweep.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_weight(weight: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df = weight[weight["mechanism"] != "Net saving (VPF vs cascade)"]
    bars = ax.bar(df["mechanism"], df["total_kg_2engines"],
                  color=[PALETTE[0], PALETTE[4]])
    ax.set(ylabel="Weight — 2-engine installation [kg]",
           title="Stage 6 — VPF Actuator vs Conventional Cascade Reverser")
    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{row.total_kg_2engines:.0f} kg", ha="center", fontsize=9)
    saving = weight[weight["mechanism"].str.startswith("Net saving")]["total_kg_2engines"].iloc[0]
    ax.text(0.5, 0.92, f"Net saving: {saving:.0f} kg", transform=ax.transAxes,
            ha="center", fontsize=10, color="green")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"mechanism_weight.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _build_summary(sweep, optimal, weight) -> str:
    opt = optimal.iloc[0] if not optimal.empty else {}
    save = weight[weight["mechanism"].str.startswith("Net saving")]
    save_kg = float(save["total_kg_2engines"].iloc[0]) if not save.empty else 0.0
    return (
        "=" * 60 + "\nSTAGE 6 — REVERSE THRUST SUMMARY\n" + "=" * 60 + "\n\n"
        f"Optimal pitch setting:   Δβ = {opt.get('delta_beta_deg', 'N/A'):+.1f}°\n"
        f"Max reverse thrust:      {opt.get('T_reverse_N', 0)/1000:.1f} kN (mid-span)\n"
        f"Effective AoA:           {opt.get('alpha_eff_deg', 0):.1f}°\n"
        f"Stall margin:            {opt.get('stall_margin', 0):.1f}°\n\n"
        "MECHANISM WEIGHT COMPARISON (2-engine installation)\n"
        f"  VPF actuator:          {weight.iloc[0]['total_kg_2engines']:.0f} kg\n"
        f"  Cascade reverser:      {weight.iloc[1]['total_kg_2engines']:.0f} kg\n"
        f"  Net saving:            {save_kg:.0f} kg\n\n"
        "Key advantage: VPF achieves reverse braking by pitch rotation only —\n"
        "no cascade doors, blocker doors, or nacelle cutouts required.\n"
    )
