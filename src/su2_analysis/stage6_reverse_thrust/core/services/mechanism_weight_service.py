"""Mechanism weight comparison: VPF actuator vs. conventional cascade reverser."""
from __future__ import annotations
import pandas as pd
from su2_analysis.config_loader import EngineParameters


def compute_mechanism_weight(engine: EngineParameters) -> pd.DataFrame:
    """Compare VPF pitch actuator weight vs. conventional cascade reverser.

    Returns
    -------
    DataFrame with weight breakdown per engine and for two-engine installation.
    """
    mw = engine.mechanism_weight
    n_engines = 2

    vpf_weight_per_engine     = mw.vpf_actuator_fraction     * engine.dry_weight_kg
    cascade_weight_per_engine = mw.cascade_reverser_fraction  * engine.dry_weight_kg

    vpf_total     = vpf_weight_per_engine     * n_engines
    cascade_total = cascade_weight_per_engine * n_engines
    saving        = cascade_total - vpf_total

    rows = [
        {
            "mechanism":           "VPF pitch actuator",
            "fraction_dry_weight": mw.vpf_actuator_fraction,
            "weight_per_engine_kg": vpf_weight_per_engine,
            "total_kg_2engines":   vpf_total,
        },
        {
            "mechanism":           "Conventional cascade reverser",
            "fraction_dry_weight": mw.cascade_reverser_fraction,
            "weight_per_engine_kg": cascade_weight_per_engine,
            "total_kg_2engines":   cascade_total,
        },
        {
            "mechanism":           "Net saving (VPF vs cascade)",
            "fraction_dry_weight": mw.cascade_reverser_fraction - mw.vpf_actuator_fraction,
            "weight_per_engine_kg": cascade_weight_per_engine - vpf_weight_per_engine,
            "total_kg_2engines":   saving,
        },
    ]
    return pd.DataFrame(rows)
