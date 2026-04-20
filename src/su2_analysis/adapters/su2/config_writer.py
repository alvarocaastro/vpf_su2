"""Write SU2 .cfg configuration files for RANS airfoil polar runs."""
from __future__ import annotations
from pathlib import Path


# Solution/restart use CSV format so restart chaining works without binary I/O
_CFG_TEMPLATE = """\
% ─────────────────────────────────────────────────────────────────
%  SU2 Configuration – RANS Airfoil Polar (auto-generated)
% ─────────────────────────────────────────────────────────────────

% Problem definition
SOLVER= RANS
KIND_TURB_MODEL= {turb_model}
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
READ_BINARY_RESTART= NO

% Compressible flow
MACH_NUMBER= {mach}
AOA= {aoa}
SIDESLIP_ANGLE= 0.0
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_TEMPERATURE= {T_inf}
REYNOLDS_NUMBER= {reynolds}
REYNOLDS_LENGTH= {chord}
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= {chord}
REF_AREA= {chord}

% Boundary conditions
MARKER_EULER= ( airfoil )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

% Numerical scheme
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= {cfl}
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 1.0, {cfl_max} )
MAX_DELTA_TIME= 1E6

% Convective scheme
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05

% Viscous scheme
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

% Convergence
CONV_RESIDUAL_MINVAL= {conv_residual}
CONV_STARTITER= 10
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-6

% Iteration
ITER= {max_iter}
OUTPUT_WRT_FREQ= {max_iter}

% Input / output
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
SOLUTION_FILENAME= solution_flow
RESTART_FILENAME= restart_flow
OUTPUT_FILES= ( RESTART_ASCII, CSV )
VOLUME_FILENAME= volume_flow
SURFACE_FILENAME= surface_flow
HISTORY_OUTPUT= ( ITER, RMS_RHO, RMS_RHO_E, LIFT, DRAG, MOMENT_Z )
WRT_ZONE_HIST= NO
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG )
"""


def write_su2_config(
    output_path: Path,
    mesh_file: Path,
    mach: float,
    aoa: float,
    reynolds: float,
    chord: float,
    T_inf: float,
    max_iter: int = 2000,
    conv_residual: float = -7.0,
    cfl: float = 5.0,
    turb_model: str = "SA",
) -> Path:
    """Write a SU2 .cfg file for a single AoA RANS run.

    Parameters
    ----------
    output_path   : destination .cfg file
    mesh_file     : absolute path to the .su2 mesh
    mach          : freestream Mach number
    aoa           : angle of attack [degrees]
    reynolds      : chord-based Reynolds number
    chord         : reference chord [m]
    T_inf         : freestream static temperature [K]
    max_iter      : max solver iterations (use ~2000 cold, ~800 restart)
    conv_residual : log10 residual drop target
    cfl           : initial CFL number (adaptive CFL is enabled)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = _CFG_TEMPLATE.format(
        turb_model=turb_model,
        mach=mach,
        aoa=aoa,
        T_inf=T_inf,
        reynolds=reynolds,
        chord=chord,
        cfl=cfl,
        cfl_max=cfl * 20,
        conv_residual=conv_residual,
        max_iter=max_iter,
        mesh_file=str(mesh_file).replace("\\", "/"),
    )
    output_path.write_text(content)
    return output_path
