"""Run SU2_CFD.exe with live output streaming and restart-chained polar sweeps."""
from __future__ import annotations
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


class SU2ConvergenceError(RuntimeError):
    """Raised when SU2 fails to converge within the allowed iterations."""


class SU2TimeoutError(RuntimeError):
    """Raised when SU2 exceeds the allowed wall-clock time."""


# ── Low-level single run ──────────────────────────────────────────────────────

def _stream_pipe(pipe, log_path: Path) -> None:
    """Forward subprocess stdout lines to console and a log file."""
    with open(log_path, "w") as fh:
        for raw in pipe:
            line = raw.rstrip()
            print(f"  [SU2] {line}", flush=True)
            fh.write(line + "\n")


def run_su2(
    su2_exe: Path,
    cfg_file: Path,
    work_dir: Path,
    timeout: int = 600,
    max_retries: int = 3,
) -> Path:
    """Execute SU2_CFD.exe once and return the path to history.csv.

    Live output is printed to the console prefixed with [SU2] and saved
    to su2_output.log inside work_dir.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    history_file = work_dir / "history.csv"
    su2_log      = work_dir / "su2_output.log"

    for attempt in range(1, max_retries + 1):
        log.info(
            "  SU2 attempt %d/%d — %s",
            attempt, max_retries, work_dir.name,
        )
        t0   = time.monotonic()
        proc = subprocess.Popen(
            [str(su2_exe), str(cfg_file)],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        streamer = threading.Thread(
            target=_stream_pipe, args=(proc.stdout, su2_log), daemon=True
        )
        streamer.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            streamer.join(timeout=2)
            if attempt == max_retries:
                raise SU2TimeoutError(
                    f"SU2 timed out after {timeout}s on {cfg_file.name}"
                )
            log.warning("  Timeout on attempt %d, retrying...", attempt)
            continue

        streamer.join(timeout=5)
        elapsed = time.monotonic() - t0
        log.info("  SU2 finished in %.1f s (exit %d)", elapsed, proc.returncode)

        if proc.returncode != 0:
            log.warning("  Non-zero exit %d on attempt %d", proc.returncode, attempt)
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"SU2 exited {proc.returncode} after {max_retries} attempts "
                    f"for {cfg_file.name}"
                )
            continue

        if not history_file.exists():
            log.warning("  history.csv missing after attempt %d", attempt)
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"SU2 produced no history.csv for {cfg_file.name}"
                )
            continue

        return history_file

    raise SU2ConvergenceError(f"SU2 failed after {max_retries} attempts.")


# ── Restart-chained polar sweep ───────────────────────────────────────────────

_SOLUTION_CSV = "solution_flow.csv"
_RESTART_CSV  = "restart_flow.csv"


def run_polar_sweep(
    su2_exe: Path,
    mesh_file: Path,
    base_cfg_file: Path,
    alpha_list: List[float],
    sweep_dir: Path,
    timeout_per_alpha: int = 600,
    max_retries: int = 2,
) -> dict[float, Path]:
    """Run a full AoA polar sweep using SU2 restart chaining.

    Instead of starting each AoA from a uniform freestream (cold start),
    every run after the first uses the previous converged solution as the
    initial condition.  This typically cuts iterations to convergence by 5-10×.

    The base .cfg file must already have RESTART_SOL= NO set; this function
    flips it to YES for all subsequent angles automatically.

    Parameters
    ----------
    su2_exe       : path to SU2_CFD.exe
    mesh_file     : path to the .su2 mesh (embedded in base_cfg_file)
    base_cfg_file : template .cfg with AOA placeholder (will be overwritten per alpha)
    alpha_list    : sorted list of angles of attack [degrees]
    sweep_dir     : parent directory; one sub-dir is created per AoA
    timeout_per_alpha : wall-clock limit per AoA run [seconds]
    max_retries   : retry attempts per AoA

    Returns
    -------
    Mapping of alpha → Path(history.csv) for converged runs.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Sort ascending so restart solution always comes from nearby AoA
    alphas = sorted(alpha_list)

    # Read the base config text once
    base_cfg_text = base_cfg_file.read_text()

    results: dict[float, Path] = {}
    restart_source: Path | None = None   # solution from previous AoA

    for i, alpha in enumerate(alphas):
        aoa_dir = sweep_dir / f"aoa_{alpha:+07.2f}"
        aoa_dir.mkdir(exist_ok=True)

        cfg_path = aoa_dir / "config.cfg"
        _write_aoa_cfg(base_cfg_text, cfg_path, alpha, restart=restart_source is not None)

        # Copy previous solution as restart for this run
        if restart_source is not None:
            dest = aoa_dir / _RESTART_CSV
            shutil.copy2(restart_source, dest)
            log.debug("  Restart from %s → %s", restart_source.name, dest)

        label = f"α={alpha:+.1f}°  ({i+1}/{len(alphas)})"
        log.info("  Sweep %s", label)

        history_file = aoa_dir / "history.csv"
        try:
            run_su2(
                su2_exe=su2_exe,
                cfg_file=cfg_path,
                work_dir=aoa_dir,
                timeout=timeout_per_alpha,
                max_retries=max_retries,
            )
            results[alpha] = history_file

            # Promote solution for next alpha
            solution = aoa_dir / _SOLUTION_CSV
            if solution.exists():
                restart_source = solution
            else:
                restart_source = None
                log.warning("  No solution_flow.csv after %s — next AoA cold start", label)

        except (SU2ConvergenceError, SU2TimeoutError) as exc:
            log.warning("  %s failed: %s — skipping (no restart propagation)", label, exc)
            restart_source = None   # broken chain; next run cold-starts

    return results


def _write_aoa_cfg(base_text: str, dest: Path, alpha: float, restart: bool) -> None:
    """Write a per-AoA config file, patching AOA and RESTART_SOL."""
    lines = []
    for line in base_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("AOA="):
            lines.append(f"AOA= {alpha}")
        elif stripped.startswith("RESTART_SOL="):
            lines.append("RESTART_SOL= YES" if restart else "RESTART_SOL= NO")
        else:
            lines.append(line)
    dest.write_text("\n".join(lines))
