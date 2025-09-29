#!/usr/bin/env python3
"""
lammps_pipeline.py
===================
A command‑line utility that mirrors the workflow demonstrated in the Kaggle notebook
"LAMMPS for Dummies" (https://www.kaggle.com/code/hengck23/lammps-for-dummies).

The script performs three high‑level stages:

1. **Data preparation** – convert a SMILES string into:
   * an RDKit molecule object (`mol.pkl`)
   * a homopolymer (`homopoly.pkl`)
   * an amorphous cell containing 10 polymer chains (`ac.pkl`)
   * a LAMMPS data file (`eq1.data`)

2. **Simulation** – generate and run three LAMMPS input scripts (eq1–eq3) that
   perform packing, annealing, and sampling.  You must point the script at a
   *working* LAMMPS executable compiled with the GPU and OMP packages.

3. **Post‑processing** – parse the simulation outputs to report radius of
   gyration (Rg) and density.

The original notebook relied on several helper modules that were shipped in the
same Kaggle dataset.  Those modules are **imported, not re‑implemented,** so make
sure they are importable (e.g. place them on `PYTHONPATH`).

Example
-------
```bash
python lammps_pipeline.py \
    --polymer-id 992359277 \
    --smiles '*CC(*)(C)C(=O)OCCF' \
    --lammps-exec /opt/lammps/bin/lmp \
    --work-dir /scratch/demo‑id‑992359277 \
    --num-omp-threads 4 
```
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem

# -----------------------------------------------------------------------------
# Local helper modules bundled with the original notebook ----------------------
# -----------------------------------------------------------------------------
import sys
sys.path.append('simulations')
from run_write_lammps_data import run_write_lammps_data
from run_write_lammps_script_eq1 import write_input as write_input_eq1
from run_write_lammps_script_eq2 import write_input as write_input_eq2
from run_write_lammps_script_eq3 import write_input as write_input_eq3
from my_helper import read_lammps_log, read_lammps_timeavg_profile

# -----------------------------------------------------------------------------
# Utility functions -----------------------------------------------------------
# -----------------------------------------------------------------------------

def prepare_data(smiles_string: str, termination_smiles: str, destination: Path) -> None:
    """Generate polymer data & LAMMPS input data files.

    Parameters
    ----------
    smiles_string : str
        SMILES representation of the repeat unit (must contain asterisks for
        attachment points).
    termination_smiles : str
        SMILES of the termination unit (defaults to "*C" in the notebook).
    destination : Path
        Directory where all output artefacts will be written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    run_write_lammps_data(smiles_string, termination_smiles, work_dir=str(destination))


def _run_subprocess(command: str, working_directory: Path, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    """Thin wrapper around :pyfunc:`subprocess.run` with sane defaults."""
    print(f"[INFO] Running command: {command}")
    completed_process: subprocess.CompletedProcess[str] = subprocess.run(
        command,
        shell=True,
        cwd=str(working_directory),
        check=False,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        encoding="utf‑8",
    )
    if completed_process.returncode != 0:
        print("[WARN] Command returned non‑zero exit status:", completed_process.returncode)
    return completed_process


def run_preset(
    preset_name: str,
    lammps_exec_path: Path,
    mpi_command_template: str,
    omp_threads: int,
    working_directory: Path,
    debug: bool = False,
) -> None:
    """Generate a LAMMPS input script and run it.

    `mpi_command_template` can refer to the following placeholders:

    * ``{LAMMPS_EXEC}`` – absolute path to the LAMMPS binary
    * ``{name}``        – preset name (eq1 / eq2 / eq3)
    """
    write_input_dispatch = {
        "eq1": write_input_eq1,
        "eq2": write_input_eq2,
        "eq3": write_input_eq3,
    }
    write_input_function = write_input_dispatch[preset_name]

    # 1. Generate <name>.in
    input_script_path = working_directory / f"{preset_name}.in"
    write_input_function(filename=str(input_script_path))

    # 2. Build command
    if debug:
        command_string = mpi_command_template.format(
            LAMMPS_EXEC=lammps_exec_path,
            name=preset_name,
        ).replace("mpirun -n 4", "mpirun -n 1")
    else:
        command_string = mpi_command_template.format(
            LAMMPS_EXEC=lammps_exec_path,
            name=preset_name,
        )

    # 3. Execute
    start_time = _dt.datetime.now()
    print(f"[INFO] Starting {preset_name} at {start_time:%Y‑%m‑%d %H:%M:%S}")
    completed = _run_subprocess(command_string, working_directory)
    elapsed = _dt.datetime.now() - start_time
    print(f"[INFO] Finished {preset_name} – elapsed {elapsed} – return code {completed.returncode}")

    # 4. Optionally dump stdout/stderr to help with debugging
    if completed.stdout:
        (working_directory / f"{preset_name}_stdout.txt").write_text(completed.stdout)
    if completed.stderr:
        (working_directory / f"{preset_name}_stderr.txt").write_text(completed.stderr)


def run_simulation(work_dir: Path, lammps_exec: Path, omp_threads: int, debug: bool) -> None:
    """Run packing, annealing, and sampling simulations sequentially."""
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    # Copy the LAMMPS data file produced during preparation into ./simulate
    simulate_dir = work_dir / "simulate"
    simulate_dir.mkdir(exist_ok=True)
    shutil.copy(work_dir / "eq1.data", simulate_dir / "eq1.data")

    presets = [
        ("eq1", "packing simulation", "mpirun -n 4 {LAMMPS_EXEC} -sf omp -pk omp 4 -in {name}.in -sc none -log {name}_log.lammps"),
        ("eq2", "annealing simulation", "mpirun -n 2 {LAMMPS_EXEC} -sf gpu -pk gpu 1 omp 4 -in {name}.in -sc none -log {name}_log.lammps"),
        ("eq3", "sampling simulation", "mpirun -n 2 {LAMMPS_EXEC} -sf gpu -pk gpu 1 omp 4 -in {name}.in -sc none -log {name}_log.lammps"),
    ]

    for preset_name, description, cmd_template in presets:
        print(f"\n[INFO] ===== {description.upper()} ({preset_name}) =====")
        run_preset(
            preset_name=preset_name,
            lammps_exec_path=lammps_exec,
            mpi_command_template=cmd_template,
            omp_threads=omp_threads,
            working_directory=simulate_dir,
            debug=debug,
        )


def analyse_results(work_dir: Path) -> tuple[float, float]:
    """Return (Rg, density) by parsing `eq3` outputs."""
    simulate_dir = work_dir / "simulate"
    profile_path = simulate_dir / "eq3.rg.profile"
    log_path = simulate_dir / "eq3.log"

    df_rg = read_lammps_timeavg_profile(str(profile_path))
    df_log = read_lammps_log(str(log_path))

    mean_rg: float = float(df_rg["Rg"].mean())
    mean_density: float = float(df_log["Density"].mean())
    return mean_rg, mean_density

# -----------------------------------------------------------------------------
# Command‑line interface -------------------------------------------------------
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End‑to‑end polymer MD pipeline using LAMMPS (derived from Kaggle notebook).")
    parser.add_argument("--polymer-id", type=int, required=True, help="Numeric identifier used only for folder naming / bookkeeping.")
    parser.add_argument("--smiles", type=str, required=True, help="Repeat unit SMILES with asterisks marking attachment points.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Working directory where all outputs will be stored.")
    parser.add_argument("--termination-smiles", type=str, default="*C", help="Termination unit SMILES (default: '*C').")
    parser.add_argument("--lammps-exec", type=Path, default='/media/shared/ExpansionDrive2/Polymer/simulations/lammps/build/lmp', help="Path to LAMMPS executable (must support GPU & OMP packages).")
    parser.add_argument("--num-omp-threads", type=int, default=4, help="OMP thread count passed via OMP_NUM_THREADS.")
    parser.add_argument("--skip-simulation", action="store_true", help="Only run data preparation, skip the MD stage.")
    parser.add_argument("--debug", action="store_true", help="Run each LAMMPS job with a single MPI rank for quick testing.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # ---------------------------------------------------------------------
    # Stage A: data preparation
    # ---------------------------------------------------------------------
    print("[INFO] === Stage A: data preparation ===")
    prepare_data(
        smiles_string=args.smiles,
        termination_smiles=args.termination_smiles,
        destination=args.work_dir,
    )

    # ---------------------------------------------------------------------
    # Stage B: simulation (optional)
    # ---------------------------------------------------------------------
    if not args.skip_simulation:
        print("\n[INFO] === Stage B: molecular dynamics simulation ===")
        run_simulation(
            work_dir=args.work_dir,
            lammps_exec=args.lammps_exec,
            omp_threads=args.num_omp_threads,
            debug=args.debug,
        )

        # -----------------------------------------------------------------
        # Stage C: analysis
        # -----------------------------------------------------------------
        print("\n[INFO] === Stage C: analyse results ===")
        mean_rg, mean_density = analyse_results(args.work_dir)
        print(f"\n[RESULT] Polymer‑ID: {args.polymer_id}\n         SMILES: {args.smiles}\n         Rg: {mean_rg:.6f}\n         Density: {mean_density:.6f}\n")

    print("[INFO] Pipeline complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
