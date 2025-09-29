import subprocess
from pathlib import Path
import shutil
import multiprocessing
from multiprocessing import Pool
import polars as pl
import pandas as pd
import numpy as np
import time
import os
import hashlib
import traceback
import pickle
import random
import concurrent
import socket

from hengck23.run_write_lammps_data import run_write_lammps_data

import sys
sys.path.append('..')
from fingerprinting_v5 import get_features_dataframe
from train_failure_predictor import get_conformer_energies

# Outputs (opt_method, geom_algorithm).
# Likely to return fast unstable config when we can likely get away with using it,
# returns slower more stable config for SMILES that are likely to cause crashes otherwise.
def pick_conformation_search_config(smiles: str) -> tuple[str, str]:
    # LOAD MODEL.
    with open('crash_predictor.pkl', 'rb') as model_file:
        success_fail_prediction_model = pickle.load(model_file)

    # COMPUTE FEATURES.
    smiles_df = pd.DataFrame([{'SMILES': smiles}])
    features_df = get_features_dataframe(
        smiles_df=smiles_df,
        morgan_fingerprint_dim=0,
        atom_pair_fingerprint_dim=0,
        torsion_dim=0,
        use_maccs_keys=0,
        use_graph_features=0,
        backbone_sidechain_detail_level=1
    )

    min_energies, mean_energies, max_energies, non_converged_counts = get_conformer_energies(
        smiles_df['SMILES'].to_list())
    
    features_df['min_energy'] = min_energies
    features_df['mean_energy'] = mean_energies
    features_df['max_energy'] = max_energies
    features_df['non_converged_count'] = non_converged_counts

    # GENERATE PREDICTION.
    # 1 -> Fast unstable config likely to succeed, so use it.
    # 0 -> Use slower config to avoid crashing.
    p_success_with_fast_config = success_fail_prediction_model.predict_proba(features_df)[0, 1]
    TEMPERATURE = 0.5
    warped_odds = (p_success_with_fast_config / (1 - p_success_with_fast_config)) ** (1 / TEMPERATURE)
    p_use_fast_config = warped_odds / (1 + warped_odds)
    use_fast_config = random.random() < p_use_fast_config
    print('INFO (p_success_with_fast_config, p_use_fast_config, use_fast_config):', p_success_with_fast_config, p_use_fast_config, use_fast_config)
    if use_fast_config:
        return 'hf', 'RFO' # High probability of succeeding in <= 15 minutes, but has 50/50 chance of getting stuck and crashing after an hour.
    else:
        return 'b97-3c', 'NR' # Usually succeeds after burning a couple hours of CPU time. Less crash prone, but never fast.

def try_run_simulation(smiles, termination_smiles, lmp_path, relaxation_script_name, working_directory_path, skip_hard_polymers):
    start_time = time.time()

    # SKIP IF RESULTS ALREADY EXIST.
    if working_directory_path.exists():
        return

    try:
        # PICK CONFIGURATION.
        opt_method, geom_algorithm = pick_conformation_search_config(smiles)
        if skip_hard_polymers and (opt_method != 'hf'):
            return

        # GENERATE POLYMER CHAIN.
        # This:
        #   1. Generates a conformer for the monomer.
        #   2. Assigns charges.
        #   3. Polymerizes the monomer to form a polymer chain.
        run_write_lammps_data(
            smiles = smiles, 
            ter_smiles = termination_smiles, 
            work_dir = working_directory_path,
            mpi=1,
            omp=8,
            mem = 32_000 if skip_hard_polymers else 64_000, # MB
            atoms_per_chain = 600,
            chain_count = 10,
            density = 0.15,
            opt_method=opt_method,
            geom_algorithm=geom_algorithm
        )
        chain_generation_runtime_sec = time.time() - start_time

        # RUN RELAXATION SCRIPT.
        # Brief simulation to get simulation cell near equilibrium.
        shutil.copy(relaxation_script_name, (working_directory_path / relaxation_script_name))
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            f'mpirun -n 1 {lmp_path} -sf gpu -pk gpu 1 omp 4 -in {relaxation_script_name} -sc none -log relaxation_log.lammps',
            shell=True,
            cwd=str(working_directory_path),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        if completed_process.returncode != 0:
            print("[WARN] Command returned non-zero exit status:", completed_process.returncode)

        if completed_process.stdout:
            (working_directory_path / f"mpirun_lmp_stdout.txt").write_text(completed_process.stdout)
        if completed_process.stderr:
            (working_directory_path / f"mpirun_lmp_stderr.txt").write_text(completed_process.stderr)

        # PRINT TIMING STATS.
        total_runtime_sec = time.time() - start_time
        relaxation_runtime_sec = total_runtime_sec - chain_generation_runtime_sec
        runtime_stats_path = working_directory_path / "runtime_stats.txt"
        runtime_stats_path.write_text(f'Chain gen runtime (sec): {chain_generation_runtime_sec}\nSimulation runtime (sec): {relaxation_runtime_sec}\nTotal runtime (sec): {total_runtime_sec}')
    except:
        # LOG ERROR DETAILS TO FILE.
        exception_text = traceback.format_exc()
        print(exception_text)
        (working_directory_path / f"fatal_exception.txt").write_text(exception_text)

def generate_equilibrium_data(smiles: str, working_directory_path: Path, skip_hard_polymers: bool, lmp_path: str):
    TERMINATION_SMILES = '*C'
    # LMP_PATH = 
    RELAXATION_SCRIPT_NAME = 'relax.lmp'
    TIMEOUT_HOURS = 6
    TIMEOUT_SECONDS = TIMEOUT_HOURS * 3600
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
        future = thread_pool.submit(
            try_run_simulation,
            smiles,
            TERMINATION_SMILES,
            lmp_path,
            RELAXATION_SCRIPT_NAME,
            working_directory_path,
            skip_hard_polymers
        )
        try:
            future.result(timeout=TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            timeout_message = f"Timed out after {TIMEOUT_HOURS} hours."
            print("[WARN]", timeout_message)
            (working_directory_path / "timeout_exception.txt").write_text(timeout_message)
            return

if __name__ == '__main__':
    # GET SMILES.
    all_smiles = pl.read_csv('../data/PI1M/PI1M.csv')['SMILES'].to_list()
    smiles_indices = np.unique(np.random.randint(0, len(all_smiles), 200))
    selected_smiles = [all_smiles[index] for index in smiles_indices]

    # FORM CONFIGS.
    hostname = socket.gethostname()
    if hostname == 'mlserver3':
        skip_hard_smiles = False
        lmp_path = '/media/shared/ExpansionDrive2/Polymer/simulations/lammps/build/lmp'
    elif hostname == 'UbuntuDesktop':
        skip_hard_smiles = True
        lmp_path = '/media/shared/ExpansionDrive2/Polymer/simulations/lammps_ud/build/lmp'
    else:
        print('ERROR - running on unknown host:', hostname)
        exit()

    configs = [
        (smiles, Path(f'work_dirs/PI1M_hybrid/{index}'), skip_hard_smiles, lmp_path)
        for smiles, index in zip(selected_smiles, smiles_indices)
    ]
    # configs = []


    # PROCESS SMILES.
    multiprocessing.set_start_method('spawn')
    with Pool(1) as worker_pool:
        worker_pool.starmap(generate_equilibrium_data, configs)