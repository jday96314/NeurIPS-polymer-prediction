import glob
from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import polars as pl
import os
from sklearn.model_selection import cross_val_score
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors, Descriptors3D
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import pickle

import sys
sys.path.append('./')
from fingerprinting_v5 import get_features_dataframe

def get_success_flags_df():
    # FIND SIMULATION WORKING DIRECTORIES.
    work_dir_paths = glob.glob('simulations/work_dirs/host_hf/*')
    work_dir_paths += glob.glob('simulations/work_dirs/host_b97/*')
    
    # LOAD SMILES.
    all_smiles = pl.read_csv('data/from_host/train.csv')['SMILES'].to_list()

    # GET SUCCESS/FAIL RECORDS.
    success_fail_records = []
    for working_directory_path in work_dir_paths:
        smiles_index = int(working_directory_path.split('/')[-1])
        smiles = all_smiles[smiles_index]

        success = os.path.exists(f'{working_directory_path}/eq1_final.data')
        failure = os.path.exists(f'{working_directory_path}/fatal_exception.txt')
        if not (success or failure):
            continue

        success_fail_records.append({
            'SMILES': smiles,
            'success': int(success)
        })

    success_flags_df = pl.DataFrame(success_fail_records)
    success_flags_df = success_flags_df.unique(subset='SMILES')
    return success_flags_df

def compute_conformer_energies(smiles: str):
    base_molecule = Chem.MolFromSmiles(smiles)
    if base_molecule is None:
        raise ValueError("Invalid SMILES provided.")

    # Add explicit hydrogens (important for 3D geometry/energies)
    molecule_with_hydrogens = Chem.AddHs(base_molecule)

    # 2) Generate multiple conformers
    embedding_parameters = AllChem.ETKDGv3()
    embedding_parameters.randomSeed = 2025
    embedding_parameters.pruneRmsThresh = 0.1  # prune near-duplicates
    embedding_parameters.numThreads = 0        # use all available threads

    number_of_conformers = 10
    try:
        conformer_id_list = AllChem.EmbedMultipleConfs(
            molecule_with_hydrogens,
            numConfs = number_of_conformers,
            params = embedding_parameters
        )

        # 3) Optimize each conformer (prefer MMFF94s; fallback to UFF)
        use_mmff = AllChem.MMFFHasAllMoleculeParams(molecule_with_hydrogens)
        if use_mmff:
            optimization_results = AllChem.MMFFOptimizeMoleculeConfs(
                molecule_with_hydrogens,
                mmffVariant = "MMFF94s",
                maxIters = 1000
            )
        else:
            optimization_results = AllChem.UFFOptimizeMoleculeConfs(
                molecule_with_hydrogens,
                maxIters = 1000
            )
    except:
        return np.nan,  np.nan, np.nan, number_of_conformers

    # RDKit returns (convergence_flag, energy) for each conformer
    conformer_energies = [result_tuple[1] for result_tuple in optimization_results]

    # Optional: note any non-converged conformers
    non_converged_count = sum(1 for result_tuple in optimization_results if result_tuple[0] != 0)
    # if non_converged_count > 0:
    #     print(f"Warning: {non_converged_count} conformers did not fully converge.")

    # 4) Compute energy statistics
    minimum_energy = min(conformer_energies)
    maximum_energy = max(conformer_energies)
    mean_energy = np.mean(conformer_energies)
    return minimum_energy, mean_energy, maximum_energy, non_converged_count

def get_conformer_energies(all_smiles):
    if len(all_smiles) > 1:
        min_energies, mean_energies, max_energies, non_converged_counts = [], [], [], []
        
        multiprocessing.set_start_method('spawn')
        with Pool(8) as worker_pool:
            worker_outputs = worker_pool.map(compute_conformer_energies, all_smiles)
        for minimum_energy, mean_energy, maximum_energy, non_converged_count in worker_outputs:
            min_energies.append(minimum_energy)
            mean_energies.append(mean_energy)
            max_energies.append(maximum_energy)
            non_converged_counts.append(non_converged_count)

        return min_energies, mean_energies, max_energies, non_converged_counts
    else:
        minimum_energy, mean_energy, maximum_energy, non_converged_count = compute_conformer_energies(all_smiles[0])
        return [minimum_energy], [mean_energy], [maximum_energy], [non_converged_count]

if __name__ == '__main__':
    # LOAD DATA.
    success_flags_df = get_success_flags_df()
    
    train_test_features: pl.DataFrame = get_features_dataframe(
        smiles_df=success_flags_df[['SMILES']], 
        morgan_fingerprint_dim=0,
        atom_pair_fingerprint_dim=0,
        torsion_dim=0,
        use_maccs_keys=0,
        use_graph_features=0,
        backbone_sidechain_detail_level=1
    )
    train_test_labels=success_flags_df['success']

    # ADD ENERGY FEATURES.
    min_energies, mean_energies, max_energies, non_converged_counts = get_conformer_energies(
        success_flags_df['SMILES'].to_list())
    
    train_test_features['min_energy'] = min_energies
    train_test_features['mean_energy'] = mean_energies
    train_test_features['max_energy'] = max_energies
    train_test_features['non_converged_count'] = non_converged_counts

    # INITIALIZE MODEL.
    # model = VotingClassifier(
    #     estimators=[
    #         ('lgbm', LGBMClassifier()),
    #         ('xgb', XGBClassifier())
    #     ],
    #     weights=[2,1],
    #     voting='soft'
    # )
    # for n_estimators in [50, 100, 200]:
    for n_estimators in [50]:
        print(f'Trying with {n_estimators} estimators')

        model = LGBMClassifier(n_estimators=n_estimators, verbose=-1)

        # TEST MODEL.
        scores = []
        for _ in range(5):
            cv_f1_scores = cross_val_score(
                model,
                train_test_features,
                train_test_labels,
                cv=5,
                scoring="f1"
            )
            scores.append(cv_f1_scores.mean())

        print(f"Mean F1 score: {np.mean(scores):.4f}")

    # TRAIN & SAVE FINAL MODEL.
    model.fit(train_test_features, train_test_labels)
    with open('simulations/crash_predictor.pkl', 'wb') as output_file:
        pickle.dump(model, output_file)