import numpy as np
from radonpy.core import utils, poly, const
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors, Descriptors3D
from typing import Dict, List
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from radonpy.ff.gaff2_mod import GAFF2
from radonpy.sim import qm
from multiprocessing import Pool
import time
import os
import pandas as pd
from tqdm import tqdm
import traceback

def build_chain_once(monomer: Chem.Mol,
                     terminator: Chem.Mol,
                     natoms_target: int) -> Chem.Mol:
    n_repeat_units: int = poly.calc_n_from_num_atoms(
        monomer,
        natom = natoms_target,
        terminal1 = terminator
    )
    
    chain: Chem.Mol = poly.polymerize_rw(
        monomer,
        n_repeat_units,
        tacticity='atactic',
    )
    chain = poly.terminate_rw(
        chain, 
        terminator,
    )
    
    return n_repeat_units, chain

def get_lowest_energy_monomer(
        monomer: Chem.Mol,
        n_conformers: int = 10,
        max_iters_minimisation: int = 400
    ) -> Chem.Mol:
    """
    Embed several ETKDG conformers, MMFF-relax each, and keep the lowest-energy one.
    Returns the *same* Mol object with exactly one conformer.
    """
    # 1) Embed the pool.
    params = AllChem.ETKDGv3()
    conf_ids: List[int] = rdDistGeom.EmbedMultipleConfs(
        monomer,
        numConfs = n_conformers,
    )

    # 2) MMFF-relax and score.
    energies: Dict[int, float] = {}
    mmff_props = AllChem.MMFFGetMoleculeProperties(monomer)
    for cid in conf_ids:
        AllChem.MMFFOptimizeMolecule(
            monomer,
            confId = cid,
            mmffVariant = 'MMFF94s',
            maxIters = max_iters_minimisation
        )
        ff = AllChem.MMFFGetMoleculeForceField(monomer, mmff_props, confId = cid)
        energies[cid] = ff.CalcEnergy()

    # 3) Stash the best conformer *before* wiping others.
    best_cid: int = min(energies, key = energies.get)
    best_conf: Chem.Conformer = Chem.Conformer(monomer.GetConformer(best_cid))

    # 4) Drop all conformers, then add the winner back.
    monomer.RemoveAllConformers()
    monomer.AddConformer(best_conf, assignId = True)

    return monomer

def process_single_smiles(
        monomer_smiles: str,
        n_atoms_chain: int = 600,
        n_realisations: int = 1) -> Dict[str, float]:
    per_realisation: List[Dict[str, float]] = []
    for k in range(n_realisations):
        monomer = utils.mol_from_smiles(monomer_smiles)
        monomer = get_lowest_energy_monomer(monomer)
        
        terminator = utils.mol_from_smiles('*C')

        n_repeat_units, chain = build_chain_once(
            monomer,
            terminator,
            n_atoms_chain,
        )
        
        descriptors = Descriptors3D.CalcMolDescriptors3D(chain)
        n_atoms = n_repeat_units * monomer.GetNumAtoms()
        descriptors['Rg_normalised'] = descriptors['RadiusOfGyration'] / np.sqrt(n_atoms)

        per_realisation.append(descriptors)

    # arithmetic mean; you could also return np.median(...)
    keys = per_realisation[0].keys()
    return {
        key: float(np.median([d[key] for d in per_realisation]))
        for key in keys
    }

def safe_process_single_smiles(smiles: str) -> Dict[str, float]:
    try:
        return process_single_smiles(smiles)
    except Exception as e:
        print('Failed for', smiles)
        traceback.print_exc()

        return {desc: None for desc in [
            "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
            "PMI1", "PMI2", "PMI3", "PBF", "RadiusOfGyration", "SpherocityIndex",
            "used_random_fallback"
        ]}

if __name__ == '__main__':
    # INPUT_CSV = "data/from_host/train.csv"
    # OUTPUT_CSV = F"data/from_host/train_with_v2_3d_descriptors.csv"

    INPUT_CSV = "data/from_natsume/train_merged.csv"
    OUTPUT_CSV = F"data/from_natsume/train_merged_with_v2_3d_descriptors.csv"

    # N_REPEAT = 3

    const.print_level = 2

    # df = pd.read_csv(INPUT_CSV).sample(n=128).reset_index(drop=True)
    # df = pd.read_csv(INPUT_CSV).head(20)
    df = pd.read_csv(INPUT_CSV)
    smiles_list = df["SMILES"].tolist()

    with Pool(processes=32, maxtasksperchild=1) as pool:
        results = list(tqdm(pool.imap(
            safe_process_single_smiles, 
            smiles_list), 
            total=len(smiles_list)))

    results_df = pd.DataFrame(results)

    final_df = pd.concat([df, results_df], axis=1)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Descriptors computed and saved to {OUTPUT_CSV}")
