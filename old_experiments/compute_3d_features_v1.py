from typing import Dict
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors, Descriptors3D
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import Chem

def build_linear_polymer_from_dummy_monomer(
    monomer_smiles: str,
    repeat_count: int
) -> Chem.Mol:
    """
    Accepts either
      • [*:1]…[*:2]   (map-labelled dummies)
      • *…*           (unlabelled dummies – first two are used)
    and returns an N-mer with the dummies converted to sp³ carbons.
    """
    monomer = Chem.MolFromSmiles(monomer_smiles)
    if monomer is None:
        raise ValueError("SMILES failed to parse")

    dummy_atoms = [a for a in monomer.GetAtoms() if a.GetSymbol() == '*']
    if len(dummy_atoms) < 2:
        raise ValueError("Need at least two '*' atoms as attachment points")

    # --- pick the two anchor atoms -----------------------------------------
    labelled = {a.GetAtomMapNum(): a for a in dummy_atoms if a.GetAtomMapNum()}
    if 1 in labelled and 2 in labelled:
        a1, a2 = labelled[1].GetIdx(), labelled[2].GetIdx()
    else:                                 # fall back: first two dummies
        a1, a2 = dummy_atoms[0].GetIdx(), dummy_atoms[1].GetIdx()

    # --- stitch the chain ---------------------------------------------------
    polymer = Chem.RWMol(monomer)
    last_anchor2 = a2                          # anchor-2 of first copy

    for _ in range(1, repeat_count):
        offset = polymer.GetNumAtoms()
        polymer.InsertMol(monomer)
        polymer.AddBond(last_anchor2, a1 + offset, Chem.BondType.SINGLE)
        last_anchor2 = a2 + offset             # update for next turn

    # --- convert every remaining '*' into a carbon so UFF recognises it -----
    for atom in polymer.GetAtoms():
        if atom.GetSymbol() == '*':
            atom.SetAtomicNum(6)               # carbon
            atom.SetIsAromatic(False)          # avoid accidental aromaticity

    Chem.SanitizeMol(polymer)                  # recompute valence, rings, etc.
    return polymer.GetMol()

def robust_embed_and_minimise(
    mol: Chem.Mol,
    forcefield: str = "UFF",
    add_hydrogens: bool = True
) -> Chem.Mol:
    """
    • tries ETKDGv3 first (heavy atoms only),
    • if that fails, retries with random-coordinate embedding,
    • finally runs a light FF clean-up (UFF or MMFF).

    Returns the molecule *with* a conformer on success,
    else raises RuntimeError.
    """
    work = Chem.AddHs(mol) if add_hydrogens else Chem.Mol(mol)

    # ---------- first attempt: standard ETKDG ------------------------------
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.randomSeed = 0xC0FFEE

    def _embed(mol, embed_params):
        """Helper so we can run EmbedMolecule() inside a thread."""
        return rdDistGeom.EmbedMolecule(mol, embed_params)

    with ThreadPoolExecutor(max_workers = 1) as pool:
        future = pool.submit(_embed, work, params)
        try:
            conf_id = future.result(timeout = TIMEOUT_SECONDS)          # seconds
        except TimeoutError:
            print('WARNING: Timeout hit (1)!')
            conf_id = -1                                   # force fallback
            # NOTE: the thread keeps running in the background,
            # but we ignore its result and continue immediately.

    # ---------- fallback: inexpensive random-coords embedding -------------
    used_random_fallback = False
    if conf_id == -1:
        params.useRandomCoords = True
        with ThreadPoolExecutor(max_workers = 1) as pool:
            future = pool.submit(_embed, work, params)
            try:
                conf_id = future.result(timeout = TIMEOUT_SECONDS)
                used_random_fallback = True
            except TimeoutError:
                print('WARNING: Timeout hit (2)!')

    if conf_id == -1:
        raise RuntimeError("3-D embedding failed after all attempts")

    # ---------- quick force-field clean-up --------------------------------
    if forcefield.upper() == "UFF":
        AllChem.UFFOptimizeMolecule(work, confId=conf_id, maxIters=200)
    elif forcefield.upper() == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(work):
            raise RuntimeError("MMFF parameters missing for some atoms")
        AllChem.MMFFOptimizeMolecule(work, confId=conf_id, maxIters=200)
    else:
        raise ValueError("forcefield must be 'UFF' or 'MMFF'")

    return work, used_random_fallback


def compute_all_3d_descriptors(
    input_molecule: Chem.Mol,
    conformer_id: int = -1
) -> Dict[str, float]:
    """
    Compute *all* 3-D descriptors available in RDKit’s Descriptors3D module.

    Parameters
    ----------
    input_molecule : Chem.Mol
        Molecule that already has at least one conformer.
    conformer_id   : int, optional
        ID of the conformer to use.  Defaults to -1 (RDKit’s “active” conformer).

    Returns
    -------
    Dict[str, float]
        Mapping from descriptor name → descriptor value.
        Includes (current RDKit 2024.09.1):  
        Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2,  
        PMI1, PMI2, PMI3, PBF, RadiusOfGyration, SpherocityIndex.
    """
    raw_descriptor_dictionary: Dict[str, float] = Descriptors3D.CalcMolDescriptors3D(
        input_molecule,
        confId = conformer_id
    )

    # Ensure plain Python floats (rdkit *may* return numpy.float64).
    cleaned_descriptor_dictionary: Dict[str, float] = {
        descriptor_name: float(descriptor_value)
        for descriptor_name, descriptor_value in raw_descriptor_dictionary.items()
    }
    return cleaned_descriptor_dictionary

def process_single_smiles(smiles):
    try:
        oligomer = build_linear_polymer_from_dummy_monomer(smiles, N_REPEAT)
        conformer, used_random_fallback = robust_embed_and_minimise(oligomer)
        descriptors = compute_all_3d_descriptors(conformer)
        descriptors['used_random_fallback'] = used_random_fallback
    except:
        print('Failed for', smiles)
        traceback.print_exc()

        descriptors = {desc: None for desc in [
            "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
            "PMI1", "PMI2", "PMI3", "PBF", "RadiusOfGyration", "SpherocityIndex",
            "used_random_fallback"
        ]}
        descriptors['used_random_fallback'] = None

    return descriptors

if __name__ == '__main__':
    N_REPEAT = 2
    TIMEOUT_SECONDS = 30

    INPUT_CSV = "data/from_natsume/train_merged.csv"
    OUTPUT_CSV = F"data/from_natsume/train_merged_with_{N_REPEAT}m_{TIMEOUT_SECONDS}t_descriptors.csv"

    # N_REPEAT = 3

    # df = pd.read_csv(INPUT_CSV).sample(n=400).reset_index(drop=True)
    # df = pd.read_csv(INPUT_CSV).head(20)
    df = pd.read_csv(INPUT_CSV)
    smiles_list = df["SMILES"].tolist()

    with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
        results = list(tqdm(pool.imap(
            process_single_smiles, 
            smiles_list), 
            total=len(smiles_list)))

    results_df = pd.DataFrame(results)

    final_df = pd.concat([df, results_df], axis=1)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Descriptors computed and saved to {OUTPUT_CSV}")
