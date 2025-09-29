"""
Backbone‑ vs. Side‑Chain Polymer Descriptors
===========================================

Given a repeat‑unit SMILES marked with two "[*]" connection points,
compute a comprehensive feature vector that includes:

1. All RDKit molecular descriptors for:
   • the entire monomer
   • the backbone sub‑molecule
   • the side‑chain sub‑molecule(s)

2. Additional fragment‑aware features inspired by recent literature:
   backbone_aromatic_fraction, sidechain_mass, grafting_density, etc.

Author : you
Python  : ≥ 3.11
RDKit   : ≥ 2023.03
"""

from __future__ import annotations

import math
import multiprocessing
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdchem
from tqdm.auto import tqdm
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import AllChem
from functools import lru_cache

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
PROPERTY_NAMES: Tuple[str, ...] = tuple(
    rdMolDescriptors.Properties.GetAvailableProperties()
)
PROPERTY_CALCULATOR = rdMolDescriptors.Properties(PROPERTY_NAMES)

ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES = ['backbone_exactmw', 'backbone_amw', 'backbone_lipinskiHBA', 'backbone_lipinskiHBD', 'backbone_NumRotatableBonds', 'backbone_NumHBD', 'backbone_NumHBA', 'backbone_NumHeavyAtoms', 'backbone_NumAtoms', 'backbone_NumHeteroatoms', 'backbone_NumAmideBonds', 'backbone_FractionCSP3', 'backbone_NumRings', 'backbone_NumAromaticRings', 'backbone_NumAliphaticRings', 'backbone_NumSaturatedRings', 'backbone_NumHeterocycles', 'backbone_NumAromaticHeterocycles', 'backbone_NumSaturatedHeterocycles', 'backbone_NumAliphaticHeterocycles', 'backbone_NumSpiroAtoms', 'backbone_NumBridgeheadAtoms', 'backbone_NumAtomStereoCenters', 'backbone_NumUnspecifiedAtomStereoCenters', 'backbone_labuteASA', 'backbone_tpsa', 'backbone_CrippenClogP', 'backbone_CrippenMR', 'backbone_chi0v', 'backbone_chi1v', 'backbone_chi2v', 'backbone_chi3v', 'backbone_chi4v', 'backbone_chi0n', 'backbone_chi1n', 'backbone_chi2n', 'backbone_chi3n', 'backbone_chi4n', 'backbone_hallKierAlpha', 'backbone_kappa1', 'backbone_kappa2', 'backbone_kappa3', 'backbone_Phi', 'sidechain_exactmw', 'sidechain_amw', 'sidechain_lipinskiHBA', 'sidechain_lipinskiHBD', 'sidechain_NumRotatableBonds', 'sidechain_NumHBD', 'sidechain_NumHBA', 'sidechain_NumHeavyAtoms', 'sidechain_NumAtoms', 'sidechain_NumHeteroatoms', 'sidechain_NumAmideBonds', 'sidechain_FractionCSP3', 'sidechain_NumRings', 'sidechain_NumAromaticRings', 'sidechain_NumAliphaticRings', 'sidechain_NumSaturatedRings', 'sidechain_NumHeterocycles', 'sidechain_NumAromaticHeterocycles', 'sidechain_NumSaturatedHeterocycles', 'sidechain_NumAliphaticHeterocycles', 'sidechain_NumSpiroAtoms', 'sidechain_NumBridgeheadAtoms', 'sidechain_NumAtomStereoCenters', 'sidechain_NumUnspecifiedAtomStereoCenters', 'sidechain_labuteASA', 'sidechain_tpsa', 'sidechain_CrippenClogP', 'sidechain_CrippenMR', 'sidechain_chi0v', 'sidechain_chi1v', 'sidechain_chi2v', 'sidechain_chi3v', 'sidechain_chi4v', 'sidechain_chi0n', 'sidechain_chi1n', 'sidechain_chi2n', 'sidechain_chi3n', 'sidechain_chi4n', 'sidechain_hallKierAlpha', 'sidechain_kappa1', 'sidechain_kappa2', 'sidechain_kappa3', 'sidechain_Phi', 'backbone_aromatic_fraction', 'backbone_aromatic_ring_count', 'backbone_rotatable_density', 'sidechain_rotatable_density', 'relative_rigidity', 'sidechain_mass', 'longest_sidechain_length', 'sidechain_count', 'grafting_density', 'sidechain_spacing_std', 'monomer_vdw_surface', 'backbone_vdw_surface', 'sidechain_vdw_surface', 'backbone_polarizability', 'sidechain_polarizability', 'monomer_polarizability']

IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES = [
    'grafting_density',
    'relative_rigidity',
    'sidechain_rotatable_density',
    'sidechain_chi1n',
    'sidechain_CrippenClogP',
    'sidechain_chi0n',
    'backbone_aromatic_fraction',
    'backbone_CrippenClogP',
    'sidechain_FractionCSP3',
    'sidechain_Phi',
    'sidechain_kappa3',
    'backbone_FractionCSP3',
    'sidechain_kappa2',
    'longest_sidechain_length',
    'sidechain_chi1v',
    'sidechain_NumAtoms',
    'sidechain_chi2v',
    'sidechain_kappa1',
    'backbone_rotatable_density',
    'sidechain_chi4v'
]

EXTRA_SIDECHAIN_BACKBONE_FEATURE_NAMES = [
    'backbone_mass',
    'sidechain_backbone_mass_ratio',
    'sidechain_backbone_heavy_atom_ratio',
    'sidechain_backbone_tpsa_ratio',
    'simplified_grafting_density',
]

def get_sub_molecule(
    parent_molecule: Chem.Mol,
    atom_indices: Sequence[int]
) -> Chem.Mol:
    """
    Create an RDKit Mol containing *only* `atom_indices` plus the bonds
    between them. Guarantees the result is sanitised even when aromatic
    flags become inconsistent (common when slicing out fragments).
    """
    atom_indices_set = set(atom_indices)
    emol = Chem.RWMol()
    index_map: Dict[int, int] = {}

    # copy atoms
    for orig_idx in atom_indices:
        new_idx = emol.AddAtom(parent_molecule.GetAtomWithIdx(orig_idx))
        index_map[orig_idx] = new_idx

    # copy bonds
    for bond in parent_molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in atom_indices_set and end in atom_indices_set:
            emol.AddBond(
                index_map[begin], index_map[end], bond.GetBondType()
            )

    sub_mol = emol.GetMol()

    try:
        # full sanitisation (fast when it succeeds)
        Chem.SanitizeMol(sub_mol)
    except (rdchem.AtomKekulizeException, rdchem.KekulizeException):
        # fall back: skip kekulisation, then rebuild aromaticity
        light_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        Chem.SanitizeMol(sub_mol, sanitizeOps=light_ops)
        Chem.SetAromaticity(sub_mol, Chem.AromaticityModel.AROMATICITY_DEFAULT)

    # optional: add explicit Hs so heavy‑atom counts stay comparable
    sub_mol = Chem.AddHs(sub_mol)
    return sub_mol


def ensure_ring_info(molecule: Chem.Mol) -> None:
    """
    Populate valence/aromatic caches and perceive rings so that
    descriptor calculators won’t assert on an un‑initialised RingInfo.
    Safe to call repeatedly.
    """
    # Updates valence & implicit/explicit H counts
    molecule.UpdatePropertyCache(strict=False)

    # Fast ring perception that fills RingInfo without touching bond orders
    # (If rings are already perceived this is a no‑op.)
    Chem.FastFindRings(molecule)

def compute_rdkit_descriptors(molecule: Chem.Mol) -> Dict[str, float]:
    """
    Compute the full RDKit descriptor vector for `molecule`.
    Guarantees that RingInfo is initialised first.
    """
    ensure_ring_info(molecule)
    values = PROPERTY_CALCULATOR.ComputeProperties(molecule)
    return dict(zip(PROPERTY_NAMES, values))


# -----------------------------------------------------------------------------
# Backbone / side‑chain identification
# -----------------------------------------------------------------------------
def process_polymer_smiles(
    smiles_string: str
) -> Tuple[Chem.Mol | None, List[int]]:
    """
    Strip out the two '[*]' dummy atoms and return:

    • cleaned RDKit Mol
    • indices of the two attachment atoms (after removal)

    If parsing fails, returns (None, []).
    """
    molecule = Chem.MolFromSmiles(smiles_string)
    if molecule is None:
        return None, []

    star_neighbors: List[int] = []
    indices_to_delete: List[int] = []

    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0:  # star / attachment marker
            star_neighbors.extend(neigh.GetIdx() for neigh in atom.GetNeighbors())
            indices_to_delete.append(atom.GetIdx())

    editable = Chem.RWMol(molecule)
    for idx in sorted(indices_to_delete, reverse=True):
        editable.RemoveAtom(idx)

    cleaned_mol = editable.GetMol()

    # Remap neighbor indices after deletions
    adjusted_neighbors: List[int] = []
    for original_idx in star_neighbors:
        removed_before = sum(1 for deleted in indices_to_delete if deleted < original_idx)
        adjusted_neighbors.append(original_idx - removed_before)

    return cleaned_mol, list(dict.fromkeys(adjusted_neighbors))  # unique & ordered


def identify_backbone_and_sidechains(
    cleaned_molecule: Chem.Mol,
    attachment_indices: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """
    Return indices of backbone atoms and a list of side‑chain index lists.
    Falls back gracefully when the two attachment sites are disconnected.
    """
    num_atoms: int = cleaned_molecule.GetNumAtoms()

    # ----------------------------------------------
    # 1. trivial cases
    # ----------------------------------------------
    if len(attachment_indices) < 2:
        return list(range(num_atoms)), []

    adjacency_matrix = Chem.GetAdjacencyMatrix(cleaned_molecule)
    graph = nx.from_numpy_array(adjacency_matrix)

    # ----------------------------------------------
    # 2. try the normal shortest‑path backbone
    # ----------------------------------------------
    try:
        backbone_path: List[int] = nx.shortest_path(
            graph, attachment_indices[0], attachment_indices[-1]
        )
    except nx.NetworkXNoPath:               # ← add this block
        # Two attachment atoms live in different fragments.
        # Treat the entire molecule as backbone (no side‑chains),
        # but *do* log the situation so you can inspect later if needed.
        # A production system could write to logging.warning instead.
        # print(
        #     f"[WARN] Disconnected attachment points in SMILES → "
        #     f"using whole molecule as backbone."
        # )
        return list(range(num_atoms)), []

    # ----------------------------------------------
    # 3. collect side‑chains as before
    # ----------------------------------------------
    backbone_set = set(backbone_path)
    visited = set(backbone_path)
    sidechain_indices_list: List[List[int]] = []

    for backbone_atom in backbone_path:
        for neighbor in graph.neighbors(backbone_atom):
            if neighbor in backbone_set or neighbor in visited:
                continue
            queue = [neighbor]
            current_chain: List[int] = []
            while queue:
                atom = queue.pop()
                if atom in visited or atom in backbone_set:
                    continue
                visited.add(atom)
                current_chain.append(atom)
                queue.extend(
                    neigh
                    for neigh in graph.neighbors(atom)
                    if neigh not in visited and neigh not in backbone_set
                )
            if current_chain:
                sidechain_indices_list.append(current_chain)

    return backbone_path, sidechain_indices_list


# -----------------------------------------------------------------------------
# Fragment‑specific feature calculations
# -----------------------------------------------------------------------------
def heavy_atom_indices(molecule: Chem.Mol, indices: Sequence[int]) -> List[int]:
    return [
        idx for idx in indices
        if molecule.GetAtomWithIdx(idx).GetAtomicNum() > 1
    ]


def count_aromatic_rings(molecule: Chem.Mol, atom_indices: Sequence[int]) -> int:
    """
    Count *rings* (not atoms) in which at least half the atoms lie on `atom_indices`
    and the ring is aromatic.
    """
    ri = molecule.GetRingInfo()
    rings = ri.AtomRings()
    backbone_set = set(atom_indices)
    ring_count = 0
    for ring in rings:
        if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            overlap = sum(1 for i in ring if i in backbone_set)
            if overlap >= len(ring) // 2:
                ring_count += 1
    return ring_count


def rotatable_bond_density(
    molecule: Chem.Mol,
    atom_indices: Sequence[int]
) -> float:
    """
    Rotatable bonds per heavy atom within the substructure defined by `atom_indices`.
    """
    sub_mol = get_sub_molecule(molecule, atom_indices)
    rotatable_bonds = AllChem.CalcNumRotatableBonds(sub_mol, strict=True)
    heavy_atoms = sum(
        1 for idx in atom_indices
        if molecule.GetAtomWithIdx(idx).GetAtomicNum() > 1
    )
    return rotatable_bonds / heavy_atoms if heavy_atoms else 0.0


def total_mass(molecule: Chem.Mol, atom_indices: Sequence[int]) -> float:
    """
    Sum of atomic weights (isotope‑aware) for the selected atoms.
    Uses Atom.GetMass() so it works across all RDKit versions.
    """
    return sum(
        molecule.GetAtomWithIdx(idx).GetMass()
        for idx in atom_indices
    )


def sidechain_spacing_std(attachment_indices: List[int]) -> float:
    """
    Standard deviation of attachment points along the backbone.
    The attachment indices are assumed to be in backbone order.
    """
    if len(attachment_indices) < 3:
        return 0.0
    differences = np.diff(sorted(attachment_indices))
    return float(np.std(differences, ddof=1))


def labute_asa(molecule: Chem.Mol) -> float:
    asa = rdmd.CalcLabuteASA(molecule, includeHs=False)
    return asa


def mol_volume(molecule: Chem.Mol) -> float:
    # ---------- fast path (unchanged) ----------
    if hasattr(rdmd, "CalcMolVolume"):
        try:
            ensure_ring_info(molecule)
            return rdmd.CalcMolVolume(molecule)
        except (rdchem.ConformerException, RuntimeError):
            pass   # fall through

    # ---------- embed a conformer --------------
    mol3d = Chem.Mol(molecule)                     # copy
    mol3d = Chem.AddHs(mol3d, addCoords=True)
    ensure_ring_info(mol3d)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    try:
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            return 0.0
    except (rdchem.KekulizeException, rdchem.AtomKekulizeException):
        # ETKDG failed because of broken aromatic flags → skip
        return 0.0

    try:                                           # UFF is optional
        AllChem.UFFOptimizeMolecule(mol3d, maxIters=50)
    except Exception:
        pass

    ensure_ring_info(mol3d)
    try:
        return AllChem.ComputeMolVolume(mol3d)
    except Exception:
        return 0.0


MILLER_POLARIZABILITY: dict[int, float] = {
    1: 0.666,  6: 1.75, 7: 1.10, 8: 0.802, 9: 0.557,
    15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05, 53: 5.35,
}

def miller_polarizability(molecule: Chem.Mol, atom_indices: Sequence[int]) -> float:
    return sum(
        MILLER_POLARIZABILITY.get(molecule.GetAtomWithIdx(idx).GetAtomicNum(), 0.0)
        for idx in atom_indices
    )


# -----------------------------------------------------------------------------
# Master feature extraction
# -----------------------------------------------------------------------------
@lru_cache(50_000)
def extract_sidechain_and_backbone_features(smiles_string: str) -> Dict[str, float]:
    """
    Compute global, backbone, side-chain and custom cross-fragment features.
    """
    features: Dict[str, float] = {"SMILES": smiles_string}

    cleaned_mol, attachment_indices = process_polymer_smiles(smiles_string)
    # if cleaned_mol is None:
    #     return features

    backbone_indices, sidechains = identify_backbone_and_sidechains(
        cleaned_mol, attachment_indices
    )
    sidechain_indices_flat = [idx for chain in sidechains for idx in chain]

    # ------------------------------------------------------------------
    # RDKit descriptor blocks
    # ------------------------------------------------------------------
    features.update(
        {
            f"backbone_{name}": value
            for name, value in compute_rdkit_descriptors(
                get_sub_molecule(cleaned_mol, backbone_indices)
            ).items()
        }
    )

    if sidechain_indices_flat:
        sidechain_submol = get_sub_molecule(cleaned_mol, sidechain_indices_flat)
        features.update(
            {
                f"sidechain_{name}": value
                for name, value in compute_rdkit_descriptors(sidechain_submol).items()
            }
        )
    else:
        # Fill zero so downstream code doesn’t run into KeyErrors
        for name in PROPERTY_NAMES:
            features[f"sidechain_{name}"] = 0.0

    # ------------------------------------------------------------------
    # Custom fragment features
    # ------------------------------------------------------------------
    heavy_backbone_atoms = heavy_atom_indices(cleaned_mol, backbone_indices)
    heavy_sidechain_atoms = heavy_atom_indices(cleaned_mol, sidechain_indices_flat)

    # Aromatic metrics
    features["backbone_aromatic_fraction"] = (
        sum(
            1 for idx in heavy_backbone_atoms
            if cleaned_mol.GetAtomWithIdx(idx).GetIsAromatic()
        ) / len(heavy_backbone_atoms) if heavy_backbone_atoms else 0.0
    )
    features["backbone_aromatic_ring_count"] = count_aromatic_rings(
        cleaned_mol, backbone_indices
    )

    # Rotatable bond densities & relative rigidity
    backbone_rot_density = rotatable_bond_density(cleaned_mol, backbone_indices)
    sidechain_rot_density = rotatable_bond_density(cleaned_mol, sidechain_indices_flat)
    features["backbone_rotatable_density"] = backbone_rot_density
    features["sidechain_rotatable_density"] = sidechain_rot_density
    features["relative_rigidity"] = backbone_rot_density - sidechain_rot_density

    # Mass & size descriptors
    features["sidechain_mass"] = total_mass(cleaned_mol, sidechain_indices_flat)
    features["backbone_mass"] = total_mass(cleaned_mol, backbone_indices)
    features["longest_sidechain_length"] = (
        max((len(chain) for chain in sidechains), default=0)
    )

    features["sidechain_count"] = len(sidechains)

    # Grafting metrics
    backbone_heavy_atom_count = len(heavy_backbone_atoms)
    graft_sites = len(sidechains)
    features["grafting_density"] = (
        graft_sites / backbone_heavy_atom_count if backbone_heavy_atom_count else 0.0
    )
    features["sidechain_spacing_std"] = sidechain_spacing_std(attachment_indices)

    # --- van‑der‑Waals surface & volume for each fragment -------------
    features["monomer_vdw_surface"] = labute_asa(cleaned_mol)
    features["backbone_vdw_surface"] = labute_asa(
        get_sub_molecule(cleaned_mol, backbone_indices)
    )
    features["sidechain_vdw_surface"] = (
        labute_asa(get_sub_molecule(cleaned_mol, sidechain_indices_flat))
        if sidechain_indices_flat else 0.0
    )

    # Slow:
    # features["monomer_vdw_volume"] = mol_volume(cleaned_mol)
    # features["backbone_vdw_volume"] = mol_volume(
    #     get_sub_molecule(cleaned_mol, backbone_indices)
    # )
    # features["sidechain_vdw_volume"] = (
    #     mol_volume(get_sub_molecule(cleaned_mol, sidechain_indices_flat))
    #     if sidechain_indices_flat else 0.0
    # )

    # --- Miller polarizability ---------------------------------------
    features["backbone_polarizability"] = miller_polarizability(
        cleaned_mol, backbone_indices
    )
    features["sidechain_polarizability"] = miller_polarizability(
        cleaned_mol, sidechain_indices_flat
    )
    features["monomer_polarizability"] = (
        features["backbone_polarizability"] + features["sidechain_polarizability"]
    )

    # --- Gemini Suggestions -----------------------------------------
    features["sidechain_backbone_mass_ratio"] = features["sidechain_mass"] / (features["backbone_mass"] + 1e6)
    features["sidechain_backbone_heavy_atom_ratio"] = len(heavy_sidechain_atoms) / (len(heavy_backbone_atoms) + 1e6)
    features["sidechain_backbone_tpsa_ratio"] = features["sidechain_tpsa"] / (features["backbone_tpsa"] + 1e6)
    features["simplified_grafting_density"] = features["sidechain_count"] / (len(backbone_indices) + 1e6)

    return features


# -----------------------------------------------------------------------------
# Batch runner
# -----------------------------------------------------------------------------
def analyze_polymers(
    smiles_list: Sequence[str], number_of_jobs: int = -1
) -> pd.DataFrame:
    """
    Parallel wrapper around `extract_polymer_features`.
    """
    if number_of_jobs == -1:
        number_of_jobs = multiprocessing.cpu_count()

    results = Parallel(n_jobs=number_of_jobs, backend="loky")(
        delayed(extract_polymer_features)(smiles) for smiles in tqdm(smiles_list)
    )
    df = pd.DataFrame(results).set_index("SMILES")
    return df


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # example_smiles: List[str] = [
    #     "[*]CC[*]",                      # polyethylene
    #     "[*]CC([*])c1ccccc1",           # polystyrene-like
    #     "[*]c1ccc(cc1)c2ccc(cc2)[*]",   # fully aromatic backbone
    # ]
    example_smiles = pd.read_csv('data/from_host/train.csv')['SMILES'].tolist()[:5]

    feature_table = analyze_polymers(example_smiles, number_of_jobs=1)
    pd.set_option("display.max_columns", None)
    print(feature_table.head())
