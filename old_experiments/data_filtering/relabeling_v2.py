# from autogluon.tabular import TabularPredictor
import pickle
import json
import glob
import polars as pl
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator, AllChem, rdmolops, rdMolDescriptors, rdchem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.ML.Descriptors import MoleculeDescriptors
import networkx as nx
import pandas as pd
import numpy as np
import joblib
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer
from transformers.activations import ACT2FN
from rdkit import Chem
import gc
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.data import Data, Batch 
import joblib
import glob
from torch_geometric.loader import DataLoader
from unimol_tools import MolPredict
from typing import Dict, List, Sequence, Tuple
import os
from autogluon.tabular import TabularPredictor
import joblib
from joblib import Memory
from sentence_transformers import SentenceTransformer

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

TARGET_NAMES = ["Tg", "FFV", "Tc", "Density", "Rg"]

#region Tabular

def load_tabular_models():
    MODEL_DIRECTORIES = [
        'models/TabularPredictor_20250911_005137',
        'models/TabularPredictor_20250911_082136',
    ]

    targets_to_preprocessing_configs: dict[str,dict] = {}
    targets_to_model_groups: dict[list[list]] = {}
    for target_name in TARGET_NAMES:
        # LOAD TARGET MODELS & CONFIGS.
        targets_to_preprocessing_configs[target_name] = targets_to_preprocessing_configs.get(target_name, [])
        targets_to_model_groups[target_name] = targets_to_model_groups.get(target_name, [])
        for model_directory_path in MODEL_DIRECTORIES:
            # LOAD CONFIG.
            with open(f'{model_directory_path}/{target_name}_features_config.json', 'r') as config_file:
                config = json.load(config_file)
            targets_to_preprocessing_configs[target_name].append(config)

            # LOAD MODELS.
            model_group = []
            for model_path in glob.glob(f'{model_directory_path}/{target_name}*.pkl'):
                try:
                    with open(model_path, 'rb') as model_file:
                        model = pickle.load(model_file)
                        model_group.append(model)
                except:
                    model = TabularPredictor.load(model_path, require_py_version_match=False)
                    model_group.append(model)
            targets_to_model_groups[target_name].append(model_group)

    return targets_to_preprocessing_configs, targets_to_model_groups

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
    Compute global, backbone, side‑chain and custom cross‑fragment features.
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

ALL_GEMINI_FEATURE_NAMES = ['element_fraction_C', 'element_fraction_N', 'element_fraction_O', 'halogen_count', 'halogen_fraction', 'aromatic_bond_fraction', 'double_bond_fraction', 'triple_bond_fraction', 'wiener_index', 'ring_atom_fraction', 'gasteiger_charge_mean', 'gasteiger_charge_std', 'gasteiger_charge_max_pos', 'gasteiger_charge_max_neg', 'count_ester_carbonate', 'count_sulfone_sulfonamide', 'count_ether_non_aromatic']
IMPORTANT_GEMINI_FEATURE_NAMES = ['element_fraction_C', 
 'element_fraction_O',
 'double_bond_fraction',
 'ring_atom_fraction',
 'element_fraction_N',
 'aromatic_bond_fraction',
 'gasteiger_charge_max_neg',
 'halogen_fraction',
 'wiener_index',
 'gasteiger_charge_mean']

def wiener_index(m):
    res = 0
    amat = Chem.GetDistanceMatrix(m)
    num_atoms = m.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            res += amat[i][j]
    return res

_PRED_CACHE_DIR = os.path.expanduser("~/.cache/gemini_features")
@Memory(location=_PRED_CACHE_DIR, verbose=0).cache
def compute_inexpensive_features(mol: Chem.Mol) -> Dict[str, float]:
    """
    Computes a set of computationally inexpensive features for a given
    RDKit molecule.

    This includes elemental/bond composition, simplified topological descriptors,
    and physicochemical proxies based on partial charges and SMARTS patterns.

    Args:
        mol: An RDKit molecule object.

    Returns:
        A dictionary mapping feature names to their calculated values.
    """
    if mol is None:
        return {}

    features: Dict[str, float] = {}
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    
    # Pre-calculate properties to avoid re-computation
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)

    # =========================================================================
    # Section 1: Elemental & Bond Composition Features
    # =========================================================================

    # --- Element Fractions ---
    element_counts = {atom.GetAtomicNum(): 0 for atom in mol.GetAtoms()}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1: # Only count heavy atoms
             element_counts[atom.GetAtomicNum()] = element_counts.get(atom.GetAtomicNum(), 0) + 1

    features['element_fraction_C'] = element_counts.get(6, 0) / num_heavy_atoms if num_heavy_atoms > 0 else 0
    features['element_fraction_N'] = element_counts.get(7, 0) / num_heavy_atoms if num_heavy_atoms > 0 else 0
    features['element_fraction_O'] = element_counts.get(8, 0) / num_heavy_atoms if num_heavy_atoms > 0 else 0

    # --- Halogen Count & Fraction ---
    halogen_atomic_nums = {9, 17, 35, 53}  # F, Cl, Br, I
    halogen_count = sum(count for atomic_num, count in element_counts.items() if atomic_num in halogen_atomic_nums)
    features['halogen_count'] = halogen_count
    features['halogen_fraction'] = halogen_count / num_heavy_atoms if num_heavy_atoms > 0 else 0

    # --- Bond Type Ratios ---
    num_bonds = mol.GetNumBonds()
    if num_bonds > 0:
        num_aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        num_double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
        num_triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.TRIPLE)
        
        num_non_aromatic_bonds = num_bonds - num_aromatic_bonds

        features['aromatic_bond_fraction'] = num_aromatic_bonds / num_bonds
        features['double_bond_fraction'] = num_double_bonds / num_non_aromatic_bonds if num_non_aromatic_bonds > 0 else 0
        features['triple_bond_fraction'] = num_triple_bonds / num_non_aromatic_bonds if num_non_aromatic_bonds > 0 else 0
    else:
        features['aromatic_bond_fraction'] = 0
        features['double_bond_fraction'] = 0
        features['triple_bond_fraction'] = 0


    # =========================================================================
    # Section 2: Simplified Topological & Shape Descriptors
    # =========================================================================

    # --- Wiener Index ---
    # The sum of the shortest paths between all pairs of heavy atoms.
    features['wiener_index'] = wiener_index(mol)

    # --- Ring Atom Fraction ---
    # The number of atoms that are part of any ring.
    ring_info = mol.GetRingInfo()
    unique_ring_atoms = set()
    for ring in ring_info.AtomRings():
        unique_ring_atoms.update(ring)
    
    features['ring_atom_fraction'] = len(unique_ring_atoms) / num_heavy_atoms if num_heavy_atoms > 0 else 0


    # =========================================================================
    # Section 3: Fast Physicochemical Proxies
    # =========================================================================

    # --- Gasteiger Partial Charges Statistics ---
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol, nIter=12)
        charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms() if not np.isnan(atom.GetDoubleProp('_GasteigerCharge'))]
        if charges:
            features['gasteiger_charge_mean'] = np.mean(charges)
            features['gasteiger_charge_std'] = np.std(charges)
            features['gasteiger_charge_max_pos'] = max(c for c in charges if c > 0) if any(c > 0 for c in charges) else 0
            features['gasteiger_charge_max_neg'] = min(c for c in charges if c < 0) if any(c < 0 for c in charges) else 0
        else:
             features.update({'gasteiger_charge_mean': 0, 'gasteiger_charge_std': 0, 'gasteiger_charge_max_pos': 0, 'gasteiger_charge_max_neg': 0})
    except Exception: # Handle cases where charge calculation fails
        features.update({'gasteiger_charge_mean': 0, 'gasteiger_charge_std': 0, 'gasteiger_charge_max_pos': 0, 'gasteiger_charge_max_neg': 0})
        
    # --- SMARTS-based Functional Group Counts ---
    smarts_patterns = {
        'count_ester_carbonate': Chem.MolFromSmarts('[CX3](=O)[OX2]'),
        'count_sulfone_sulfonamide': Chem.MolFromSmarts('[SD4](=O)(=O)'),
        'count_ether_non_aromatic': Chem.MolFromSmarts('[OD2]([C;!$(C=O)])[C;!$(C=O)]')
    }

    for name, pattern in smarts_patterns.items():
        if pattern:
            features[name] = len(mol.GetSubstructMatches(pattern))
        else: # Should not happen with valid SMARTS
            features[name] = 0
            
    return features

debug = False

@lru_cache(maxsize=1_000_000)
def get_feature_vector(
        smiles: str,
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int,
        use_extra_backbone_sidechain_features: bool,
        gemini_features_detail_level: int):
    # PARSE SMILES.
    mol = Chem.MolFromSmiles(smiles)
    
    # GET DESCRIPTORS.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    descriptor_generator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = np.array(descriptor_generator.CalcDescriptors(mol))

    # print('descriptors:', len(descriptors))

    # GET MORGAN FINGERPRINT.
    morgan_fingerprint = np.array([])
    if morgan_fingerprint_dim > 0:
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan_fingerprint_dim)
        morgan_fingerprint = list(morgan_generator.GetFingerprint(mol))

    # print('morgan_fingerprint:', len(morgan_fingerprint))

    # GET ATOM PAIR FINGERPRINT.
    atom_pair_fingerprint = np.array([])
    if atom_pair_fingerprint_dim > 0:
        atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=atom_pair_fingerprint_dim)
        atom_pair_fingerprint = list(atom_pair_generator.GetFingerprint(mol))

    # print('atom_pair_fingerprint:', len(atom_pair_fingerprint))

    # GET MACCS.
    maccs_keys = np.array([])
    if use_maccs_keys:
        maccs_keys = MACCSkeys.GenMACCSKeys(mol)
        maccs_keys = list(maccs_keys)

    # print('maccs_keys:', len(maccs_keys))

    # GET TORSION FINGERPRINT.
    torsion_fingerprint = np.array([])
    if torsion_dim > 0:
        torsion_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=torsion_dim)
        torsion_fingerprint = list(torsion_generator.GetFingerprint(mol))

    # print('torsion_fingerprint:', len(torsion_fingerprint))

    # GET GRAPH FEATURES.
    graph_features = []
    if use_graph_features:
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
        graph = nx.from_numpy_array(adjacency_matrix)
        graph_diameter = nx.diameter(graph) if nx.is_connected(graph) else 0
        avg_shortest_path = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 0
        cycle_count = len(list(nx.cycle_basis(graph)))
        graph_features = [graph_diameter, avg_shortest_path, cycle_count]

    # print('graph_features:', len(graph_features))

    # GET SIDECHAIN & BACKBONE FEATURES.
    extra_sidechain_backbone_feature_names = EXTRA_SIDECHAIN_BACKBONE_FEATURE_NAMES if use_extra_backbone_sidechain_features else []
    if (backbone_sidechain_detail_level == 0) and (not use_extra_backbone_sidechain_features):
        sidechain_backbone_features = []
    elif (backbone_sidechain_detail_level == 0) and use_extra_backbone_sidechain_features:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in extra_sidechain_backbone_feature_names]
    elif backbone_sidechain_detail_level == 1:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES + extra_sidechain_backbone_feature_names]
    elif backbone_sidechain_detail_level == 2:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES + extra_sidechain_backbone_feature_names]
    else:
        assert False, f'Invalid backbone vs. sidechain detail level: {backbone_sidechain_detail_level}'

    # print('sidechain_backbone_features:', len(sidechain_backbone_features))

    # GET GEMINI FEATURES.
    if gemini_features_detail_level == 0:
        gemini_features = []
    elif gemini_features_detail_level == 1:
        gemini_features = compute_inexpensive_features(mol)
        gemini_features = [gemini_features[name] for name in IMPORTANT_GEMINI_FEATURE_NAMES]
    elif gemini_features_detail_level == 2:
        gemini_features = compute_inexpensive_features(mol)
        gemini_features = [gemini_features[name] for name in ALL_GEMINI_FEATURE_NAMES]
    else:
        assert False, f'Invalid backbone vs. sidechain detail level: {backbone_sidechain_detail_level}'

    # print('descriptors:', len(descriptors))

    # CONCATENATE FEATURES.
    features = np.concatenate([
        descriptors, 
        morgan_fingerprint, 
        atom_pair_fingerprint, 
        maccs_keys, 
        torsion_fingerprint,
        graph_features,
        sidechain_backbone_features,
        gemini_features
    ])
    return features

def _get_standard_features_dataframe(
        smiles_df: pd.DataFrame, 
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int,
        use_extra_backbone_sidechain_features: bool,
        gemini_features_detail_level: int) -> pd.DataFrame:
    # GET FEATURE NAMES.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    morgan_col_names = [f'mfp_{i}' for i in range(morgan_fingerprint_dim)]
    atom_pair_col_names = [f'ap_{i}' for i in range(atom_pair_fingerprint_dim)]
    maccs_col_names = [f'maccs_{i}' for i in range(167)] if use_maccs_keys else []
    torsion_col_names = [f'tt_{i}' for i in range(torsion_dim)]
    graph_col_names = ['graph_diameter', 'avg_shortest_path', 'num_cycles'] if use_graph_features else []
    extra_sidechain_col_names = EXTRA_SIDECHAIN_BACKBONE_FEATURE_NAMES if use_extra_backbone_sidechain_features else []
    sidechain_col_names = [[], IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES, ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES][backbone_sidechain_detail_level] + extra_sidechain_col_names
    gemini_col_names = [[], IMPORTANT_GEMINI_FEATURE_NAMES, ALL_GEMINI_FEATURE_NAMES][gemini_features_detail_level]
    feature_col_names = descriptor_names + morgan_col_names + atom_pair_col_names + maccs_col_names + torsion_col_names + graph_col_names + sidechain_col_names + gemini_col_names

    # print('descriptor_names:', len(descriptor_names))
    # print('morgan_col_names:', len(morgan_col_names))
    # print('atom_pair_col_names:', len(atom_pair_col_names))
    # print('maccs_col_names:', len(maccs_col_names))
    # print('torsion_col_names:', len(torsion_col_names))
    # print('graph_col_names:', len(graph_col_names))
    # print('extra_sidechain_col_names:', len(extra_sidechain_col_names))
    # print('sidechain_col_names:', len(sidechain_col_names))
    # print('gemini_col_names:', len(gemini_col_names))
    
    # GET FEATURES.
    try:
        features_df = pd.DataFrame(
            np.vstack([
                get_feature_vector(
                    smiles,
                    morgan_fingerprint_dim,
                    atom_pair_fingerprint_dim,
                    torsion_dim,
                    use_maccs_keys,
                    use_graph_features,
                    backbone_sidechain_detail_level,
                    use_extra_backbone_sidechain_features,
                    gemini_features_detail_level
                ) 
                for smiles 
                in smiles_df['SMILES']]),
            columns=feature_col_names
        )
    except:
        global debug
        debug=True
        features_df = pd.DataFrame(
            np.vstack([
                get_feature_vector(
                    smiles,
                    morgan_fingerprint_dim,
                    atom_pair_fingerprint_dim,
                    torsion_dim,
                    use_maccs_keys,
                    use_graph_features,
                    backbone_sidechain_detail_level,
                    use_extra_backbone_sidechain_features,
                    gemini_features_detail_level
                ) 
                for smiles 
                in smiles_df['SMILES']]),
            columns=feature_col_names
        )

    # CLEAN FEATURES.
    f32_max = np.finfo(np.float32).max
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df[features_df > f32_max] = np.nan
    features_df[features_df < -f32_max] = np.nan

    return features_df

_PRED_CACHE_DIR = "predicted_features"
@Memory(location=_PRED_CACHE_DIR, verbose=0).cache
def _get_predicted_features_dataframe(smiles_df: pd.DataFrame, models_directory_path: str) -> pd.DataFrame:
    # LOAD MODELS.
    model_paths = glob.glob(os.path.join(models_directory_path, '*.joblib'))
    model_paths = sorted(model_paths)
    models = [joblib.load(model_path) for model_path in model_paths]

    # LOAD FEATURES CONFIG.
    features_config_path = os.path.join(models_directory_path, 'features_config.json')
    with open(features_config_path, 'r') as features_file:
        features_config = json.load(features_file)

    # COMPUTE INPUT FEATURES.
    input_features_df = _get_standard_features_dataframe(
        smiles_df, 
        **features_config, 
        gemini_features_detail_level=0, 
        use_extra_backbone_sidechain_features=False
    )

    # COMPUTE PREDICTED FEATURES.
    predicted_features_df = pd.DataFrame()
    for model_path, model in zip(model_paths, models):
        predictions = model.predict(input_features_df)
        col_name = os.path.splitext(os.path.basename(model_path))[0]
        predicted_features_df[col_name] = predictions

    return predicted_features_df

_POLY_BERT_CACHE_DIR = "poly_bert_embeddings"
@Memory(location=_POLY_BERT_CACHE_DIR, verbose=0).cache
def _get_poly_bert_embeddings(smiles_df: pd.DataFrame, polybert_embedding_dim_count: int):
    polyBERT = SentenceTransformer('kuelumbus/polyBERT')
    embeddings = polyBERT.encode(smiles_df['SMILES'].to_list())
    
    embedding_col_names = [f'polyBERT_{index}' for index in range(len(embeddings[0]))]
    features_df = pd.DataFrame(embeddings, columns=embedding_col_names)
    
    ranked_features = ['polyBERT_360', 'polyBERT_68', 'polyBERT_32', 'polyBERT_207', 'polyBERT_127', 'polyBERT_348', 'polyBERT_189', 'polyBERT_285', 'polyBERT_424', 'polyBERT_415', 'polyBERT_51', 'polyBERT_384', 'polyBERT_210', 'polyBERT_492', 'polyBERT_204', 'polyBERT_410', 'polyBERT_203', 'polyBERT_402', 'polyBERT_50', 'polyBERT_88', 'polyBERT_457', 'polyBERT_584', 'polyBERT_112', 'polyBERT_295', 'polyBERT_544', 'polyBERT_190', 'polyBERT_408', 'polyBERT_338', 'polyBERT_54', 'polyBERT_179', 'polyBERT_246', 'polyBERT_471', 'polyBERT_540', 'polyBERT_280', 'polyBERT_428', 'polyBERT_512', 'polyBERT_82', 'polyBERT_275', 'polyBERT_417', 'polyBERT_154', 'polyBERT_449', 'polyBERT_554', 'polyBERT_74', 'polyBERT_396', 'polyBERT_91', 'polyBERT_208', 'polyBERT_375', 'polyBERT_288', 'polyBERT_201', 'polyBERT_400', 'polyBERT_289', 'polyBERT_134', 'polyBERT_468', 'polyBERT_183', 'polyBERT_228', 'polyBERT_0', 'polyBERT_171', 'polyBERT_436', 'polyBERT_95', 'polyBERT_151', 'polyBERT_297', 'polyBERT_36', 'polyBERT_186', 'polyBERT_316', 'polyBERT_81', 'polyBERT_463', 'polyBERT_302', 'polyBERT_227', 'polyBERT_78', 'polyBERT_168', 'polyBERT_145', 'polyBERT_3', 'polyBERT_170', 'polyBERT_423', 'polyBERT_571', 'polyBERT_301', 'polyBERT_176', 'polyBERT_432', 'polyBERT_60', 'polyBERT_552', 'polyBERT_262', 'polyBERT_300', 'polyBERT_93', 'polyBERT_169', 'polyBERT_191', 'polyBERT_160', 'polyBERT_503', 'polyBERT_380', 'polyBERT_451', 'polyBERT_548', 'polyBERT_57', 'polyBERT_570', 'polyBERT_332', 'polyBERT_441', 'polyBERT_99', 'polyBERT_478', 'polyBERT_477', 'polyBERT_394', 'polyBERT_58', 'polyBERT_308', 'polyBERT_118', 'polyBERT_85', 'polyBERT_101', 'polyBERT_84', 'polyBERT_475', 'polyBERT_440', 'polyBERT_299', 'polyBERT_397', 'polyBERT_325', 'polyBERT_327', 'polyBERT_244', 'polyBERT_1', 'polyBERT_100', 'polyBERT_209', 'polyBERT_343', 'polyBERT_109', 'polyBERT_226', 'polyBERT_21', 'polyBERT_370', 'polyBERT_367', 'polyBERT_23', 'polyBERT_193', 'polyBERT_476', 'polyBERT_369', 'polyBERT_556', 'polyBERT_357', 'polyBERT_335', 'polyBERT_511', 'polyBERT_597', 'polyBERT_494', 'polyBERT_309', 'polyBERT_517', 'polyBERT_562', 'polyBERT_376', 'polyBERT_165', 'polyBERT_336', 'polyBERT_293', 'polyBERT_546', 'polyBERT_158', 'polyBERT_42', 'polyBERT_591', 'polyBERT_218', 'polyBERT_215', 'polyBERT_458', 'polyBERT_383', 'polyBERT_216', 'polyBERT_337', 'polyBERT_253', 'polyBERT_153', 'polyBERT_496', 'polyBERT_328', 'polyBERT_22', 'polyBERT_578', 'polyBERT_319', 'polyBERT_56', 'polyBERT_196', 'polyBERT_433', 'polyBERT_257', 'polyBERT_86', 'polyBERT_486', 'polyBERT_281', 'polyBERT_594', 'polyBERT_497', 'polyBERT_7', 'polyBERT_66', 'polyBERT_15', 'polyBERT_286', 'polyBERT_290', 'polyBERT_37', 'polyBERT_557', 'polyBERT_312', 'polyBERT_141', 'polyBERT_506', 'polyBERT_304', 'polyBERT_598', 'polyBERT_40', 'polyBERT_466', 'polyBERT_10', 'polyBERT_320', 'polyBERT_264', 'polyBERT_500', 'polyBERT_72', 'polyBERT_501', 'polyBERT_76', 'polyBERT_270', 'polyBERT_434', 'polyBERT_374', 'polyBERT_16', 'polyBERT_558', 'polyBERT_550', 'polyBERT_454', 'polyBERT_260', 'polyBERT_177', 'polyBERT_138', 'polyBERT_28', 'polyBERT_2', 'polyBERT_461', 'polyBERT_223', 'polyBERT_499', 'polyBERT_229', 'polyBERT_513', 'polyBERT_233', 'polyBERT_89', 'polyBERT_490', 'polyBERT_19', 'polyBERT_219', 'polyBERT_514', 'polyBERT_135', 'polyBERT_132', 'polyBERT_474', 'polyBERT_483', 'polyBERT_70', 'polyBERT_339', 'polyBERT_491', 'polyBERT_41', 'polyBERT_46', 'polyBERT_493', 'polyBERT_504', 'polyBERT_157', 'polyBERT_391', 'polyBERT_373', 'polyBERT_566', 'polyBERT_102', 'polyBERT_5', 'polyBERT_205', 'polyBERT_47', 'polyBERT_314', 'polyBERT_479', 'polyBERT_65', 'polyBERT_156', 'polyBERT_63', 'polyBERT_581', 'polyBERT_545', 'polyBERT_276', 'polyBERT_291', 'polyBERT_379', 'polyBERT_429', 'polyBERT_108', 'polyBERT_582', 'polyBERT_350', 'polyBERT_352', 'polyBERT_361', 'polyBERT_251', 'polyBERT_49', 'polyBERT_324', 'polyBERT_27', 'polyBERT_247', 'polyBERT_509', 'polyBERT_577', 'polyBERT_284', 'polyBERT_182', 'polyBERT_372', 'polyBERT_206', 'polyBERT_508', 'polyBERT_87', 'polyBERT_242', 'polyBERT_147', 'polyBERT_534', 'polyBERT_188', 'polyBERT_237', 'polyBERT_502', 'polyBERT_273', 'polyBERT_403', 'polyBERT_298', 'polyBERT_307', 'polyBERT_368', 'polyBERT_61', 'polyBERT_267', 'polyBERT_541', 'polyBERT_322', 'polyBERT_580', 'polyBERT_527', 'polyBERT_238', 'polyBERT_358', 'polyBERT_192', 'polyBERT_199', 'polyBERT_555', 'polyBERT_498', 'polyBERT_495', 'polyBERT_94', 'polyBERT_202', 'polyBERT_453', 'polyBERT_442', 'polyBERT_75', 'polyBERT_38', 'polyBERT_55', 'polyBERT_488', 'polyBERT_29', 'polyBERT_425', 'polyBERT_124', 'polyBERT_447', 'polyBERT_366', 'polyBERT_259', 'polyBERT_404', 'polyBERT_387', 'polyBERT_146', 'polyBERT_537', 'polyBERT_221', 'polyBERT_536', 'polyBERT_178', 'polyBERT_437', 'polyBERT_572', 'polyBERT_163', 'polyBERT_287', 'polyBERT_392', 'polyBERT_45', 'polyBERT_531', 'polyBERT_507', 'polyBERT_243', 'polyBERT_465', 'polyBERT_567', 'polyBERT_510', 'polyBERT_266', 'polyBERT_444', 'polyBERT_194', 'polyBERT_388', 'polyBERT_330', 'polyBERT_586', 'polyBERT_9', 'polyBERT_6', 'polyBERT_122', 'polyBERT_313', 'polyBERT_589', 'polyBERT_239', 'polyBERT_344', 'polyBERT_549', 'polyBERT_200', 'polyBERT_418', 'polyBERT_574', 'polyBERT_560', 'polyBERT_155', 'polyBERT_416', 'polyBERT_214', 'polyBERT_363', 'polyBERT_575', 'polyBERT_482', 'polyBERT_174', 'polyBERT_356', 'polyBERT_305', 'polyBERT_181', 'polyBERT_347', 'polyBERT_217', 'polyBERT_371', 'polyBERT_184', 'polyBERT_364', 'polyBERT_282', 'polyBERT_354', 'polyBERT_590', 'polyBERT_44', 'polyBERT_161', 'polyBERT_59', 'polyBERT_144', 'polyBERT_345', 'polyBERT_180', 'polyBERT_470', 'polyBERT_129', 'polyBERT_469', 'polyBERT_526', 'polyBERT_148', 'polyBERT_125', 'polyBERT_113', 'polyBERT_248', 'polyBERT_351', 'polyBERT_92', 'polyBERT_24', 'polyBERT_43', 'polyBERT_473', 'polyBERT_52', 'polyBERT_409', 'polyBERT_349', 'polyBERT_220', 'polyBERT_274', 'polyBERT_98', 'polyBERT_103', 'polyBERT_53', 'polyBERT_587', 'polyBERT_487', 'polyBERT_543', 'polyBERT_412', 'polyBERT_67', 'polyBERT_230', 'polyBERT_321', 'polyBERT_283', 'polyBERT_30', 'polyBERT_136', 'polyBERT_104', 'polyBERT_445', 'polyBERT_419', 'polyBERT_73', 'polyBERT_456', 'polyBERT_123', 'polyBERT_64', 'polyBERT_333', 'polyBERT_386', 'polyBERT_265', 'polyBERT_472', 'polyBERT_164', 'polyBERT_481', 'polyBERT_128', 'polyBERT_489', 'polyBERT_198', 'polyBERT_568', 'polyBERT_406', 'polyBERT_547', 'polyBERT_20', 'polyBERT_389', 'polyBERT_271', 'polyBERT_225', 'polyBERT_167', 'polyBERT_35', 'polyBERT_579', 'polyBERT_551', 'polyBERT_278', 'polyBERT_139', 'polyBERT_272', 'polyBERT_166', 'polyBERT_398', 'polyBERT_133', 'polyBERT_515', 'polyBERT_8', 'polyBERT_83', 'polyBERT_80', 'polyBERT_462', 'polyBERT_378', 'polyBERT_317', 'polyBERT_310', 'polyBERT_268', 'polyBERT_599', 'polyBERT_14', 'polyBERT_279', 'polyBERT_79', 'polyBERT_254', 'polyBERT_411', 'polyBERT_39', 'polyBERT_329', 'polyBERT_185', 'polyBERT_519', 'polyBERT_25', 'polyBERT_107', 'polyBERT_521', 'polyBERT_120', 'polyBERT_435', 'polyBERT_152', 'polyBERT_385', 'polyBERT_197', 'polyBERT_71', 'polyBERT_443', 'polyBERT_353', 'polyBERT_480', 'polyBERT_126', 'polyBERT_342', 'polyBERT_121', 'polyBERT_77', 'polyBERT_459', 'polyBERT_18', 'polyBERT_583', 'polyBERT_359', 'polyBERT_110', 'polyBERT_13', 'polyBERT_323', 'polyBERT_69', 'polyBERT_296', 'polyBERT_306', 'polyBERT_595', 'polyBERT_341', 'polyBERT_255', 'polyBERT_563', 'polyBERT_175', 'polyBERT_505', 'polyBERT_533', 'polyBERT_539', 'polyBERT_245', 'polyBERT_522', 'polyBERT_250', 'polyBERT_431', 'polyBERT_414', 'polyBERT_173', 'polyBERT_224', 'polyBERT_381', 'polyBERT_149', 'polyBERT_455', 'polyBERT_114', 'polyBERT_249', 'polyBERT_421', 'polyBERT_426', 'polyBERT_240', 'polyBERT_365', 'polyBERT_172', 'polyBERT_117', 'polyBERT_318', 'polyBERT_485', 'polyBERT_211', 'polyBERT_467', 'polyBERT_355', 'polyBERT_382', 'polyBERT_464', 'polyBERT_256', 'polyBERT_4', 'polyBERT_460', 'polyBERT_520', 'polyBERT_346', 'polyBERT_390', 'polyBERT_393', 'polyBERT_62', 'polyBERT_576', 'polyBERT_232', 'polyBERT_524', 'polyBERT_115', 'polyBERT_131', 'polyBERT_484', 'polyBERT_334', 'polyBERT_377', 'polyBERT_303', 'polyBERT_535', 'polyBERT_31', 'polyBERT_142', 'polyBERT_222', 'polyBERT_407', 'polyBERT_263', 'polyBERT_315', 'polyBERT_159', 'polyBERT_326', 'polyBERT_592', 'polyBERT_111', 'polyBERT_565', 'polyBERT_162', 'polyBERT_529', 'polyBERT_569', 'polyBERT_340', 'polyBERT_518', 'polyBERT_235', 'polyBERT_97', 'polyBERT_446', 'polyBERT_150', 'polyBERT_399', 'polyBERT_422', 'polyBERT_241', 'polyBERT_516', 'polyBERT_187', 'polyBERT_137', 'polyBERT_195', 'polyBERT_561', 'polyBERT_12', 'polyBERT_538', 'polyBERT_116', 'polyBERT_236', 'polyBERT_252', 'polyBERT_525', 'polyBERT_401', 'polyBERT_564', 'polyBERT_34', 'polyBERT_105', 'polyBERT_413', 'polyBERT_11', 'polyBERT_106', 'polyBERT_234', 'polyBERT_530', 'polyBERT_277', 'polyBERT_405', 'polyBERT_450', 'polyBERT_438', 'polyBERT_213', 'polyBERT_588', 'polyBERT_420', 'polyBERT_212', 'polyBERT_90', 'polyBERT_439', 'polyBERT_528', 'polyBERT_596', 'polyBERT_292', 'polyBERT_258', 'polyBERT_261', 'polyBERT_430', 'polyBERT_311', 'polyBERT_17', 'polyBERT_26', 'polyBERT_96', 'polyBERT_143', 'polyBERT_119', 'polyBERT_130', 'polyBERT_532', 'polyBERT_395', 'polyBERT_542', 'polyBERT_559', 'polyBERT_33', 'polyBERT_593', 'polyBERT_427', 'polyBERT_294', 'polyBERT_48', 'polyBERT_452', 'polyBERT_573', 'polyBERT_585', 'polyBERT_523', 'polyBERT_362', 'polyBERT_553', 'polyBERT_140', 'polyBERT_269', 'polyBERT_448', 'polyBERT_331', 'polyBERT_231']
    features_df = features_df[ranked_features[:polybert_embedding_dim_count]]

    return features_df

def get_features_dataframe(
        smiles_df: pd.DataFrame, 
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int,
        use_extra_backbone_sidechain_features: bool,
        models_directory_path: str | None,
        predicted_features_detail_level: int,
        gemini_features_detail_level: int,
        polybert_embedding_dim_count: int
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
    # COMPUTE "STANDARD" FEATURES.
    features_df = _get_standard_features_dataframe(
        smiles_df, 
        morgan_fingerprint_dim,
        atom_pair_fingerprint_dim,
        torsion_dim,
        use_maccs_keys,
        use_graph_features,
        backbone_sidechain_detail_level,
        use_extra_backbone_sidechain_features,
        gemini_features_detail_level
    )

    # MAYBE COMPUTE PREDICTED SIMULATION RESULT FEATURES.
    if (models_directory_path is not None) and (predicted_features_detail_level > 0):
        predicted_features_df = _get_predicted_features_dataframe(smiles_df, models_directory_path)

        if predicted_features_detail_level == 1:
            IMPORTANT_FEATURE_NAMES = [
                'xgb_ffv',
                'xgb_density_g_cm3',
                'xgb_homopoly_NPR1',
                'xgb_homopoly_NPR2',
                'xgb_homopoly_Eccentricity',
                'xgb_homopoly_Asphericity',
                'xgb_homopoly_SpherocityIndex',
                'xgb_monomer_NPR2',
                'xgb_monomer_NPR1',
                'xgb_monomer_Asphericity',
                'xgb_lambda1_A2',
                'xgb_monomer_SpherocityIndex',
                'xgb_diffusivity_A2_per_ps',
                'xgb_p10_persistence_length',
                'xgb_homopoly_PBF',
                'xgb_lambda2_A2',
                'xgb_voxel_count_occupied',
                'xgb_homopoly_LabuteASA',
                'xgb_occupied_volume_A3',
                'xgb_box_volume_A3'
            ]
            predicted_features_df = predicted_features_df[IMPORTANT_FEATURE_NAMES]

        features_df = pd.concat([features_df, predicted_features_df], axis=1)

    # MAYBE COMPUTE polyBERT FINGERPRINTS.
    if polybert_embedding_dim_count > 0:
        polybert_embeddings_df = _get_poly_bert_embeddings(smiles_df, polybert_embedding_dim_count)
        features_df = pd.concat([features_df, polybert_embeddings_df], axis=1)

    return features_df

def get_tabular_predictions(smiles_csv_path):
    # LOAD MODELS.
    targets_to_preprocessing_configs, targets_to_model_groups = load_tabular_models()

    # LOAD DATA.
    test_df = pd.read_csv(smiles_csv_path)

    # INFERENCE.
    for target_name in TARGET_NAMES:
        # LOAD MODEL GROUPS.
        preprocessing_configs = targets_to_preprocessing_configs[target_name]
        model_groups = targets_to_model_groups[target_name]

        # GENERATE PREDICTIONS WITH EACH GROUP.
        model_groups_predictions = []
        for preprocessing_config, model_group in zip(preprocessing_configs, model_groups):
            # PREPROCESS DATA.
            features_df = get_features_dataframe(test_df, **preprocessing_config)

            # GENERATE PREDICTIONS.
            model_group_predictions = []
            for model in model_group:
                predictions = model.predict(features_df)
                model_group_predictions.append(predictions)

            # RECORD MEAN PREDICTIONS.
            model_group_predictions = np.mean(model_group_predictions, axis=0)
            model_groups_predictions.append(model_group_predictions)        

        # RECORD OVERALL AVERAGE.
        final_predictions = np.average(model_groups_predictions, axis=0)
        test_df[target_name] = final_predictions

    return test_df

#region BERT

class ContextPooler(nn.Module):
    def __init__(self, hidden_size, dropout_prob, activation_name):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = ACT2FN[activation_name]

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0] # Extract CLS token (first token)

        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertRegressor(nn.Module):
    def __init__(
            self, 
            pretrained_model_path, 
            context_pooler_kwargs = {'hidden_size': 384, 'dropout_prob': 0.144, 'activation_name': 'gelu'}):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_path)
        self.pooler = ContextPooler(**context_pooler_kwargs)
        
        # Final classification layer
        pooler_output_dim = context_pooler_kwargs['hidden_size']
        self.output = torch.nn.Linear(pooler_output_dim, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)        
        regression_output = self.output(pooled_output)

        return regression_output
    
def augment_smiles(smiles: str, n_augs: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [smiles]
    augmented = {smiles}
    for _ in range(n_augs * 2):
        if len(augmented) >= n_augs: break
        aug_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True); augmented.add(aug_smiles)
    return list(augmented)

def get_single_bert_predictions_df(test_df, root_finetuned_weights_path, foundation_model_path, hidden_size, fold_count):
    N_AUGMENTATIONS = 42
    RANDOM_STATE = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    tokenizer = AutoTokenizer.from_pretrained(foundation_model_path)
    
    bert_predictions_df = test_df.copy()
    for target in TARGET_NAMES:
        print(f"    Generating Test predictions with TTA for {target}...")
        all_models_target_preds = []
        for fold_id in range(fold_count):
            # FIND WEIGHTS.
            grouped_by_fold = os.path.exists(f'{root_finetuned_weights_path}/fold_{fold_id}')
            model_directory_path = f'{root_finetuned_weights_path}/fold_{fold_id}' if grouped_by_fold else root_finetuned_weights_path
            raw_state_dict = torch.load(f'{model_directory_path}/polymer_bert_v2_{target}.pth')
            
            # LOAD MODEL & SCALER.
            model = BertRegressor(
                foundation_model_path,
                context_pooler_kwargs={
                    "hidden_size": hidden_size,
                    "dropout_prob": 0.144,
                    "activation_name": "gelu",
                }
            )
            clean_state_dict = {
                key.removeprefix("_orig_mod."): tensor
                for key, tensor in raw_state_dict.items()
            }
            model.load_state_dict(clean_state_dict)
            
            model = model.to(DEVICE).eval()
            scaler = joblib.load(f'{model_directory_path}/scaler_{target}.pkl')
    
            # INFERENCE.
            target_preds = []
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                augmented_smiles_list = augment_smiles(row['SMILES'], N_AUGMENTATIONS)
                inputs = tokenizer(augmented_smiles_list, return_tensors='pt', truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad(): 
                    preds = model(**inputs)
                
                scaled_preds = preds.cpu().numpy(); 
                unscaled_preds = scaler.inverse_transform(scaled_preds).flatten(); 
                final_pred = np.median(unscaled_preds)
                target_preds.append(final_pred)
    
            all_models_target_preds.append(target_preds)
    
            # CLEANUP.
            del model, scaler
            gc.collect()
            torch.cuda.empty_cache()
    
        target_preds = np.mean(all_models_target_preds, axis = 0)
        bert_predictions_df[target] = target_preds
    return bert_predictions_df

def get_bert_predictions(smiles_csv_path, use_new_models):
    # LOAD DATA.
    test_df = pd.read_csv(smiles_csv_path)
    
    if not use_new_models:
        model_configs = [
            {
                "root_finetuned_weights_path": "models/20250906_191941_modern_et_8",
                "foundation_model_path": "answerdotai/ModernBERT-base",
                "hidden_size": 768,
                "fold_count": 1,
            },
            {
                "root_finetuned_weights_path": "models/20250906_191750_codebert_et_8",
                "foundation_model_path": "microsoft/codebert-base",
                "hidden_size": 768,
                "fold_count": 1,
            },
            {
                "root_finetuned_weights_path": "models/20250906_191852_poly_et_32",
                "foundation_model_path": "kuelumbus/polyBERT",
                "hidden_size": 600,
                "fold_count": 1,
            },
        ]
    else:
        model_configs = [
            {
                "root_finetuned_weights_path": "models/20250906_191941_modern_et_8",
                "foundation_model_path": "answerdotai/ModernBERT-base",
                "hidden_size": 768,
                "fold_count": 1,
            },
            {
                "root_finetuned_weights_path": "models/20250906_191750_codebert_et_8",
                "foundation_model_path": "microsoft/codebert-base",
                "hidden_size": 768,
                "fold_count": 1,
            },
        ]
    submission_dfs = []
    for model_config in model_configs:
        predictions_df = get_single_bert_predictions_df(test_df, **model_config)
        submission_dfs.append(predictions_df)

    bert_submission_df = submission_dfs[0].copy()

    # Compute the unweighted mean for each target variable
    for target_variable in TARGET_NAMES:
        bert_submission_df[target_variable] = (
            sum(df[target_variable] for df in submission_dfs) / len(submission_dfs)
        )

    return bert_submission_df

#region D-MPNN

def _maybe_dropout(rate: float) -> nn.Module:
    return nn.Dropout(rate) if rate > 0.0 else nn.Identity()

def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """
    Fully-featured replacement for torch_scatter.scatter_add.
    Allocates a zero tensor of shape (dim_size, src.size(1)) and
    accumulates `src` rows whose destinations are in `index`.
    """
    out = torch.zeros(
        dim_size,
        src.size(1),
        dtype = src.dtype,
        device = src.device,
    )
    out.scatter_add_(
        dim = 0,
        index = index.unsqueeze(-1).expand_as(src),
        src = src,
    )
    return out

class BasicDMPNN(nn.Module):
    def __init__(
        self,
        atom_emb: int,
        msg_dim: int,
        msg_passes: int,
        out_hidden: int,
        bond_emb: int = 16,
        emb_drop: float = 0,
        msg_drop: float = 0,
        head_drop: float = 0,
    ):
        super().__init__()
        self.atom_embeddings = nn.Sequential(
            nn.Embedding(119, atom_emb),
            _maybe_dropout(emb_drop),
        )
        self.bond_embeddings = nn.Sequential(
            nn.Embedding(4, bond_emb),
            _maybe_dropout(emb_drop),
        )
        self.msg_init = nn.Linear(atom_emb + bond_emb, msg_dim)
        self.msg_update = nn.Linear(atom_emb + bond_emb + msg_dim, msg_dim)

        self.msg_passes = msg_passes
        self.msg_dropout = _maybe_dropout(msg_drop)

        self.readout = nn.Sequential(
            _maybe_dropout(head_drop),
            nn.Linear(msg_dim, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 1),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        atom = self.atom_embeddings(data.x)          # (num_atoms, atom_emb)
        bond = self.bond_embeddings(data.edge_attr)  # (num_edges, bond_emb)
        src, dst = data.edge_index                   # (2, num_edges)

        # 1. initial edge → message
        msg = F.relu(self.msg_init(torch.cat([atom[src], bond], dim=1)))

        # 2. message-passing iterations
        for _ in range(self.msg_passes):
            agg = _scatter_sum(msg, dst, dim_size=atom.size(0))
            upd_in = torch.cat([atom[src], bond, agg[src]], dim=1)
            msg = F.relu(self.msg_update(upd_in))
            msg = self.msg_dropout(msg)

        # 3. node & molecule readout
        node_state = _scatter_sum(msg, dst, dim_size=atom.size(0))
        num_molecules = int(data.batch.max().item()) + 1
        mol_state = _scatter_sum(node_state, data.batch, dim_size=num_molecules)

        return self.readout(mol_state).squeeze(-1)

def load_d_mpnn_models():
    MODEL_DIRECTORY_PATHS = [
        'models/d_mpnn_Tg_435959_retuned_extra',
        'models/d_mpnn_FFV_43_retuned_extra',
        'models/d_mpnn_Tc_242_retuned_extra',
        'models/d_mpnn_Density_225_retuned_extra',
        'models/d_mpnn_Rg_13631_retuned_extra',
    ]

    TARGETS_TO_CONFIGS = { # Tuned with extra data.
        "Tg": { # 43.595
            "atom_emb": 455,
            "msg_dim": 97,
            "out_hidden": 128,
            "msg_passes": 3,
            "emb_drop": 0.07971650425969778,
            "head_drop": 0.07706917345977361,
        },
        "FFV": { # 0.00438
            "atom_emb": 135,
            "msg_dim": 956,
            "out_hidden": 256,
            "msg_passes": 6,
            "head_drop": 0.18157995019962409,
        },
        "Tc": { # 0.02428
            "atom_emb": 54,
            "msg_dim": 455,
            "out_hidden": 256,
            "msg_passes": 8,
            "emb_drop": 0.06153220662036694,
            "msg_drop": 0.08610554567427026,
            "head_drop": 0.11941615204224441,
        },
        "Density": { # 0.02257
            "atom_emb": 102,
            "msg_dim": 395,
            "out_hidden": 224,
            "msg_passes": 5,
            "msg_drop": 0.07305572012066236,
        },
        "Rg": { # 1.3631
            "atom_emb": 26,
            "msg_dim": 833,
            "out_hidden": 256,
            "msg_passes": 5,
            "emb_drop": 0.12492099902514082,
            "head_drop": 0.20503078366943217,
        }
    }

    targets_to_models = {}
    targets_to_scalers = {}
    for model_directory_path in MODEL_DIRECTORY_PATHS:
        target_name = model_directory_path.split('/')[-1].split('_')[2]
        targets_to_models[target_name] = []
        targets_to_scalers[target_name] = []
        
        model_paths = glob.glob(f'{model_directory_path}/*.pth')
        for model_path in model_paths:
            model = BasicDMPNN(**TARGETS_TO_CONFIGS[target_name]).cuda()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            targets_to_models[target_name].append(model)

        scaler_paths = glob.glob(f'{model_directory_path}/*.pkl')
        for scaler_path in scaler_paths:
            scaler = joblib.load(scaler_path)
            targets_to_scalers[target_name].append(scaler)

        print(f'Loaded {len(targets_to_models[target_name])} models and {len(targets_to_scalers[target_name])} scalers for {target_name}')

    return targets_to_models, targets_to_scalers

def bond_type_to_int(bond: Chem.Bond) -> int:
    mapping = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    return mapping.get(bond.GetBondType(), 0)

def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_indices, edge_attributes = [], []
    for bond in mol.GetBonds():
        start_index, end_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_type_to_int(bond)
        edge_indices += [(start_index, end_index), (end_index, start_index)]
        edge_attributes += [bond_type, bond_type]
    return Data(
        x=torch.tensor(atom_nums, dtype=torch.long),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t(),
        edge_attr=torch.tensor(edge_attributes, dtype=torch.long),
    )

@torch.no_grad()
def get_predictions(model, dataloader, scaler):
    predictions = []
    for batch in dataloader:
        batch = batch.cuda()
        batch_predictions = model(batch)
        predictions.extend(batch_predictions.cpu().numpy())

    predictions = np.array(predictions).reshape(-1,1)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def get_d_mpnn_predictions(smiles_csv_path):
    # PREPARE DATA.
    test_df = pd.read_csv(smiles_csv_path)
    test_graphs = [smiles_to_graph(smiles) for smiles in tqdm(test_df.SMILES, desc='Creating graphs')]
    test_dataloader = DataLoader(test_graphs, shuffle=False, batch_size=64)

    # LOAD MODELS.
    targets_to_models, targets_to_scalers = load_d_mpnn_models()

    # INFERENCE.
    for target_name in targets_to_models.keys():
        ensemble_predictions = []
        models = targets_to_models[target_name]
        scalers = targets_to_scalers[target_name]
        for model, scaler in zip(models, scalers):
            model_predictions = get_predictions(model, test_dataloader, scaler)
            ensemble_predictions.append(model_predictions)

        ensemble_predictions = np.mean(ensemble_predictions, axis=0)
        test_df[target_name] = ensemble_predictions

    return test_df

#region Uni-Mol 2

def can_embed(smiles_string: str) -> bool:
    """
    Return True only if RDKit can parse the SMILES *and*
    `AllChem.EmbedMolecule` succeeds (status == 0).

    Any parsing, sanitisation, or embedding error ⇒ False.
    """
    try:
        molecule = Chem.MolFromSmiles(smiles_string)

        if molecule.GetNumAtoms(onlyExplicit=False) > 110: # FFV only (130 OOM)
            return False

        if molecule is None:
            return False
            
        embed_status: int = AllChem.EmbedMolecule(
            molecule,
            maxAttempts=5,
            clearConfs=True,
        )
        return embed_status == 0
    except:
        # traceback.print_exc()
        return False

'''
def get_unimol_predictions(input_filepath):
    UNIMOL_TARGETS_TO_PATHS = {
        'Rg': 'models/UniMol2_2025_08_17_TabM/Rg',
        'Tc': 'models/UniMol2_2025_09_07_TabM/Tc',
        'Tg': 'models/UniMol2_2025_08_17_TabM/Tg',
        'Density': 'models/UniMol2_2025_08_17_TabM/Density',
    }

    input_df = pl.read_csv(input_filepath)
    uni_mol_submission_df = pd.DataFrame([
        {'id': row_index, 'SMILES': smiles, 'Rg': np.nan, 'Tc': np.nan, 'Tg': np.nan, 'Density': np.nan}
        for row_index, smiles in enumerate(input_df['SMILES'].to_list())
    ])
    test_df = pl.DataFrame(uni_mol_submission_df)
    uni_mol_submission_df = uni_mol_submission_df.set_index('id')

    for target_name, predictor_path in UNIMOL_TARGETS_TO_PATHS.items():
        # PREPROCESS DATA.
        subset_df = (
            test_df
            .filter(
                pl.col("SMILES").map_elements(
                    can_embed,
                    return_dtype=pl.Boolean,
                )
            )
            ['id', 'SMILES']
        )
        preprocessed_data_path = f'{target_name}_SMILES.csv'
        subset_df.write_csv(preprocessed_data_path)

        # LOAD MODEL(s).
        print('Predictor path:', predictor_path)
        predictor = MolPredict(load_model=predictor_path)

        # INFERENCE.
        predictions = predictor.predict(data = preprocessed_data_path)

        # UPDATE SUBMISSION.
        new_prediction_series = pd.Series(
            predictions.ravel(),       # → 1‑D
            index=subset_df['id'].to_list(),
            name=target_name,
            dtype="float64",
        )
        uni_mol_submission_df[target_name] = np.nan
        uni_mol_submission_df[target_name].update(new_prediction_series)

        # CLEANUP.
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    return uni_mol_submission_df
'''
    
def get_unimol_predictions(input_filepath):
    UNIMOL_TARGETS_TO_PATHS = {
        'Rg': ['models/UniMol2_2025_08_17_TabM/Rg'],
        'Tc': [
            'models/UniMol2_2025_08_17_TabM/Tc',
            'models/UniMol2_2025_09_07_TabM/Tc'
        ],
        'Tg': ['models/UniMol2_2025_08_17_TabM/Tg'],
        'Density': [
            'models/UniMol2_2025_08_17_TabM/Density',
            'models/UniMol2_2025_09_07_TabM/Density',
        ],
    }
    CONFORMER_GENERATION_SEEDS = [
        9513,
        5492
    ]

    input_df = pl.read_csv(input_filepath)
    uni_mol_submission_df = pd.DataFrame([
        {'id': row_index, 'SMILES': smiles, 'Rg': np.nan, 'Tc': np.nan, 'Tg': np.nan, 'Density': np.nan}
        for row_index, smiles in enumerate(input_df['SMILES'].to_list())
    ])
    test_df = pl.DataFrame(uni_mol_submission_df)
    uni_mol_submission_df = uni_mol_submission_df.set_index('id')

    for target_name, predictor_paths in UNIMOL_TARGETS_TO_PATHS.items():
        # PREPROCESS DATA.
        subset_df = (
            test_df
            .filter(
                pl.col("SMILES").map_elements(
                    can_embed,
                    return_dtype=pl.Boolean,
                )
            )
            ['id', 'SMILES']
        )
        preprocessed_data_path = f'{target_name}_SMILES.csv'
        subset_df.write_csv(preprocessed_data_path)

        # INFERENCE.
        all_predictions = []
        for seed_index, seed in enumerate(CONFORMER_GENERATION_SEEDS):
            # LOAD MODEL.
            model_index = seed_index % len(predictor_paths)
            model_path = predictor_paths[model_index]
            predictor = MolPredict(load_model=model_path)

            print(f'Generating predictions with model = {model_path}, seed = {seed}')

            # GENERATE PREDICTIONS
            predictor.config['seed'] = seed
            predictions = predictor.predict(data = preprocessed_data_path)
            flattend_predictions = predictions.ravel()
            all_predictions.append(flattend_predictions)

            # CLEANUP.
            del predictor
            gc.collect()
            torch.cuda.empty_cache()

            # MAYBE EXIT EARLY.
            if (seed_index + 1) >= max(len(np.unique(CONFORMER_GENERATION_SEEDS)), len(predictor_paths)):
                break

        # UPDATE SUBMISSION.
        averaged_predictions = np.mean(all_predictions, axis=0)
        new_prediction_series = pd.Series(
            averaged_predictions,
            index=subset_df['id'].to_list(),
            name=target_name,
            dtype="float64",
        )
        uni_mol_submission_df[target_name] = np.nan
        uni_mol_submission_df[target_name].update(new_prediction_series)

    return uni_mol_submission_df

#region Main

def combine_weighted_columns(
        df_always: pd.DataFrame,
        df_sometimes: pd.DataFrame,
        key_column: str,
        value_column: str,
        weight_always: float,
        weight_sometimes: float) -> pd.DataFrame:
    merged = df_always.merge(
        df_sometimes,
        on=key_column,
        suffixes=("_always", "_sometimes")
    )

    col_always = f"{value_column}_always"
    col_sometimes = f"{value_column}_sometimes"

    merged[value_column] = np.where(
        merged[col_sometimes].notna(),
        weight_always * merged[col_always] + weight_sometimes * merged[col_sometimes],
        merged[col_always]
    )

    return merged[[key_column, value_column]]

if __name__ == '__main__':
    DATA_FILENAMES = [
        # 'train_tc-natsume_full-dmitry_extra.csv',
        # 'train_host_extra.csv',
        # 'train_host_plus_leaks_extra.csv',
        # 'PI1070.csv'
        # 'LAMALAB_CURATED_Tg_structured_polymerclass.csv',
        # 'PI1M_500.csv',
        # 'PI1M_5000.csv',
        'PI1M_50000.csv',
        # 'realistic_50k_full.csv'
    ]
    for filename in DATA_FILENAMES:
        # GET PREDICTIONS.
        # input_filepath = f'data/from_host_v2/{filename}'
        # input_filepath = f'data/RadonPy/data/{filename}'
        # input_filepath = f'data/{filename}'
        input_filepath = f'data/PI1M/{filename}'
        # input_filepath = f'data/polyOne_partitioned_mini/{filename}'

        basic_bert_predictions_df = get_bert_predictions(input_filepath, use_new_models=False)
        pseudo_bert_predictions_df = get_bert_predictions(input_filepath, use_new_models=True)
        unimol_predictions_df = get_unimol_predictions(input_filepath)
        # d_mpnn_predictions_df = get_d_mpnn_predictions(input_filepath)
        tabular_predictions_df = get_tabular_predictions(input_filepath)
        
        # ENSEMBLE PREDICTIONS.
        predictions_df = tabular_predictions_df.copy()
        for target_name in TARGET_NAMES:
            # predictions_df[target_name] = (4*tabular_predictions_df[target_name] + 4*bert_predictions_df[target_name] + 3*d_mpnn_predictions_df[target_name]) / (4+4+3)
            # predictions_df[target_name] = (tabular_predictions_df[target_name] + bert_predictions_df[target_name]) / 2
            predictions_df[target_name] = (1*tabular_predictions_df[target_name] + 1*basic_bert_predictions_df[target_name] + 3*pseudo_bert_predictions_df[target_name])/5

            if target_name in unimol_predictions_df.columns:
                print('Pre Uni-Mol:\n', predictions_df)
                merged_df = combine_weighted_columns(
                    df_always = predictions_df,
                    df_sometimes = unimol_predictions_df,
                    key_column = 'SMILES',
                    value_column = target_name,
                    # weight_always = (4+4+3) / (6+4+4+3),
                    # weight_sometimes = 6 / (6+4+4+3)
                    # weight_always = 0.4,
                    # weight_sometimes = 0.6
                    weight_always = (5.0 / 6.0),
                    weight_sometimes = (1.0 / 6.0)
                )
                print('\nMerged:\n', merged_df)
                predictions_df[target_name] = merged_df[target_name]
                print('\nPost Uni-Mol:\n', predictions_df)
            
        # SAVE PREDICTIONS.
        output_filepath = f'data_filtering/relabeled_datasets/{filename}'.replace('.csv', '_v3.csv')
        predictions_df.to_csv(output_filepath, index=False)