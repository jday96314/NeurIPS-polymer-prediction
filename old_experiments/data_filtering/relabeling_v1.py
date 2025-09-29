# from autogluon.tabular import TabularPredictor
import pickle
import json
import glob
import polars as pl
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator, AllChem, rdmolops, rdMolDescriptors, rdchem
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

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

TARGET_NAMES = ["Tg", "FFV", "Tc", "Density", "Rg"]

#region Tabular

def load_tabular_models():
    MODEL_DIRECTORIES = [
        # 'models/LGBMRegressor_20250714_195253',
        # 'models/XGBRegressor_20250714_194646',
        # 'models/CatBoostRegressor_20250714_212440',
        # 'models/TabularPredictor_20250715_190532',
        'models/LGBMRegressor_20250729_074514',
        'models/XGBRegressor_20250729_080504',
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
    except nx.NetworkXNoPath:               # ← add this block
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

    return features

@lru_cache(maxsize=1_000_000)
def get_feature_vector(
        smiles: str,
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int = 0):
    # PARSE SMILES.
    mol = Chem.MolFromSmiles(smiles)
    
    # GET DESCRIPTORS.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    descriptor_generator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = np.array(descriptor_generator.CalcDescriptors(mol))

    # GET MORGAN FINGERPRINT.
    morgan_fingerprint = np.array([])
    if morgan_fingerprint_dim > 0:
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan_fingerprint_dim)
        morgan_fingerprint = list(morgan_generator.GetFingerprint(mol))

    # GET ATOM PAIR FINGERPRINT.
    atom_pair_fingerprint = np.array([])
    if atom_pair_fingerprint_dim > 0:
        atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=atom_pair_fingerprint_dim)
        atom_pair_fingerprint = list(atom_pair_generator.GetFingerprint(mol))

    # GET MACCS.
    maccs_keys = np.array([])
    if use_maccs_keys:
        maccs_keys = MACCSkeys.GenMACCSKeys(mol)
        maccs_keys = list(maccs_keys)

    # GET TORSION FINGERPRINT.
    torsion_fingerprint = np.array([])
    if torsion_dim > 0:
        torsion_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=torsion_dim)
        torsion_fingerprint = list(torsion_generator.GetFingerprint(mol))

    # GET GRAPH FEATURES.
    graph_features = []
    if use_graph_features:
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
        graph = nx.from_numpy_array(adjacency_matrix)
        graph_diameter = nx.diameter(graph) if nx.is_connected(graph) else 0
        avg_shortest_path = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 0
        cycle_count = len(list(nx.cycle_basis(graph)))
        graph_features = [graph_diameter, avg_shortest_path, cycle_count]

    # GET SIDECHAIN & BACKBONE FEATURES.
    if backbone_sidechain_detail_level == 0:
        sidechain_backbone_features = []
    elif backbone_sidechain_detail_level == 1:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES]
    elif backbone_sidechain_detail_level == 2:
        sidechain_backbone_features = extract_sidechain_and_backbone_features(smiles)
        sidechain_backbone_features = [sidechain_backbone_features[name] for name in ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES]
    else:
        assert False, f'Invalid backbone vs. sidechain detail level: {backbone_sidechain_detail_level}'

    # CONCATENATE FEATURES.
    features = np.concatenate([
        descriptors, 
        morgan_fingerprint, 
        atom_pair_fingerprint, 
        maccs_keys, 
        torsion_fingerprint,
        graph_features,
        sidechain_backbone_features
    ])
    return features

def get_features_dataframe(
        smiles_df: pd.DataFrame, 
        morgan_fingerprint_dim: int,
        atom_pair_fingerprint_dim: int,
        torsion_dim: int,
        use_maccs_keys: bool,
        use_graph_features: bool,
        backbone_sidechain_detail_level: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    # GET FEATURE NAMES.
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    morgan_col_names = [f'mfp_{i}' for i in range(morgan_fingerprint_dim)]
    atom_pair_col_names = [f'ap_{i}' for i in range(atom_pair_fingerprint_dim)]
    maccs_col_names = [f'maccs_{i}' for i in range(167)] if use_maccs_keys else []
    torsion_col_names = [f'tt_{i}' for i in range(torsion_dim)]
    graph_col_names = ['graph_diameter', 'avg_shortest_path', 'num_cycles'] if use_graph_features else []
    sidechain_col_names = [[], IMPORTANT_SIDECHAIN_BACKBONE_FEATURE_NAMES, ALL_SIDECHAIN_BACKBONE_FEATURE_NAMES][backbone_sidechain_detail_level]
    feature_col_names = descriptor_names + morgan_col_names + atom_pair_col_names + maccs_col_names + torsion_col_names + graph_col_names + sidechain_col_names

    # GET FEATURES.
    features_df = pd.DataFrame(
        np.vstack([
            get_feature_vector(
                smiles,
                morgan_fingerprint_dim,
                atom_pair_fingerprint_dim,
                torsion_dim,
                use_maccs_keys,
                use_graph_features,
                backbone_sidechain_detail_level
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
            # token_type_ids=None,
            position_ids=None):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
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

def get_bert_predictions(smiles_csv_path):
    # LOAD DATA.
    test_df = pd.read_csv(smiles_csv_path)
    
    # INFERENCE.
    # ROOT_FINETUNED_WEIGHTS_PATH = 'models/20250704_140813_chemberta'
    # FOUNDATION_MODEL_PATH = 'DeepChem/ChemBERTa-77M-MTR'
    ROOT_FINETUNED_WEIGHTS_PATH = 'models/20250728_083626_modern_et_8'
    FOUNDATION_MODEL_PATH = 'answerdotai/ModernBERT-base'
    HIDDEN_SIZE = 768
    FOLD_COUNT = 5
    DEVICE = 'cuda'
    N_AUGMENTATIONS = 42

    tokenizer = AutoTokenizer.from_pretrained(FOUNDATION_MODEL_PATH)

    bert_submission_df = test_df.copy()
    for target in TARGET_NAMES:
        print(f"    Generating Test predictions with TTA for {target}...")
        all_models_target_preds = []
        for fold_id in range(FOLD_COUNT):
            model = BertRegressor(
                FOUNDATION_MODEL_PATH,
                context_pooler_kwargs={
                    "hidden_size": HIDDEN_SIZE,
                    "dropout_prob": 0.144,
                    "activation_name": "gelu",
                }
            )
            raw_state_dict = torch.load(f'{ROOT_FINETUNED_WEIGHTS_PATH}/fold_{fold_id}/polymer_bert_v2_{target}.pth')
            clean_state_dict = {
                key.removeprefix("_orig_mod."): tensor
                for key, tensor in raw_state_dict.items()
            }
            model.load_state_dict(clean_state_dict)
            
            model = model.to(DEVICE).eval()
            scaler = joblib.load(f'{ROOT_FINETUNED_WEIGHTS_PATH}/fold_{fold_id}/scaler_{target}.pkl')
            
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

            del model, scaler
            gc.collect()
            torch.cuda.empty_cache()

        target_preds = np.mean(all_models_target_preds, axis = 0)
        bert_submission_df[target] = target_preds

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
        bond_emb: int,
        msg_dim: int,
        msg_passes: int,
        out_hidden: int,
        emb_drop: float,
        msg_drop: float,
        head_drop: float,
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
        # 'models/d_mpnn_Tg_502011',
        # 'models/d_mpnn_FFV_45',
        # 'models/d_mpnn_Tc_255',
        # 'models/d_mpnn_Density_237',
        # 'models/d_mpnn_Rg_14906',
        'models/d_mpnn_Tg_443307_tuned_extra',
        'models/d_mpnn_FFV_43_manual_extra',
        'models/d_mpnn_Tc_243_tuned_extra',
        'models/d_mpnn_Density_224_tuned_extra',
        'models/d_mpnn_Rg_13707_tuned_extra',
    ]

    TARGETS_TO_CONFIGS = {
        "Tg": { # Rerun MAE = 50.201104736328126
            "atom_emb": 40,
            "bond_emb": 16,
            "msg_dim": 242,
            "out_hidden": 62,
            "msg_passes": 5,
            "emb_drop": 0,
            "msg_drop": 0,
            "head_drop": 0.2691323902671593,
        },
        "FFV": { # Rerun MAE = 0.004520646389573812
            'atom_emb': 131, 
            "bond_emb": 16,
            'msg_dim': 515, 
            'out_hidden': 608, 
            'msg_passes': 7, 
            'emb_drop': 0.055918540183476806, 
            'msg_drop': 0, 
            'head_drop': 0.05, 
        },
        "Tc": { # Rerun MAE = 0.025551460683345795
            "atom_emb": 545,
            "bond_emb": 16,
            "msg_dim": 723,
            "out_hidden": 222,
            "msg_passes": 5,
            "emb_drop": 0,
            "msg_drop": 0.20329077329185974,
            "head_drop": 0.10640722466508153,
        },
        "Density": { # Rerun MAE = 0.023745716735720634
            "atom_emb": 31,
            "bond_emb": 16,
            "msg_dim": 305,
            "out_hidden": 786,
            "msg_passes": 5,
            "emb_drop": 0,
            "msg_drop": 0,
            "head_drop": 0,
        },
        "Rg": { # Rerun MAE = 1.490615677833557
            "atom_emb": 51,
            "bond_emb": 16,
            "msg_dim": 926,
            "out_hidden": 369,
            "msg_passes": 4,
            "emb_drop": 0,
            "msg_drop": 0,
            "head_drop": 0.10354626111613492,
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
    
def get_unimol_predictions(input_filepath):
    UNIMOL_TARGETS_TO_PATHS = {
        'Rg': 'models/UniMol2_2025_08_17_TabM/Rg',
        'Tc': 'models/UniMol2_2025_08_17_TabM/Tc',
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
        # 'PI1M_5000.csv',
        # 'PI1M_50000.csv',
        'realistic_50k_full.csv'
    ]
    for filename in DATA_FILENAMES:
        # GET PREDICTIONS.
        # input_filepath = f'data/from_host_v2/{filename}'
        # input_filepath = f'data/RadonPy/data/{filename}'
        # input_filepath = f'data/{filename}'
        # input_filepath = f'data/PI1M/{filename}'
        input_filepath = f'data/polyOne_partitioned_mini/{filename}'
        bert_predictions_df = get_bert_predictions(input_filepath)
        d_mpnn_predictions_df = get_d_mpnn_predictions(input_filepath)
        tabular_predictions_df = get_tabular_predictions(input_filepath)
        unimol_predictions_df = get_unimol_predictions(input_filepath)
        
        # ENSEMBLE PREDICTIONS.
        predictions_df = tabular_predictions_df.copy()
        for target_name in TARGET_NAMES:
            predictions_df[target_name] = (4*tabular_predictions_df[target_name] + 4*bert_predictions_df[target_name] + 3*d_mpnn_predictions_df[target_name]) / (4+4+3)

            if target_name in unimol_predictions_df.columns:
                print('Pre Uni-Mol:\n', predictions_df)
                merged_df = combine_weighted_columns(
                    df_always = predictions_df,
                    df_sometimes = unimol_predictions_df,
                    key_column = 'SMILES',
                    value_column = target_name,
                    weight_always = (4+4+3) / (6+4+4+3),
                    weight_sometimes = 6 / (6+4+4+3)
                )
                print('\nMerged:\n', merged_df)
                predictions_df[target_name] = merged_df[target_name]
                print('\nPost Uni-Mol:\n', predictions_df)
            
        # SAVE PREDICTIONS.
        output_filepath = f'data_filtering/relabeled_datasets/{filename}'
        predictions_df.to_csv(output_filepath, index=False)