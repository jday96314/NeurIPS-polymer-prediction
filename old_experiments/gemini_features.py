from __future__ import annotations

from typing import Dict
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem import rdPartialCharges
from joblib import Memory

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


if __name__ == '__main__':
    # --- Example Usage ---
    # Poly(methyl methacrylate) or PMMA repeat unit
    pmma_smiles = '[*]C(C)(C(=O)OC)C[*]'
    mol = Chem.MolFromSmiles(pmma_smiles.replace('[*]', '')) # Remove polymer markup for featurization

    if mol:
        calculated_features = compute_inexpensive_features(mol)

        print(f"Features for PMMA repeat unit ({Chem.MolToSmiles(mol)}):\n")
        for name, value in calculated_features.items():
            print(f"{name:<30}: {value:.4f}")

    # Example with a sulfone and ether group
    pes_smiles = 'c1ccc(S(=O)(=O)c2ccccc2)cc1'
    mol_pes = Chem.MolFromSmiles(pes_smiles)

    if mol_pes:
        print("\n" + "="*50 + "\n")
        calculated_features_pes = compute_inexpensive_features(mol_pes)
        print(f"Features for PES-like fragment ({pes_smiles}):\n")
        for name, value in calculated_features_pes.items():
            print(f"{name:<30}: {value:.4f}")

    print(list(calculated_features_pes.keys()))