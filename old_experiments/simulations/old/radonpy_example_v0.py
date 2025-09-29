#!/usr/bin/env python3
"""
Feature generators for polymer ML models.

Strategy-A  – “static”:  build a short chain, assign GAFF2_mod,
                         mine the FF parameters + cheap 3-D descriptors.

Strategy-B  – “flash MD”: re-use the FF / topology produced in A,
                           relax for a few ps in LAMMPS, then collect
                           trajectory-level features (Rg, MSD, energies).

Author: 2025-06-23
"""

from __future__ import annotations
import json
import pathlib
import tempfile
import time
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem import rdPartialCharges

# RadonPy imports (0.2.10)
from radonpy.core import utils as rutils, poly as rpoly
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim.lammps import LAMMPS


# ---------- user-facing helpers ------------------------------------------------

def _saturate_link_atoms_with_H(mol: Chem.Mol) -> Chem.Mol:
    """
    Convert every wildcard atom (*) at the oligomer ends into a hydrogen.
    Compatible with RadonPy 0.2.x where `cap_mol` is missing.
    """
    rw = Chem.RWMol(mol)

    link_indices = [at.GetIdx() for at in rw.GetAtoms() if at.GetAtomicNum() == 0]
    # 0-valence '*' atoms sit only at the two chain ends after polymerize_rw()
    for idx in sorted(link_indices, reverse=True):
        neighbour = rw.GetAtomWithIdx(rw.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx())

        # remove the dummy bond (*–C) …
        rw.RemoveBond(idx, neighbour.GetIdx())

        # … drop the dummy atom …
        rw.RemoveAtom(idx)

        # … and add a real hydrogen in its place
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(neighbour.GetIdx(), h_idx, Chem.BondType.SINGLE)

    mol_h = rw.GetMol()
    Chem.SanitizeMol(mol_h)
    return mol_h


def build_chain(repeat_unit_smiles: str,
                degree_polymerization: int = 20) -> Chem.Mol:
    """RadonPy-compatible chain with hydrogens already on the ends."""
    monomer = rutils.mol_from_smiles(repeat_unit_smiles)
    chain   = rpoly.polymerize_rw(monomer,
                                  degree_polymerization,
                                  tacticity="atactic")

    chain   = _saturate_link_atoms_with_H(chain)
    return chain


def assign_force_field(polymer: Chem.Mol) -> GAFF2_mod:
    """
    GAFF2_mod parameterisation that is fast on RadonPy 0.2.x
    (no Psi4, no QM).  We pre-populate Gasteiger charges and
    then call ff_assign() with no optional kwargs.
    """
    # 1️⃣ add cheap partial charges
    rdPartialCharges.ComputeGasteigerCharges(polymer)

    # 2️⃣ parameterise
    ff = GAFF2_mod()
    ff.ff_assign(polymer)          # ← no charge_method / qm flags needed
    return ff


def ff_stats(ff: GAFF2_mod) -> Dict[str, float]:
    """
    Robustly fetch GAFF2 tables on RadonPy 0.2.x.

    The ‘param’ container behaves like a dict whose keys look like
    {'bond', 'angle', 'dihedral', 'nonbond', ...} or the same in upper case.
    """
    p = ff.param                       # <class 'radonpy.core.utils.Container'>

    def _tbl(name_prefix: str):
        "Return the first table whose key starts with name_prefix (case-insensitive)."
        for k in p.keys():
            if k.lower().startswith(name_prefix):
                return p[k]
        raise KeyError(f"Could not find section '{name_prefix}' in ff.param")

    bond_tbl      = _tbl("bond")       # columns: k, r0
    angle_tbl     = _tbl("angle")      # columns: k, theta0
    dihedral_tbl  = _tbl("dihedral")   # columns: k, n, delta
    lj_tbl        = _tbl("nonbond")    # columns: epsilon, sigma
                                        #  (sometimes called 'pair' or 'vdw';
                                        #   the helper will still catch it)

    stats = {
        "bond_k_mean"      : float(bond_tbl["k"].mean()),
        "angle_k_mean"     : float(angle_tbl["k"].mean()),
        "dihedral_k_mean"  : float(dihedral_tbl["k"].mean()),
        "lj_sigma_mean"    : float(lj_tbl["sigma"].mean()),
        "lj_epsilon_mean"  : float(lj_tbl["epsilon"].mean()),
        "lj_sigma_std"     : float(lj_tbl["sigma"].std()),
        "lj_epsilon_std"   : float(lj_tbl["epsilon"].std()),
    }
    return stats

def rdkit_3d_descriptors(mol: Chem.Mol, max_attempts: int = 3) -> Dict[str, float]:
    """Generate a single 3-D conformer and compute cheap descriptors."""
    mol = Chem.AddHs(mol, addCoords=True)
    success = False
    for _ in range(max_attempts):
        if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) == 0:
            success = True
            break
    if not success:
        raise RuntimeError("RDKit could not generate a 3-D conformer.")
    # UFF minimisation (≈ 20–50 ms)
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    # Radius of gyration
    rg = float(np.sqrt(((coords - coords.mean(axis=0)) ** 2).sum(axis=1).mean()))
    # RDKit’s built-in 3-D descriptors
    pmi1 = Descriptors3D.Asphericity(mol)
    pmi2 = Descriptors3D.Eccentricity(mol)

    return {
        "rg_single_conf": rg,
        "asphericity": pmi1,
        "eccentricity": pmi2,
    }


def run_flash_md(
    force_field: GAFF2_mod,
    working_directory: pathlib.Path,
    *,
    n_steps: int = 1000,
    timestep_fs: float = 1.0,
    temperature_k: float = 300.0,
) -> Dict[str, float]:
    """
    Rapid NVT relax (~1 ps) to obtain dynamical features.
    Falls back to a no-op if the installed RadonPy/LAMMPS wrapper
    lacks any <something>system() builder (minimal PyPI wheels).
    """
    lmp = LAMMPS(
        work_dir=str(working_directory),
        mpi=1,
        omp=1,
        gpu=1,
        log_name="flash",
    )

    # -------- 1️⃣  pick whichever builder is available -------------------
    builders = [
        name for name in dir(lmp)
        if "system" in name.lower() and callable(getattr(lmp, name))
    ]
    if not builders:
        print(
            "RadonPy build lacks MD-system helpers "
            "(create_system / create_md_system). "
            "Skipping flash-MD feature generation."
        )
        return {}          # ← still lets the pipeline finish

    builder_name = (
        "create_md_system" if "create_md_system" in builders
        else "create_system" if "create_system" in builders
        else builders[0]                     # last-chance fallback
    )

    getattr(lmp, builder_name)(
        force_field.param,
        mol_type="single_chain",
        density=0.3,          # g cm⁻³ (loose box → quick relax)
        padding=10.0,         # Å
    )

    # -------- 2️⃣  run a tiny trajectory ---------------------------------
    lmp.set_integrator("nvt", temp=temperature_k, damp=100.0)
    lmp.run(n_steps, timestep_fs)

    # -------- 3️⃣  collect features --------------------------------------
    epot = np.asarray(lmp.trajectory["pe"])
    rg   = np.asarray(lmp.trajectory["Rg"])
    msd  = np.asarray(lmp.trajectory["msd"])

    h = epot.size // 2                      # discard first half (settling)

    features = {
        "flash_rg_mean": float(rg[h:].mean()),
        "flash_rg_std":  float(rg[h:].std()),
        "epot_mean":     float(epot[h:].mean()),
        "epot_std":      float(epot[h:].std()),
        "msd_final":     float(msd[-1]),
    }

    lmp.finalize()
    return features


# ---------- one-shot driver ----------------------------------------------------

def features_for_polymer(smiles: str) -> Dict[str, float]:
    """
    One call = ≤ 20 s.  Returns merged feature dict from Strategy-A + B.
    """
    start = time.perf_counter()

    chain = build_chain(smiles)                 # ≈ 0.4 s
    ff = assign_force_field(chain)              # ≈ 2–3 s

    feats: Dict[str, float] = {}
    # feats.update(ff_stats(ff))                  # ≈ 1 ms
    feats.update(rdkit_3d_descriptors(chain))   # ≈ 0.05 s

    # “flash MD” (~ 10 s on T4; comment out if too tight)
    with tempfile.TemporaryDirectory() as td:
        feats.update(run_flash_md(ff, pathlib.Path(td)))

    feats["wall_time_s"] = round(time.perf_counter() - start, 2)
    return feats


# ---------- quick demo ---------------------------------------------------------

if __name__ == "__main__":
    ps_smiles = "*C(C*)c1ccccc1"            # polystyrene repeat unit
    data = features_for_polymer(ps_smiles)
    print(json.dumps(data, indent=2))
