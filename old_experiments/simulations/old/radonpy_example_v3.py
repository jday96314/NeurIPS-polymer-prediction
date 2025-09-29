import numpy as np
from radonpy.core import utils, poly
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors, Descriptors3D
from typing import Dict, List
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from radonpy.ff.gaff2_mod import GAFF2
from radonpy.sim import qm
from multiprocessing import Pool

def build_chain_once(monomer: Chem.Mol,
                     terminator: Chem.Mol,
                     natoms_target: int) -> Chem.Mol:
    n_repeat_units: int = poly.calc_n_from_num_atoms(
        monomer,
        natom = natoms_target,
        terminal1 = terminator
    )
    
    # ff = GAFF2()
    # qm.assign_charges(monomer, charge='gasteiger', work_dir='temp', tmp_dir='temp')
    chain: Chem.Mol = poly.polymerize_rw(
        monomer,
        n_repeat_units,
        # tacticity: isotactic, syndiotactic, or atactic
        tacticity='atactic',
        # opt='lammps',
        # ff=ff
    )
    chain = poly.terminate_rw(
        chain, 
        terminator,
        # opt='lammps',
        # ff=ff
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
        # params = params,
        # pruneRmsThresh = 0.35
    )

    # 2) MMFF-relax and score.
    energies: Dict[int, float] = {}
    mmff_props = AllChem.MMFFGetMoleculeProperties(monomer)
    # mmff_props = AllChem.UFFGetMoleculeProperties(monomer)
    for cid in conf_ids:
        AllChem.MMFFOptimizeMolecule(
        # AllChem.UFFOptimizeMolecule(
            monomer,
            confId = cid,
            mmffVariant = 'MMFF94s',
            maxIters = max_iters_minimisation
        )
        ff = AllChem.MMFFGetMoleculeForceField(monomer, mmff_props, confId = cid)
        # ff = AllChem.UFFGetMoleculeForceField(monomer, mmff_props, confId = cid)
        energies[cid] = ff.CalcEnergy()

    # 3) Stash the best conformer *before* wiping others.
    best_cid: int = min(energies, key = energies.get)
    best_conf: Chem.Conformer = Chem.Conformer(monomer.GetConformer(best_cid))

    # 4) Drop all conformers, then add the winner back.
    monomer.RemoveAllConformers()
    monomer.AddConformer(best_conf, assignId = True)

    return monomer

def descriptors_mean_over_ensemble(monomer_smiles: str,
                                   n_atoms_chain: int = 600,
                                   n_realisations: int = 2,
                                   ) -> Dict[str, float]:

    # embed_params = AllChem.ETKDGv3()
    # rdDistGeom.EmbedMolecule(monomer, embed_params)
    # # AllChem.MMFFOptimizeMolecule(monomer, maxIters = 500)
    # AllChem.UFFOptimizeMolecule(monomer, maxIters = 500)



    per_realisation: List[Dict[str, float]] = []
    for k in range(n_realisations):
        # monomer = Chem.MolFromSmiles(monomer_smiles)
        # Chem.AddHs(monomer)
        try:
            monomer = utils.mol_from_smiles(monomer_smiles)
        except:
            # radonpy internally assumes no wildcards will have double bonds, which is unsafe and can cause exceptions to be thrown, so we replace some double bonds with single bonds.
            sanitized_monomer_smiles = monomer_smiles.replace('*=','*-').replace('=*','-*')
            print(f'WARNING: Sanitized "{monomer_smiles}" -> "{sanitized_monomer_smiles}"')
            monomer = utils.mol_from_smiles(sanitized_monomer_smiles)

        monomer = get_lowest_energy_monomer(monomer)
        
        terminator = utils.mol_from_smiles('*C')
        # embed_params = AllChem.ETKDGv3()
        # rdDistGeom.EmbedMolecule(terminator, embed_params)
        # AllChem.MMFFOptimizeMolecule(terminator, maxIters = 400)

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
    return {key: float(np.median([d[key] for d in per_realisation]))
            for key in keys}

if __name__ == '__main__':
    TEST_SMILES = [
        # '*c1ccc(Oc2ccc(S(=O)(=O)c3ccc(Oc4ccc(-c5nc(-c6cccc(-c7nc(*)c(-c8ccccc8)[nH]7)c6)[nH]c5-c5ccccc5)cc4)cc3)cc2)cc1',
        '*=Nc1ccc(Oc2ccc(C(c3ccc(Oc4ccc(/N=C5/OC(=O)c6cc7c(cc65)C(=*)OC7=O)cc4)cc3)(C(F)(F)F)C(F)(F)F)cc2)cc1',
        '*Nc1ccc(N*)c2c3ccc(cc3)c12',
        '*CC(*)C(=O)Oc1ccccc1C',
        '*CC(*)C(=O)Oc1ccccc1',
        '*CC(C)S(*)(=O)=O',
        '*CC(*)c1cc(C(C)C)ccc1C(C)C',
        '*CC(*)CCCN(CC(C)C)CC(C)C',
        '*c1ccc(-c2ccc(C(*)(c3ccccc3)C(F)(F)F)cc2)cc1',
    ]
    ACTUAL_RG_VALUES = [
        0, 
        0, 
        12.7, 13.4, 21.0, 11.5, 9.7, 19.3
    ]

    # all_descriptors = [descriptors_mean_over_ensemble(smiles) for smiles in TEST_SMILES]
    with Pool(32) as worker_pool :
        all_descriptors = worker_pool.map(descriptors_mean_over_ensemble, TEST_SMILES)

    estimated_rgs = [
        descriptors['RadiusOfGyration']
        for descriptors in all_descriptors
    ]
    print('Estimated Rg values:', estimated_rgs)
    print('Pearson correlation:', pearsonr(estimated_rgs, ACTUAL_RG_VALUES)[0])
    print('MAE:', mean_absolute_error(estimated_rgs, ACTUAL_RG_VALUES))
    
    estimated_rgs = [
        descriptors['Rg_normalised']
        for descriptors in all_descriptors
    ]
    print('\nNormalized pearson correlation:', pearsonr(estimated_rgs, ACTUAL_RG_VALUES)[0])