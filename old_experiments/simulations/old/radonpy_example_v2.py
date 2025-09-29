from radonpy.core import utils, poly
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors, Descriptors3D
from radonpy.ff.gaff2_mod import GAFF2
from radonpy.sim import qm

def compute_all_3d_descriptors(input_molecule: Chem.Mol) -> dict[str, float]:
    descriptor_names_to_raw_values: dict[str, float] = Descriptors3D.CalcMolDescriptors3D(input_molecule)

    # Ensure plain Python floats (rdkit *may* return numpy.float64).
    descriptor_names_to_cleaned_values = {
        name: float(value) 
        for name, value in 
        descriptor_names_to_raw_values.items()
    }
    return descriptor_names_to_cleaned_values

if __name__ == '__main__':
    # CREATE POLYMER.
    # smiles = '*CC(*)c1ccccc1'
    smiles = '*CC(*)C(=O)Oc1ccccc1C' # Rg = 12.7, 9.0, 16.0
    # smiles = '*CC(*)C(=O)Oc1ccccc1' # Rg = 13.4, 8.8, 8.6
    # smiles = '*CC(C)S(*)(=O)=O' # Rg = 21.0, 15.4, 14.0
    # smiles = '*CC(*)c1cc(C(C)C)ccc1C(C)C' # Rg = 11.5, 8.5, 9.3
    # smiles = '*CC(*)CCCN(CC(C)C)CC(C)C' # Rg = 9.7, 8.5, 10.1
    # smiles = '*c1ccc(-c2ccc(C(*)(c3ccccc3)C(F)(F)F)cc2)cc1' # Rg = 19.3, 8.6, 22.2
    monomer = utils.mol_from_smiles(smiles)

    embed_params = AllChem.ETKDGv3()
    rdDistGeom.EmbedMolecule(monomer, embed_params)
    AllChem.MMFFOptimizeMolecule(monomer, maxIters = 500)

    ########################
    # RADON_DIR_PATH = 'temp'
    # monomer_force_field = GAFF2()
    # # mol, energy = qm.conformation_search(monomer, ff=monomer_force_field, nconf=1000, dft_nconf=4, mpi=1, gpu=0, work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH) # takes ~1 minute
    # mol, energy = qm.conformation_search(monomer, work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH)
    # qm.assign_charges(mol, charge='RESP', work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH)
    ########################

    terminal = utils.mol_from_smiles('*C')
    # polymerization_degree = poly.calc_n_from_num_atoms(monomer, natom=600, terminal1=terminal)
    polymerization_degree = poly.calc_n_from_num_atoms(monomer, natom=600, terminal1=terminal)

    homopoly = poly.polymerize_rw(monomer, polymerization_degree)
    homopoly = poly.terminate_rw(homopoly, terminal)

    # embed_params = AllChem.ETKDGv3()
    # conformation_id = rdDistGeom.EmbedMolecule(homopoly, embed_params)

    # ASSIGN FORCE FIELD.
    force_field = GAFF2()
    ff_success = force_field.ff_assign(homopoly, charge = 'gasteiger')

    print('Polymer FF assignment successful:', ff_success)

    # # CALCULATE ELECTRONIC PROPERTIES.
    # RADON_DIR_PATH = 'temp'
    # qm.assign_charges(homopoly, charge='RESP', work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH)

    print(compute_all_3d_descriptors(homopoly))