from radonpy.core import utils, poly
from radonpy.sim import qm
from radonpy.ff.gaff2_mod import GAFF2_mod
from rdkit import Chem

def print_mol_details(mol):
    atoms = mol.GetAtoms()
    for atom in list(atoms)[:5]:
        print(atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol(), atom.GetMass())

    bonds = mol.GetBonds()
    for bond in list(bonds)[:5]:
        print(bond.GetIdx(), bond.GetBondTypeAsDouble(), bond.IsInRing(), bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())


if __name__ == '__main__':
    # CONFORMATION SEARCH OF REPEATING UNIT.
    RADON_DIR_PATH = 'temp'

    smiles = '*CC(*)c1ccccc1'
    mol = utils.mol_from_smiles(smiles)

    print_mol_details(mol)
    print(Chem.MolToMolBlock(mol))

    monomer_force_field = GAFF2_mod()
    mol, energy = qm.conformation_search(mol, ff=monomer_force_field, nconf=1000, dft_nconf=4, mpi=1, gpu=0, work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH) # takes ~1 minute

    print(Chem.MolToMolBlock(mol))

    # CALCULATION OF ELECTRONIC PROPERTIES (FOR REPEATING UNIT).
    qm.assign_charges(mol, charge='RESP', work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH)
    # qm_data = qm.sp_prop(mol, opt=False, work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH) # {'qm_total_energy': -610096.5625925089, 'qm_homo': -9.379850163190179, 'qm_lumo': 1.9606289674607098, 'qm_dipole_x': 7.951216111113496e-06, 'qm_dipole_y': 1.237493671228167e-06, 'qm_dipole_z': -1.8139875425914563e-07}
    # polar_data = qm.polarizability(mol, opt=False, work_dir=RADON_DIR_PATH, tmp_dir=RADON_DIR_PATH) # takes ~1.5 minutes, {'qm_polarizability': 9.960950971971828, 'qm_polarizability_xx': 11.740537823453923, 'qm_polarizability_yy': 11.74434423969134, 'qm_polarizability_zz': 6.39797085277022, 'qm_polarizability_xy': -0.0017087063747881388, 'qm_polarizability_xz': 0.17540689819705302, 'qm_polarizability_yz': 0.06502540094033438}

    # print(qm_data)
    # print(polar_data)
    print(Chem.MolToMolBlock(mol))

    # POLYMER CHAIN GENERATION.
    terminal = utils.mol_from_smiles('*C')
    polymerization_degree = poly.calc_n_from_num_atoms(mol, natom=600, terminal1=terminal)

    homopoly = poly.polymerize_rw(mol, polymerization_degree)
    homopoly = poly.terminate_rw(homopoly, terminal)

    print_mol_details(mol)
    print(Chem.MolToMolBlock(homopoly))

    # POLYMER FORCE FIELD ASSIGNMENT.
    homopoly_force_field = GAFF2_mod()
    ff_success = homopoly_force_field.ff_assign(homopoly)

    print('Polymer FF assignment successful:', ff_success)

    print(homopoly_force_field)
