import os
import radonpy
import rdkit

from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim import qm
from radonpy.core import poly
from radonpy.sim.lammps import MolFromLAMMPSdata, MolToLAMMPSdataBlock, MolToLAMMPSdata
from radonpy.core import utils

import joblib
import json
import datetime
from rdkit import Chem, RDLogger
from rdkit import RDLogger
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

print('radonpy', radonpy.__version__)
print('rdkit',rdkit.__version__)

def json_dump(data,filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def run_write_lammps_data_v1(
    smiles,
    ter_smiles,
    work_dir,

    #memory and num of threads
    omp_psi4 = 16, mpi = 4, omp =4,
    mem = 32_000, # MB
    atoms_per_chain=1000,
    chain_count=10,
    density=0.05
):
    os.makedirs(f'{work_dir}', exist_ok=True)
    dt_start = datetime.datetime.now()
    print(f'** Start polymerisation for {smiles}.....', str(dt_start))

    # 0. Conformation search ---
    qm_result = dict()
    qm_result['smiles'] = smiles
    qm_result['ter_smiles'] = ter_smiles

    ff = GAFF2_mod()
    mol = utils.mol_from_smiles(smiles)
    mol, energy = qm.conformation_search(
        mol,
        ff=ff,
        work_dir=f'{work_dir}',
        psi4_omp=omp_psi4,
        mpi=mpi,
        omp=omp,
        memory=mem,
        log_name='qm-smiles',
    )
    num_conformer = mol.GetNumConformers()

    print(smiles)
    print('num_conformer', num_conformer)
    print('energy', energy)
    qm_result['num_conformer'] = num_conformer
    qm_result['energy'] = energy.tolist()
    joblib.dump(mol, f'{work_dir}/mol.pkl')

    # 1. Electronic propety calculation ---
    qm.assign_charges(mol, charge='RESP',
            opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')
    # qm_data = qm.sp_prop(mol,
    #         opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')
    # polar_data = qm.polarizability(mol,
    #         opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')

    # print('qm_data', qm_data)
    # print('polar_data', polar_data)
    # qm_result['qm_data'] = qm_data
    # qm_result['polar_data'] = polar_data

    # RESP charge calculation of a termination unit  (Restrained Electrostatic Potential charges)
    ter = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(ter, charge='RESP',
            opt=True, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-ter1')

    # 2. Generate polymer chain ---
    dp = poly.calc_n_from_num_atoms(mol, natom=atoms_per_chain, terminal1=ter)  # dp =110 # degree of polymerization
    print('dp', dp)
    qm_result['dp'] = dp

    RDLogger.DisableLog('rdApp.warning')
    homopoly = poly.polymerize_rw(mol, dp, tacticity='atactic', retry=10, retry_step=100)  # retry=10, retry_step=100
    homopoly = poly.terminate_rw(homopoly, ter)
    RDLogger.EnableLog('rdApp.warning')

    joblib.dump(homopoly, f'{work_dir}/homopoly.pkl')
    utils.MolToJSON(homopoly, f'{work_dir}/homopoly.json')
    json_dump(qm_result,f'{work_dir}/qm_result.json')

    result = ff.ff_assign(homopoly)
    if not result:
        print('ERROR: Can not assign force field parameters for homopoly???')
        exit(0)
        return 1

    ac = poly.amorphous_cell(homopoly, n=chain_count, density=density)
    joblib.dump(ac, f'{work_dir}/ac.pkl')
    MolToLAMMPSdata(ac, f'{work_dir}/eq1.data')

    print(f'** Complete polymerisation for {smiles}. Elapsed time = {str(datetime.datetime.now() - dt_start)}')
    print('')
    return 0

def run_write_lammps_data_v2(
    smiles,
    ter_smiles,
    work_dir,

    #memory and num of threads
    omp_psi4 = 16, mpi = 4, omp =4,
    mem = 32_000, # MB
    atoms_per_chain=1000,
    chain_count=10,
    density=0.05
):
    os.makedirs(f'{work_dir}', exist_ok=True)
    dt_start = datetime.datetime.now()
    print(f'** Start polymerisation for {smiles}.....', str(dt_start))

    # 0. Conformation search ---
    qm_result = dict()
    qm_result['smiles'] = smiles
    qm_result['ter_smiles'] = ter_smiles

    ff = GAFF2_mod()
    mol = utils.mol_from_smiles(smiles)
    mol, energy = qm.conformation_search(
        mol,
        nconf=30,
        dft_nconf=4,
        # method="HF/3-21G",
        # psi4_options={"d_convergence": 1e-6, "maxiter": 40},
        # opt_method='hf',
        opt_method='b97-3c',
        # opt_method='mp2/aug-cc-pvdz',
        # psi4_options={'dynamic_level': 3},
        # psi4_options={'g_convergence': 'gau'},
        # opt_basis='3-21G',
        # geom_algorithm='TRUST',
        ff=ff,
        work_dir=f'{work_dir}',
        psi4_omp=omp_psi4,
        mpi=mpi,
        omp=omp,
        memory=mem,
        log_name='qm-smiles',
    )
    num_conformer = mol.GetNumConformers()

    print(smiles)
    print('num_conformer', num_conformer)
    print('energy', energy)
    qm_result['num_conformer'] = num_conformer
    qm_result['energy'] = energy.tolist()
    joblib.dump(mol, f'{work_dir}/mol.pkl')

    # 1. Electronic propety calculation ---
    print('Assigning charges...')
    qm.assign_charges(mol, charge='RESP',
            opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')
    # qm_data = qm.sp_prop(mol,
    #         opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')
    # polar_data = qm.polarizability(mol,
    #         opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')

    # print('qm_data', qm_data)
    # print('polar_data', polar_data)
    # qm_result['qm_data'] = qm_data
    # qm_result['polar_data'] = polar_data

    # RESP charge calculation of a termination unit  (Restrained Electrostatic Potential charges)
    ter = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(ter, charge='RESP',
            opt=True, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-ter1')

    # 2. Generate polymer chain ---
    dp = poly.calc_n_from_num_atoms(mol, natom=atoms_per_chain, terminal1=ter)
    print('dp', dp)
    qm_result['dp'] = dp

    RDLogger.DisableLog('rdApp.warning')
    homopoly = poly.polymerize_rw(mol, dp, tacticity='atactic', retry=10, retry_step=100)
    homopoly = poly.terminate_rw(homopoly, ter)
    RDLogger.EnableLog('rdApp.warning')

    joblib.dump(homopoly, f'{work_dir}/homopoly.pkl')
    utils.MolToJSON(homopoly, f'{work_dir}/homopoly.json')
    json_dump(qm_result,f'{work_dir}/qm_result.json')

    result = ff.ff_assign(homopoly)
    if not result:
        print('ERROR: Can not assign force field parameters for homopoly???')
        exit(0)
        return 1

    ac = poly.amorphous_cell(homopoly, n=chain_count, density=density)
    joblib.dump(ac, f'{work_dir}/ac.pkl')
    MolToLAMMPSdata(ac, f'{work_dir}/eq1.data')

    print(f'** Complete polymerisation for {smiles}. Elapsed time = {str(datetime.datetime.now() - dt_start)}')
    print('')
    return 0

def run_write_lammps_data(
    smiles,
    ter_smiles,
    work_dir,

    #memory and num of threads
    omp_psi4 = 16, mpi = 4, omp =4,
    mem = 32_000, # MB
    atoms_per_chain=1000,
    chain_count=10,
    density=0.05,
    opt_method='hf', # Fast unstable config: (hf, RFO). Slower more stable config: (b97-3c, NR)
    geom_algorithm='RFO',
):
    os.makedirs(f'{work_dir}', exist_ok=True)
    dt_start = datetime.datetime.now()
    print(f'** Start polymerisation for {smiles}.....', str(dt_start))

    # 0. Conformation search ---
    qm_result = dict()
    qm_result['smiles'] = smiles
    qm_result['ter_smiles'] = ter_smiles

    ff = GAFF2_mod()
    mol = utils.mol_from_smiles(smiles)
    
    # AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    # try:
    #     AllChem.MMFFOptimizeMolecule(mol)
    #     print("MMFF94 optimization successful. 👍")
    # except Exception as e:
    #     print(f"Warning: MMFF94 pre-optimization failed: {e}. Proceeding with unoptimized structure.")

    conformer_ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=30
    )
    for conformer_id in conformer_ids:
        AllChem.MMFFOptimizeMolecule(
            mol,
            confId=conformer_id,
            maxIters=200
        )

    mol, energy = qm.conformation_search(
        mol,
        nconf=30,
        dft_nconf=4,
        opt_method=opt_method,
        geom_algorithm=geom_algorithm,
        ff=ff,
        work_dir=f'{work_dir}',
        psi4_omp=omp_psi4,
        mpi=mpi,
        omp=omp,
        memory=mem,
        log_name='qm-smiles',
    )
    num_conformer = mol.GetNumConformers()

    print(smiles)
    print('num_conformer', num_conformer)
    print('energy', energy)
    qm_result['num_conformer'] = num_conformer
    qm_result['energy'] = energy.tolist()
    joblib.dump(mol, f'{work_dir}/mol.pkl')

    # 1. Electronic propety calculation ---
    print('Assigning charges...')
    qm.assign_charges(mol, charge='RESP',
            opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')
    # qm_data = qm.sp_prop(mol,
    #         opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')
    # polar_data = qm.polarizability(mol,
    #         opt=False, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-smiles')

    # print('qm_data', qm_data)
    # print('polar_data', polar_data)
    # qm_result['qm_data'] = qm_data
    # qm_result['polar_data'] = polar_data

    # RESP charge calculation of a termination unit  (Restrained Electrostatic Potential charges)
    ter = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(ter, charge='RESP',
            opt=True, work_dir=f'{work_dir}', omp=omp_psi4, memory=mem, log_name='qm-ter1')

    # 2. Generate polymer chain ---
    dp = poly.calc_n_from_num_atoms(mol, natom=atoms_per_chain, terminal1=ter)
    print('dp', dp)
    qm_result['dp'] = dp

    RDLogger.DisableLog('rdApp.warning')
    homopoly = poly.polymerize_rw(mol, dp, tacticity='atactic', retry=10, retry_step=100)
    homopoly = poly.terminate_rw(homopoly, ter)
    RDLogger.EnableLog('rdApp.warning')

    joblib.dump(homopoly, f'{work_dir}/homopoly.pkl')
    utils.MolToJSON(homopoly, f'{work_dir}/homopoly.json')
    json_dump(qm_result,f'{work_dir}/qm_result.json')

    result = ff.ff_assign(homopoly)
    if not result:
        print('ERROR: Can not assign force field parameters for homopoly???')
        exit(0)
        return 1

    ac = poly.amorphous_cell(homopoly, n=chain_count, density=density)
    joblib.dump(ac, f'{work_dir}/ac.pkl')
    MolToLAMMPSdata(ac, f'{work_dir}/eq1.data')

    print(f'** Complete polymerisation for {smiles}. Elapsed time = {str(datetime.datetime.now() - dt_start)}')
    print('')
    return 0