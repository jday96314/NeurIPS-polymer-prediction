def write_input(
	filename = 'eq1.in',
):
	step = { #default
		"nvt1":20_000,    #300k
		"nvt2":1_000_000, #300k to 700k
		"nvt3":1_000_000, #700k (box deform)
	}

	step = { #fast
		"nvt1":20_000,  #300k
		"nvt2":500_000, #300k to 700k
		"nvt3":500_000, #700k (box deform)
	}


	# use also lmp -in in.eq -log log_$(timestamp).lammps
	SETUP = f'''
	log eq1.log append
	
	# MD task:  Initial equilibration of the packed amorphous polymer structure (NVT)
	#log eq1.log append
	
	#setup particle ********************************************
	units real
	atom_style full
	boundary p p p
	
	# force field type ---
	pair_style lj/cut 3.0  #low interact radius, less to compute
	kspace_style none 
	dielectric 1.000000
	bond_style harmonic
	angle_style harmonic
	dihedral_style fourier
	improper_style cvff
	special_bonds amber
	pair_modify mix arithmetic
	
	neighbor 2.0 bin
	neigh_modify delay 0 every 1 check yes
	
 
	
	#force field values --- 
	read_data eq1.data
	'''

	SIMULATE = f'''
	
	# minimize first ************************************************
	 
	thermo_style custom step time temp press density enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz pxx pyy pzz pxy pxz pyz
	thermo_modify flush yes
	thermo 1000
		
	comm_modify cutoff 8.000000 # extend ghost atoms beyond the cutoff
	
	dump dump0 all custom 1000 eq1.dump id type mol x y z ix iy iz vx vy vz
	dump xtc0 all xtc 1000 eq1.xtc
	dump_modify xtc0 unwrap yes
	restart 10000 eq1.rst1 eq1.rst2
	
	#Syntax: minimize etol ftol maxiter maxeval
	min_style cg
	minimize 0.000100 0.000001 10000 100000
	
	undump dump0
	undump xtc0

	# simulate ************************************************
	reset_timestep 0
	
	dump dump0 all custom 1000 eq1.dump id type mol x y z ix iy iz vx vy vz
	dump xtc0 all xtc 1000 eq1.xtc
	dump_modify xtc0 unwrap yes
	
	#No velocity assigned after minimize: system starts at 0 K
	
	# nvt
	timestep 0.100000
	fix md2 all nvt temp 300.000000 300.000000 100.000000
	run {step["nvt1"]} #20000
	
	unfix md2
	
	# nvt
	timestep 1.000000
	fix shake3 all shake 1e-4 1000 0 m 1.0
	fix md3 all nvt temp 300.000000 700.000000 100.000000
	run {step["nvt2"]} #1000000
	
	unfix md3
	unfix shake3
	
	# Deform box to target size after heating to 700 K
	# nvt
	timestep 1.000000
	fix shake4 all shake 1e-4 1000 0 m 1.0 
	fix DEF4 all deform 1 x final -22.950210 22.950210 y final -22.950210 22.950210 z final -22.950210 22.950210 remap v
	fix md4 all nvt temp 700.000000 700.000000 100.000000
	run {step["nvt3"]} #1000000
	
	unfix md4
	unfix shake4
	unfix DEF4
	
	# save ----------------------------------
	write_dump all custom eq1_last.dump id x y z xu yu zu vx vy vz fx fy fz modify sort id
	write_data eq2.data
	quit
	'''

	with open(filename, 'w') as f:
		f.write(SETUP+SIMULATE)
	print('generated:',filename)