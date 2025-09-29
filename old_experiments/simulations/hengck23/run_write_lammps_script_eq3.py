def write_input(
	filename = 'eq3.in',
):
	step = {#default
		"npt": 5_000_000   #sampling
	}
	step = { #fast
		"npt": 200_000 #sampling
	}

	# use also lmp -in in.eq -log log_$(timestamp).lammps

	SETUP = f'''
	log eq3.log append
	
	# MD task: sampling values for of polymer 
	#log eq3.log append
 
	#setup particle ******************************************** 

	units real
	atom_style full
	boundary p p p
	
	# force field type ---
	pair_style lj/charmm/coul/long 8.0 12.0
	kspace_style pppm 1e-6
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
	read_data eq3.data
	'''

	SIMULATE = f'''
	
	# simulate ************************************************
	
	# over-written by below
	#thermo_style custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz
	#thermo_modify flush yes
	#thermo 1000
	 
	# rg definition
	compute cmol1 all chunk/atom molecule nchunk once limit 0 ids once compress no
	compute gyr1 all gyration/chunk cmol1
	fix rg1 all ave/time 1 1000 1000 c_gyr1 file eq3.rg.profile mode vector
	
	# msd definition #Mean-Squared Displacement
	compute msd1 all msd com yes average no
	fix msd1 all ave/time 1 1000 1000 c_msd1[4] mode scalar
	variable msd equal f_msd1
	
	# what to print
	thermo_style custom step time temp press density enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz pxx pyy pzz pxy pxz pyz v_msd
	# thermo_modify flush yes
	thermo 1000
	
	#saving for crash
	dump dump0 all custom 1000 eq3.dump id type mol x y z ix iy iz vx vy vz
	dump xtc0  all xtc 1000 eq3.xtc
	dump_modify xtc0 unwrap yes
	restart 10000 eq3.rst1 eq3.rst2
	
	# actual simulation
	timestep 1.000000
	fix shake1 all shake 1e-4 1000 0 m 1.0
	fix md1 all npt temp 300.000000 300.000000 100.000000 iso 1.000000 1.000000 1000.000000 nreset 1000
	run {step["npt"]} #5000000
	
	
	unfix md1
	uncompute cmol1
	unfix rg1
	uncompute gyr1
	unfix msd1
	uncompute msd1
	unfix shake1
	
	write_dump all custom eq3_last.dump id x y z xu yu zu vx vy vz fx fy fz modify sort id
	write_data eq3_last.data
	quit
	'''

	with open(filename, 'w') as f:
		f.write(SETUP+SIMULATE)
	print('generated:',filename)