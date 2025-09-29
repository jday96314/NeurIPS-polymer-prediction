def write_input(
	filename = 'eq2.in',
):
	step = [ #default
		{  "nvt1":50_000,    "nvt2":50_000,  "npt": 50_000, }, # warmup
		#add 2 more rounds ...
		{  "nvt1": 5_000,    "nvt2":10_000,  "npt":800_000, }, # final
	]
	step = [ #fast
		{  "nvt1":50_000,    "nvt2":50_000,  "npt": 50_000, }, # warmup
		{  "nvt1": 5_000,    "nvt2":10_000,  "npt":200_000, }, # final
	]

	'''
	        if step_list is None:
            step_list = [
                [50000, 50000,  50000],
                [50000, 100000, 50000],
                [50000, 100000, 50000],
                [50000, 100000, 5000],
                [5000,  10000,  5000],
                [5000,  10000,  5000],
                [5000,  10000,  800000]
            ]

        if press_ratio is None:
            press_ratio = [0.02, 0.60, 1.00, 0.50, 0.10, 0.01]
            
        max_press=50000
        press_list = np.append(np.array(press_ratio) * max_press, press)
	'''


	# use also lmp -in in.eq -log log_$(timestamp).lammps
	SETUP = f'''
	log eq2.log append
	
	# MD task: thermal and volumetric equilibration of polymer (NVT → NPT)
	#log eq2.log append
	 
	#setup particle ********************************************
	
	units real
	atom_style full
	boundary p p p
	
	# force field type ---
	pair_style lj/charmm/coul/long 8.0 12.0
	kspace_style pppm 1e-6
	bond_style harmonic
	angle_style harmonic
	dihedral_style fourier
	improper_style cvff
	special_bonds amber
	pair_modify mix arithmetic
	
	neighbor 2.0 bin  
	neigh_modify delay 0 every 1 check yes
	#neigh_modify delay 0 every 1 check yes binsize 16.0
	
	#force field values ---
	dielectric 1.000000
	read_data eq2.data
	'''

	SIMULATE = f'''
	
	# simulate ************************************************
	
	thermo_style custom step time temp press density enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz pxx pyy pzz pxy pxz pyz
	thermo_modify flush yes
	thermo 1000
	
	velocity all create 600.0 398439 mom yes rot yes dist gaussian
	
	timestep 1.000000
	neigh_modify delay 0 every 1 check yes  #yes check for nvt
	fix shake1 all shake 1e-4 1000 0 m 1.0
	fix md1 all nvt temp 600.0 600.0 100.0
	run {step[0]["nvt1"]} # 50000 # = 50 ps at 600 K
	
	unfix md1
	unfix shake1
	
	timestep 1.000000
	fix shake2 all shake 1e-4 1000 0 m 1.0
	fix md2 all nvt temp 300.0 300.0 100.0
	run {step[0]["nvt2"]} # 50000 # = 50 ps at 300 K
	
	unfix md2
	unfix shake2
	
	timestep 1.000000
	neigh_modify delay 0 every 1 check no  #no check for npt (no box change)
	fix shake3 all shake 1e-4 1000 0 m 1.0
	fix md3 all npt temp 300.0 300.0 100.0 iso 1000.0 1000.0 1000.0 nreset 1000
	run {step[0]["npt"]} # 50000  # = 50 ps pressure equilibration
	
	unfix md3
	unfix shake3
	
	
	#############################################################################
	#repeat more here if required
	
	#############################################################################
	
	##repeat
	timestep 1.000000
	neigh_modify delay 0 every 1 check yes  #yes check for nvt
	fix shake4 all shake 1e-4 1000 0 m 1.0
	fix md4 all nvt temp 600.0 600.0 100.0
	run {step[-1]["nvt1"]} # 5000  # = 5 ps at 600 K
	
	unfix md4
	unfix shake4
	
	timestep 1.000000
	fix shake5 all shake 1e-4 1000 0 m 1.0
	fix md5 all nvt temp 300.0 300.0 100.0
	run {step[-1]["nvt2"]} # 10000  # = 10 ps at 300 K
	
	unfix md5
	unfix shake5
	
	# [Final NPT — Production MD or target property sampling at 300 K and 500 atm]
	timestep 1.000000
	neigh_modify delay 0 every 1 check no  #no check for npt
	fix shake6 all shake 1e-4 1000 0 m 1.0
	fix md6 all npt temp 300.0 300.0 100.0 iso 500.0 500.0 1000.0 nreset 1000
	
	# Save restart files every 100,000 steps
	restart 100000 eq3_restart.%%10d 
	
	run {step[-1]["npt"]} # 800000  # = 800 ps (final production)
	
	unfix md6
	unfix shake6
	
	# save ----------------------------------
	#trajectory snapshot
	write_dump all custom eq2_last.dump id x y z xu yu zu vx vy vz fx fy fz modify sort id
	#lammps checkpoint: restart-ready for next script
	write_data eq3.data
	quit
	'''

	with open(filename, 'w') as f:
		f.write(SETUP+SIMULATE)
	print('generated:',filename)