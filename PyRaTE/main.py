#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#  NAME:                                                                  #
#                                                                         #
#     main                                                                #
#                                                                         #
#  DESCRIPTION:                                                           #
#                                                                         #
#     Main function of PyRaTE. Calls all other functions and takes care   #
#     of parallelization.                                                 #
#                                                                         #
#  AUTHOR:                                                                #
#                                                                         #
#  Aris E. Tritsis                                                        #
#  (aris.tritsis@epfl.ch)                                                 #
                                                                          #
from __future__ import division                                           #
from scipy.constants import m_p, parsec, c, h, k                          #
import numpy as np                                                        #
import os, sys, time                                                      #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                       Auxiliary functions                               #
import setup                                                              #
import populations                                                        #
import distance, axisRearange, render3D                                   #
import sourcef, dust, radTran                                             #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
import warnings                                                           #
from mpi4py import MPI                                                    #
                                                                          #
m_p, parsec, siny, c, h, Kb = m_p*1e+3, parsec*1e+2, 31556926., c*1e+2, h*1.e+7, k*1.e+7
cwd = os.getcwd()                                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                       $$$ PARALELIZATION $$$                            #
comm = MPI.COMM_WORLD                                                     #
iproc=comm.Get_rank()                                                     #
nproc=comm.Get_size()                                                     #
                                                                          #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#- - - - - small function to monitor progress of the code- - - - - - - - -#
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
def update_progress(progress):                                            #
	barLength = 100                                                   #
	status = ""                                                       #
	if isinstance(progress, int):                                     #
		progress = float(progress)                                #
	if not isinstance(progress, float):                               #
		progress = 0                                              #
		status = "error: progress var must be float\r\n"          #
	if progress < 0:                                                  #
		progress = 0                                              #
		status = "Halt...\r\n"                                    #
	if progress >= 1:                                                 #
		progress = 1                                              #
		status = "Done...\r\n"                                    #
	block = int(round(barLength*progress))                            #
	text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,1), status)
	sys.stdout.write(text)                                            #
	sys.stdout.flush()                                                #
#                 Make python shut up/disable useless warnings            #
warnings.filterwarnings("ignore")                                         #
warnings.filterwarnings("ignore",category=DeprecationWarning)             #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                                                                         #
#    Some utility functions useful for MPI parallel programming           #
#                                                                         #
#                                                                         #
def pprint(str="", end="\n", comm=MPI.COMM_WORLD):                        #
	"""Print for MPI parallel programs: Only rank 0 prints *str*."""  #
	if iproc == 0:                                                    #
		print (str+end)                                           #
                                                                          #
pprint('\n')                                                              #
pprint('\n')                                                              #
pprint(" "*35+"Running on %d core(s)" % comm.size)                        #
pprint('\n')                                                              #
pprint('\n')                                                              #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#           $$$  Read in parameters for specific run  $$$                 #
if iproc==0:                                                              #
	                                                                  #
	fi0, th0, up, low, npoints, Vres, line, nonLTE, GK, BgCMB, restart, ndims, dens, mol, T, vx, x, vy, y, vz, z, Cul, Clu, EinA, EinBul, EinBlu, Ener, freqs, gul, tempers, Bx, By, Bz, CulGK, CluGK, Vrange, frRange, molmass, numlevels, jmjpmp, onaxiscases = setup.setup()
	                                                                  #
	fi=np.radians(fi0)                                                #
	                                                                  #
	th=np.radians(th0)                                                #
	                                                                  #
else:                                                                     #
	fi0, th0, up, low, npoints, Vres, line, nonLTE, GK, BgCMB, restart = None, None, None, None, None, None, None, None, None, None, None
	                                                                  #
	ndims, dens, mol, T, vx, x, vy, y, vz, z, molmass, onaxiscases = None, None, None, None, None, None, None, None, None, None, None, None
	                                                                  #
	Cul, Clu, EinA, EinBul, EinBlu, Ener, freqs, gul, tempers = None, None, None, None, None, None, None, None, None
	                                                                  #
	Vrange, frRange = None, None                                      #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                $$$  GK specific  $$$                            #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	Bx, By, Bz, CulGK, CluGK, numlevels, jmjpmp = None, None, None, None, None, None, None
                                                                          #
fi0, th0, up, low = comm.bcast(fi0, root=0), comm.bcast(th0, root=0), comm.bcast(up, root=0), comm.bcast(low, root=0)
                                                                          #
npoints, Vres, line, nonLTE, GK, BgCMB, Vrange, frRange, restart = comm.bcast(npoints, root=0), comm.bcast(Vres, root=0), comm.bcast(line, root=0), comm.bcast(nonLTE, root=0), comm.bcast(GK, root=0), comm.bcast(BgCMB, root=0), comm.bcast(Vrange, root=0), comm.bcast(frRange, root=0), comm.bcast(restart, root=0)
                                                                          #
ndims, dens, mol, T, vx, x, vy, y, vz, z, molmass = comm.bcast(ndims, root=0), comm.bcast(dens, root=0), comm.bcast(mol, root=0), comm.bcast(T, root=0), comm.bcast(vx, root=0), comm.bcast(x, root=0), comm.bcast(vy, root=0), comm.bcast(y, root=0), comm.bcast(vz, root=0), comm.bcast(z, root=0), comm.bcast(molmass, root=0)
                                                                          #
Cul, Clu, EinA, EinBul, EinBlu, Ener, freqs, gul, tempers = comm.bcast(Cul, root=0), comm.bcast(Clu, root=0), comm.bcast(EinA, root=0), comm.bcast(EinBul, root=0), comm.bcast(EinBlu, root=0), comm.bcast(Ener, root=0), comm.bcast(freqs, root=0), comm.bcast(gul, root=0), comm.bcast(tempers, root=0)
                                                                          #
Bx, By, Bz, CulGK, CluGK, numlevels, jmjpmp = comm.bcast(Bx, root=0), comm.bcast(By, root=0), comm.bcast(Bz, root=0), comm.bcast(CulGK, root=0), comm.bcast(CluGK, root=0), comm.bcast(numlevels, root=0), comm.bcast(jmjpmp, root=0)
                                                                          #
onaxiscases = comm.bcast(onaxiscases, root=0)                             #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                                                                         #
# $$$ Call "populations" to get the population densities of E-levels $$$  #
#                                                                         #
#      ____                    __      __  _                              #
#     / __ \____  ____  __  __/ /___ _/ /_(_)___  ____  _____             #
#    / /_/ / __ \/ __ \/ / / / / __ `/ __/ / __ \/ __ \/ ___/             #
#   / ____/ /_/ / /_/ / /_/ / / /_/ / /_/ / /_/ / / / (__  )              #
#  /_/    \____/ .___/\__,_/_/\__,_/\__/_/\____/_/ /_/____/               #
#             /_/                                                         #
#             ____        __              __                              #
#            / __ \__  __/ /_____  __  __/ /______                        #
#           / / / / / / / __/ __ \/ / / / __/ ___/                        #
#          / /_/ / /_/ / /_/ /_/ / /_/ / /_(__  )                         #
#          \____/\__,_/\__/ .___/\__,_/\__/____/                          #
#                        /_/                                              #
#                                                                         #
#     1. TLINE (BETAS) saved but not used anywhere further down           #
#     2. PopRat (different meaning for GK and Fiducial), nlow             #
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
pprint('\n')                                                              #
                                                                          #
pprint(" "*20+"Calculating population densities. Please wait :)")         #
                                                                          #
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]                              #
                                                                          #
Popstart = time.time()                                                    #
                                                                          #
if restart == False:                                                      #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#        $$$  Cut arrays to save computational time  $$$          #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	                                                                  #
	if ndims == 2:                                                    #
		                                                          #
		dens2, mol2, T2 = dens[len(dens)//2:, :, :], mol[len(mol)//2:, :, :], T[len(T)//2:, :, :]
		                                                          #
		vx2, vy2, vz2, y2 = vx[len(vx)//2:, :, :], vy[len(vy)//2:, :, :], vz[len(vz)//2:, :, :], y[len(y)//2:]
		                                                          #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		#                                                         #
		#       Here we brake the symmetry and need vz too...     #
		#    Luckily we have everything ready to get vz in 3D     #
		#                                                         #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		if GK:                                                    #
			                                                  #
			Bx2, By2, Bz2 = Bx[len(Bx)//2:, :, :], By[len(By)//2:, :, :], Bz[len(Bz)//2:, :, :]
			                                                  #
			vz2 = render3D.render3D(vx2, x, y, typeQ = "vectorR")
			                                                  #
			vz2 = np.hstack( (np.flip(vz2, axis = 1), vz2 ))  #
			                                                  #
			vz2 = np.rot90(vz2, axes = (1, 2))                #
			                                                  #
			vz2 = vz2[:, len(vz2[0])//2:len(vz2[0]), :]       #
			                                                  #
			dens2, mol2 = render3D.render3D(dens2, x, y, typeQ = "scalar"), render3D.render3D(mol2, x, y, typeQ = "scalar")
			                                                  #
			dens2, mol2, vz2 = dens2[:, :, len(dens2[0, 0])//2:], mol2[:, :, len(mol2[0, 0])//2:], vz2[:, :, len(vz2[0, 0])//2:]
			                                                  #
		else:                                                     #
			                                                  #
			Bx2, By2, Bz2 = Bx, By, Bz                        #
			                                                  #
	else:                                                             #
		                                                          #
		dens2, mol2, T2 = dens, mol, T                            #
		                                                          #
		vx2, vy2, vz2, y2 = vx, vy, vz, y                         #
		                                                          #
		Bx2, By2, Bz2 = Bx, By, Bz                                #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#  $$$  Take care of 1D/2D arrays so that loops below work  $$$   #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	if ndims > 1:                                                     #
		                                                          #
		chopped_ys=np.empty(dens2.shape[0]//nproc, dtype=np.float64)
		                                                          #
		comm.Scatter([y2, MPI.DOUBLE], [chopped_ys, MPI.DOUBLE])  #
	                                                                  #
	else:                                                             #
		chopped_ys = y.copy()                                     #
	                                                                  #
	LevPops, TLINE = populations.populations(nonLTE, BgCMB, ndims, dens2, mol2, T2, vx2, dx, vy2, dy, vz2, dz, y2, chopped_ys, Cul, Clu, EinA, EinBul, EinBlu, Ener, freqs, gul, tempers, GK, Bx2, By2, Bz2, numlevels, CulGK, CluGK, jmjpmp)     #
	                                                                  #
	LevPops_gathered, TLINE_gathered = comm.gather(LevPops, root=0), comm.gather(TLINE, root=0)
                                                                          #
if iproc==0:                                                              #
	                                                                  #
	if restart:                                                       #
		                                                          #
		LevPops = np.load('{}/output_files/LevPops.npy'.format(cwd))
		                                                          #
		if GK:                                                    #
			                                                  #
			#    $$$ PopRat different meaning here! $$$       #
			uniquelevels = np.array([list(x) for x in set(tuple(x) for x in np.concatenate((jmjpmp[:, np.newaxis:2], jmjpmp[:, 2:np.newaxis])))])
			                                                  #
			uniquelevels = uniquelevels[np.argsort(uniquelevels[:, 0] *0.8 + uniquelevels[:, 1]*0.2)]
			                                                  #
			ilow, iup = np.min(np.where(uniquelevels[:, 0] == low)[0]), np.max(np.where(uniquelevels[:, 0] == up)[0])+1
			                                                  #
			PopRat = LevPops[:, :, :, ilow:iup]               #
			                                                  #
			ilow, iup = np.min(np.where(jmjpmp[:, 2] == low)[0]), np.max(np.where(jmjpmp[:, 0] == up)[0])+1
			                                                  #
			low = ilow                                        #
			                                                  #
			jmjpmp = jmjpmp[ilow:iup]                         #
			                                                  #
			EinBlu = EinBlu[ilow:iup]                         #
			                                                  #
		else:                                                     #
			PopRat = LevPops[:, :, :, up]/LevPops[:, :, :, low]
			                                                  #
			EinBlu = EinBlu[low]                              #
			                                                  #
		try:                                                      #
			                                                  #
			nlow = LevPops[:, :, :, low]                      #
			                                                  #
		except IndexError:                                        #
			                                                  #
			nlow = LevPops[:, :, :, 0]                        #
			                                                  #
			print("\n")                                       #
			print("                            /\ ")          #
			print("                           /  \ ")         #
			print("                          / || \ " + "  main.py, lines 200-300! len(LevPops) < low")
			print("                         /  ||  \ " + " Make sure 'uniquelevels' are correct!")
			print("                        /   ..   \ " + "Are you running a simulations with GK = True?")
			print("                       /          \ ")     #
			print("                       ------------ ")     #
			print("\n")                                       #
		                                                          #
	else:                                                             #
		                                                          #
		#  $$$ Gather arrays from all CPUs and reload needed $$$  #
		LevPops, TLINE = np.array(LevPops_gathered).astype(float), np.array(TLINE_gathered).astype(float)
		                                                          #
		np.save('{}/output_files/LevPops'.format(cwd), LevPops)   #
		                                                          #
		np.save('{}/output_files/TLINE'.format(cwd), TLINE)       #
		                                                          #
		size2 = LevPops.shape                                     #
		                                                          #
		LevPops = LevPops.reshape(nproc*size2[1], size2[2], size2[3], size2[4])
		                                                          #
		size2 = TLINE.shape                                       #
		                                                          #
		TLINE = TLINE.reshape(nproc*size2[1], size2[2], size2[3], size2[4])
		                                                          #
		if ndims == 2:                                            #
			                                                  #
			tempL, tempLINE = np.flipud(LevPops), np.flipud(TLINE)
			                                                  #
			LevPops, TLINE = np.vstack((tempL, LevPops)), np.vstack((tempLINE, TLINE))
			                                                  #
		if GK:                                                    #
			                                                  #
			#    $$$ PopRat different meaning here! $$$       #
			uniquelevels = np.array([list(x) for x in set(tuple(x) for x in np.concatenate((jmjpmp[:, np.newaxis:2], jmjpmp[:, 2:np.newaxis])))])
			                                                  #
			uniquelevels = uniquelevels[np.argsort(uniquelevels[:, 0] *0.8 + uniquelevels[:, 1]*0.2)]
			                                                  #
			ilow, iup = np.min(np.where(uniquelevels[:, 0] == low)[0]), np.max(np.where(uniquelevels[:, 0] == up)[0])+1
			                                                  #
			PopRat = LevPops[:, :, :, ilow:iup]               #
			                                                  #
			ilow, iup = np.min(np.where(jmjpmp[:, 2] == low)[0]), np.max(np.where(jmjpmp[:, 0] == up)[0])+1
			                                                  #
			low = ilow                                        #
			                                                  #
			jmjpmp = jmjpmp[ilow:iup]                         #
			                                                  #
			EinBlu = EinBlu[ilow:iup]                         #
			                                                  #
		else:                                                     #
			PopRat = LevPops[:, :, :, up]/LevPops[:, :, :, low]
			                                                  #
			EinBlu = EinBlu[low]                              #
			                                                  #
		try:                                                      #
			                                                  #
			nlow = LevPops[:, :, :, low]                      #
			                                                  #
		except IndexError:                                        #
			                                                  #
			nlow = LevPops[:, :, :, 0]                        #
			                                                  #
			print("\n")                                       #
			print("                            /\ ")          #
			print("                           /  \ ")         #
			print("                          / || \ " + "  main.py, lines 200-300! len(LevPops) < low")
			print("                         /  ||  \ " + " Make sure 'uniquelevels' are correct!")
			print("                        /   ..   \ " + "Are you running a simulations with GK = True?")
			print("                       /          \ ")     #
			print("                       ------------ ")     #
			print("\n")                                       #
		                                                          #
		np.save('{}/output_files/LevPops'.format(cwd), LevPops)   #
		                                                          #
		np.save('{}/output_files/TLINE'.format(cwd), TLINE)       #
		                                                          #
else: PopRat, nlow = None, None                                           #
                                                                          #
PopRat, nlow = comm.bcast(PopRat, root=0), comm.bcast(nlow, root=0)       #
                                                                          #
jmjpmp, EinBlu = comm.bcast(jmjpmp, root=0), comm.bcast(EinBlu, root=0)   #
                                                                          #
if GK: low = comm.bcast(low, root=0)                                      #
                                                                          #
try:                                                                      #
	                                                                  #
	grat = gul[up]/gul[low]                                           #
	                                                                  #
except IndexError:                                                        #
	                                                                  #
	grat = np.ones(len(gul))                                          #
                                                                          #
#    This is so that we don't "break" "sourcef" when GK = False           #
gam  = None                                                               #
                                                                          #
Popstop = time.time()                                                     #
                                                                          #
pprint('\n')                                                              #
                                                                          #
pprint(" "*25+"Finished in {} minutes! Starting the integration...".format(np.round((Popstop - Popstart)/60., 1)) )
#                 ______          __                                      #
#                / ____/___  ____/ /                                      #
#               / __/ / __ \/ __  /                                       #
#              / /___/ / / / /_/ /                                        #
#             /_____/_/ /_/\__,_/                                         #
#      ____                    __      __  _                              #
#     / __ \____  ____  __  __/ /___ _/ /_(_)___  ____  _____             #
#    / /_/ / __ \/ __ \/ / / / / __ `/ __/ / __ \/ __ \/ ___/             #
#   / ____/ /_/ / /_/ / /_/ / / /_/ / /_/ / /_/ / / / (__  )              #
#  /_/    \____/ .___/\__,_/_/\__,_/\__/_/\____/_/ /_/____/               #
#             /_/                                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                                                                         #
#         \\                                              //              #
#          \\                                            //               #
#           \\   $$  Special cases: 1. Sinle line  $$   //                #
#           //                      2. Cyl face on      \\                #
#          //                                            \\               #
#         //                                              \\              #
#                                                                         #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
if line == True:                                                          #
	                                                                  #
	if iproc == 0:                                                    #
		                                                          #
		dens, mol, vLOS, T, PopRat, nlow, LOSaxis, Bx, By, Bz, Blos = axisRearange.axisRearange(ndims, line, fi0, th0, dens, mol, vx, vy, vz, T, PopRat, nlow, x, y, z, Bx, By, Bz, GK)
		                                                          #
		PPV, pr, dist = [], 0, LOSaxis[1] - LOSaxis[0]            #
		                                                          #
		if GK: PPVParal = []                                      #
		                                                          #
		for f in range (0, len(frRange)):                         #
			                                                  #
			frq = frRange[f]                                  #
			                                                  #
			pr = pr + 1./len(frRange)                         #
			                                                  #
			if iproc==0: update_progress(pr)                  #
			                                                  #
			I_a, vela, Sline_a, Itotal, LineAbs_a, Sc_a, kext_a= 0., 0., 0., 0., 0., 0., 0.
			                                                  #
			if GK: I_aP, Sline_aP, ItotalP, LineAbs_aP = 0., 0., 0., 0.
			                                                  #
			for i in range (0, len(LOSaxis)):                 #
				                                          #
				densgp, Tgp, PopRatgp, nlowgp, velb = dens[i], T[i], PopRat[i], nlow[i], vLOS[i]
				                                          #
				Sline_b, LineAbs_b=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, False)
				                                          #
				Sc_b, kext_b = dust.dust(frq/c, Tgp, densgp)
				                                          #
				Dthrm = np.sqrt(Kb*Tgp/molmass/c**2) * freqs[low]
				                                          #
				I_b = radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_a, Sc_a, Sc_b, Sline_a, Sline_b, LineAbs_a, LineAbs_b, kext_a, kext_b)
				                                          #
				Itotal = I_b                              #
				                                          #
				if GK == True:                            #
					                                  #
					gam = np.arccos(Blos[i]/np.sqrt(Bx[i]**2 + By[i]**2 + Bz[i]**2))
					                                  #
					Sline_bP, LineAbs_bP=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK)
					                                  #
					I_bP=radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_aP, Sc_a, Sc_b, Sline_aP, Sline_bP, LineAbs_aP, LineAbs_bP, kext_a, kext_b)
					                                  #
					ItotalP = I_bP                    #
					                                  #
					I_aP, Sline_aP, LineAbs_aP = I_bP, Sline_bP, LineAbs_bP
					                                  #
				I_a, Sline_a, LineAbs_a, Sc_a, kext_a = I_b, Sline_b, LineAbs_b, Sc_b, kext_b
				                                          #
			PPV.append(Itotal)                                #
			                                                  #
			if GK: PPVParal.append(ItotalP)                   #
			                                                  #
		np.save("{}/output_files/PPV".format(cwd), np.array(PPV)) #
		                                                          #
		if GK: np.save("{}/output_files/PPVParal".format(cwd), np.array(PPVParal))
	                                                                  #
	sys.exit(0)                                                       #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
if line == False and ndims == 2 and fi0 == 90:                            #
	                                                                  #
	if iproc == 0:                                                    #
		                                                          #
		PPV, pr = [], 0                                           #
		                                                          #
		if GK: PPVParal = []                                      #
		                                                          #
		for i in range (0, len(dens[0])):                         #
			                                                  #
			pr = pr + 1./len(dens[0])                         #
			                                                  #
			if iproc==0: update_progress(pr)                  #
			                                                  #
			ILOS = np.empty(len(frRange))                     #
			                                                  #
			if GK: ILOSParal = np.empty(len(frRange))         #
			                                                  #
			for f in range (0, len(frRange)):                 #
				                                          #
				frq = frRange[f]                          #
				                                          #
				I_a, vela, Sline_a, Itotal, LineAbs_a, Sc_a, kext_a= 0., 0., 0., 0., 0., 0., 0.
				                                          #
				if GK: I_aP, Sline_aP, ItotalP, LineAbs_aP = 0., 0., 0., 0.
				                                          #
				for j in range (0, len(dens)):            #
					                                  #
					densgp, Tgp, PopRatgp, nlowgp, velb=dens[j, i, 0], T[j, i, 0], PopRat[j, i, 0], nlow[j, i, 0], vy[j, i, 0]
					                                  #
					Dthrm = np.sqrt(Kb*Tgp/molmass/c**2) * freqs[low]
					                                  #
					Sline_b, LineAbs_b=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK=False)
					                                  #
					Sc_b, kext_b = dust.dust(frq/c, Tgp, densgp)
					                                  #
					I_b=radTran.radTran(frq, vela, velb, dy, freqs[low], Dthrm, I_a, Sc_a, Sc_b, Sline_a, Sline_b, LineAbs_a, LineAbs_b, kext_a, kext_b)
					                                  #
					Itotal = I_b                      #
					                                  #
					if GK == True:                    #
						#  LOS vector: (0, 1, 0)  #
						# gam = LOSvec * Bvec /   #
						#   (|LOSvec||Bvec|)      #
						gam = np.arccos(By[j, i, 0]/np.sqrt(Bx[j, i, 0]**2 + By[j, i, 0]**2))
						                          #
						Sline_bP, LineAbs_bP=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK)
						                          #
						I_bP=radTran.radTran(frq, vela, velb, dy, freqs[low], Dthrm, I_aP, Sc_a, Sc_b, Sline_aP, Sline_bP, LineAbs_aP, LineAbs_bP, kext_a, kext_b)
						                          #
						ItotalP = I_bP            #
						                          #
						I_aP, Sline_aP, LineAbs_aP = I_bP, Sline_bP, LineAbs_bP
						                          #
					I_a, Sline_a, LineAbs_a, Sc_a, kext_a, vela = I_b, Sline_b, LineAbs_b, Sc_b, kext_b, velb
					                                  #
				ILOS[f]=Itotal                            #
				                                          #
				if GK: ILOSParal[f]=ItotalP               #
			                                                  #
			PPV.append(np.array(ILOS).astype(float))          #
			                                                  #
			if GK: PPVParal.append( np.array(ILOSParal).astype(float) )
		                                                          #
		PPV = np.array(PPV)                                       #
		                                                          #
		np.save('{}/output_files/PPV'.format(cwd), PPV)           #
		                                                          #
		if GK:                                                    #
			PPVParal = np.array(PPVParal)                     #
			                                                  #
			np.save('{}/output_files/PPVParal'.format(cwd), PPVParal)
	sys.exit(0)                                                       #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                                                                         #
#         \\                                                  //          #
#          \\                                                //           #
#           \\  $$ Prepare data for on axis integration $$  //            #
#           //  $$      (and off-axis for 2D case)      $$  \\            #
#          //                                                \\           #
#         //                                                  \\          #
#                                                                         #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
if iproc == 0:                                                            #
	                                                                  #
	if np.any(np.sum(onaxiscases - [ndims, fi0, th0], axis = 1) == 0.) or ndims == 2:
		                                                          #
		dens, mol, vLOS, T, PopRat, nlow, LOSaxis, RowAxis, Blos = axisRearange.axisRearange(ndims, line, fi0, th0, dens, mol, vx, vy, vz, T, PopRat, nlow, x, y, z, Bx, By, Bz, GK)
		                                                          #
		dist = LOSaxis[1] - LOSaxis[0]                            #
		                                                          #
		if ndims == 2:                                            #
			                                                  #
			dens, mol = render3D.cntCart(dens, x, typeQ = "scalar"), render3D.cntCart(mol, x, typeQ = "scalar")
			                                                  #
			T, nlow = render3D.cntCart(T, x, typeQ = "scalar"), render3D.cntCart(nlow, x, typeQ = "scalar")
			                                                  #
			if not GK:                                        #
				                                          #
				PopRat = render3D.cntCart(PopRat, x, typeQ = "scalar")
				                                          #
			else:                                             #
				                                          #
				temp = np.empty( (dens.shape[0], dens.shape[1], dens.shape[2], len(PopRat[0, 0, 0])) )
				                                          #
				for l in range (0, len(PopRat[0, 0, 0])): #
					                                  #
					temp[:, :, :, l] = render3D.cntCart(PopRat[:, :, :, l], x, typeQ = "scalar")
					                                  #
				PopRat = temp.copy()                      #
				                                          #
				del temp                                  #
				                                          #
				By = render3D.cntCart(By, x, typeQ = "vectorZ")
				                                          #
				Bx = render3D.cntCart(Bx, x, typeQ = "vectorR")
				                                          #
				Bz = np.hstack( (np.flip(Bx, axis = 1), Bx ))
				                                          #
				Bz = np.rot90(Bz, axes = (1, 2))          #
				                                          #
				Bz = Bz[:, len(Bz[0])//2:len(Bz[0]), :]   #
				                                          #
				Blos = Bx                                 #
				                                          #
			vLOS = render3D.cntCart(vx, x, typeQ = "vectorR") #
			                                                  #
			vy = render3D.cntCart(vy, x, typeQ = "vectorZ")   #
			                                                  #
			#* * * * * * * * * * * * * * * * * * * * * * * * *#
			#                                                 #
			#  $  A bit of renaming here to be consistent  $  #
			#  $    throughout. vLOS is vx in 2D always    $  #
			#                                                 #
			#* * * * * * * * * * * * * * * * * * * * * * * * *#
			if not np.any(np.sum(onaxiscases - [ndims, fi0, th0], axis = 1) == 0.):
				                                          #
				vx, Bx = vLOS, Blos                       #
				                                          #
				vz = vx.copy()                            #
	                                                                  #
else:                                                                     #
	                                                                  #
	dens, mol, vLOS, T, PopRat, nlow, LOSaxis, RowAxis, Blos, dist = None, None, None, None, None, None, None, None, None, None
	                                                                  #
	vx, vy, vz, Bx, By, Bz = None, None, None, None, None, None       #
                                                                          #
dens, mol, vLOS, T = comm.bcast(dens, root=0), comm.bcast(mol, root=0), comm.bcast(vLOS, root=0), comm.bcast(T, root=0)
                                                                          #
PopRat, nlow, LOSaxis, RowAxis, Blos, dist = comm.bcast(PopRat, root=0), comm.bcast(nlow, root=0), comm.bcast(LOSaxis, root=0), comm.bcast(RowAxis, root=0), comm.bcast(Blos, root=0), comm.bcast(dist, root=0)
                                                                          #
vx, vy, vz, Bx, By, Bz = comm.bcast(vx, root=0), comm.bcast(vy, root=0), comm.bcast(vz, root=0), comm.bcast(Bx, root=0), comm.bcast(By, root=0), comm.bcast(Bz, root=0)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                                                                         #
#         \\                                              //              #
#          \\                                            //               #
#           \\  $$ Generic case, on axis integration $$ //                #
#           //                                          \\                #
#          //                                            \\               #
#         //                                              \\              #
#                                                                         #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
if np.any(np.sum(onaxiscases - [ndims, fi0, th0], axis = 1) == 0.):       #
	                                                                  #
	#-----------------------------------------------------------------#
	#                                                                 #
	# $ Consider half the grid in 2D case to save computational time $#
	#                                                                 #
	#-----------------------------------------------------------------#
	size = dens.shape                                                 #
	                                                                  #
	if ndims == 2:                                                    #
		                                                          #
		temp = RowAxis[len(RowAxis)//2:]                          #
		                                                          #
	else:                                                             #
		                                                          #
		temp = RowAxis                                            #
		                                                          #
	cpuRow = np.empty(len(temp)//nproc, dtype=np.float64)             #
	                                                                  #
	comm.Scatter([temp, MPI.DOUBLE], [cpuRow, MPI.DOUBLE])            #
	                                                                  #
	PPV, pr = [], 0.                                                  #
	                                                                  #
	if GK: PPVParal = []                                              #
	                                                                  #
	for j in range (0, len(cpuRow)):                                  #
		                                                          #
		j, pr = np.argmin(np.absolute(RowAxis-cpuRow[j])), pr + 1./len(cpuRow)
		                                                          #
		if iproc==0: update_progress(pr)                          #
		                                                          #
		rows=np.empty((size[1], len(frRange)))                    #
		                                                          #
		if GK: rowsParal=np.empty((size[1], len(frRange)))        #
		                                                          #
		for i in range (0, size[1]):                              #
			                                                  #
			ILOS=np.empty(len(frRange))                       #
			                                                  #
			if GK: ILOSParal = np.empty(len(frRange))         #
			                                                  #
			for f in range (0, len(frRange)):                 #
				                                          #
				frq = frRange[f]                          #
				                                          #
				I_a, vela, Sline_a, Itotal, LineAbs_a, Sc_a, kext_a= 0., 0., 0., 0., 0., 0., 0.
				                                          #
				if GK: I_aP, Sline_aP, ItotalP, LineAbs_aP = 0., 0., 0., 0.
				                                          #
				for k in range (0, size[2]):              #
					                                  #
					densgp, Tgp, PopRatgp, nlowgp, velb=dens[j, i, k], T[j, i, k], PopRat[j, i, k], nlow[j, i, k], vLOS[j, i, k]
					                                  #
					Sline_b, LineAbs_b=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK=False)
					                                  #
					Sc_b, kext_b = dust.dust(frq/c, Tgp, densgp)
					                                  #
					Dthrm = np.sqrt(Kb*Tgp/molmass/c**2) * freqs[low]
					                                  #
					I_b=radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_a, Sc_a, Sc_b, Sline_a, Sline_b, LineAbs_a, LineAbs_b, kext_a, kext_b)
					                                  #
					Itotal = I_b                      #
					                                  #
					if GK == True:                    #
						#   LOS vector: (1, 0, 0) #
						#  gam = LOSvec * Bvec /  #
						#  (|LOSvector||Bvector|) #
						gam = np.arccos(Blos[j, i, k]/np.sqrt(Bx[j, i, k]**2 + By[j, i, k]**2 + Bz[j, i, k]**2))
						                          #
						Sline_bP, LineAbs_bP=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK)
						                          #
						I_bP=radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_aP, Sc_a, Sc_b, Sline_aP, Sline_bP, LineAbs_aP, LineAbs_bP, kext_a, kext_b)
						                          #
						ItotalP = I_bP            #
						                          #
						I_aP, Sline_aP, LineAbs_aP = I_bP, Sline_bP, LineAbs_bP
					                                  #
					I_a, Sline_a, LineAbs_a, Sc_a, kext_a, vela = I_b, Sline_b, LineAbs_b, Sc_b, kext_b, velb
				                                          #
				ILOS[f]=Itotal                            #
				                                          #
				if GK: ILOSParal[f]=ItotalP               #
			                                                  #
			rows[i] = np.array(ILOS).astype(float)            #
			                                                  #
			if GK: rowsParal[i] = np.array(ILOSParal).astype(float)
			                                                  #
		PPV.append(np.array(rows).astype(float))                  #
		                                                          #
		if GK: PPVParal.append(np.array(rowsParal).astype(float)) #
	                                                                  #
	PPV_gathered = comm.gather(np.array(PPV), root = 0)               #
	                                                                  #
	if GK: PPV_gatheredParal = comm.gather(np.array(PPVParal), root = 0)
	                                                                  #
	if iproc == 0:                                                    #
		                                                          #
		PPV = np.array(PPV_gathered).astype(float)                #
		                                                          #
		np.save('{}/output_files/PPV'.format(cwd), PPV)           #
		                                                          #
		size2 = PPV.shape                                         #
		                                                          #
		PPV = PPV.reshape(nproc*size2[1], size2[2], size2[3])     #
		                                                          #
		if ndims == 2:                                            #
			                                                  #
			PPV = np.vstack((np.flipud(PPV), PPV))            #
			                                                  #
			PPV = np.hstack((np.fliplr(PPV), PPV))            #
		                                                          #
		np.save('{}/output_files/PPV'.format(cwd), PPV)           #
		                                                          #
		if GK:                                                    #
			                                                  #
			PPVParal = np.array(PPV_gatheredParal).astype(float)
			                                                  #
			np.save('{}/output_files/PPVParal'.format(cwd), PPVParal)
			                                                  #
			PPVParal = PPVParal.reshape(nproc*size2[1], size2[2], size2[3])
			                                                  #
			if ndims == 2:                                    #
				                                          #
				PPVParal = np.vstack((np.flipud(PPVParal), PPVParal))
				                                          #
				PPVParal = np.hstack((np.fliplr(PPVParal), PPVParal))
			                                                  #
			np.save('{}/output_files/PPVParal'.format(cwd), PPVParal)
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#         \\                                              //              #
#          \\                                            //               #
#           \\ $$ Generic case, off axis integration $$ //                #
#           // $$ Run "rayTraceTest.py" for examples $$ \\                #
#          //                                            \\               #
#         //                                              \\              #
#                                                                         #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
else:                                                                     #
	                                                                  #
	ux = np.cos(np.radians(th0)) * np.sin(np.radians(fi0))            #
	                                                                  #
	uy = np.sin(np.radians(th0)) * np.sin(np.radians(fi0))            #
	                                                                  #
	uz = np.cos(np.radians(fi0))                                      #
	                                                                  #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#  $$             Direction vector of the line               $$   #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *.#
	rayvec = np.array([uy, ux, uz])                                   #
	                                                                  #
	if ndims == 2: rayvec[1], z = 0., np.concatenate((-z[::-1], z))   #
	                                                                  #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#   $$       What should be the starting y to have a ray     $$   #
	#   $$               that passes through y[0]?               $$   #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *.#
	                                                                  #
	minY, maxY = np.inf, -np.inf                                      #
	                                                                  #
	for j in range (0, len(y)):                                       #
		                                                          #
		for k in range (0, len(z)):                               #
			                                                  #
			yray = y[j] + (z - z[k])/rayvec[2]*rayvec[0]      #
			                                                  #
			if np.max(yray) > maxY: maxY = np.max(yray)       #
			                                                  #
			if np.min(yray) < minY: minY = np.min(yray)       #
			                                                  #
	if ndims == 3:                                                    #
		                                                          #
		minX, maxX = np.inf, -np.inf                              #
		                                                          #
		for i in range (0, len(x)):                               #
			                                                  #
			for k in range (0, len(z)):                       #
				                                          #
				xray = x[i] + (z - z[k])/rayvec[2]*rayvec[1]
				                                          #
				if np.max(xray) > maxX: maxX = np.max(xray)
				                                          #
				if np.min(xray) < minX: minX = np.min(xray)
				                                          #
	else:                                                             #
		                                                          #
		minX, maxX = np.min(x), np.max(x[len(x)//2:])             #
	                                                                  #
	#     $$$              Define ray limits               $$$        #
	yy, xx = np.arange(minY, maxY+dy, dy), np.arange(minX, maxX+dx, dx)
	                                                                  #
	#     $$$    x, y, z are coordinates in cell centers   $$$        #
	#     $$$   adjust accordingly to go up to cell faces  $$$        #
	minX, maxX, minY, maxY = np.min(x) - dx/2., np.max(x) + dx/2., np.min(y) - dy/2., np.max(y) + dy/2.
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                 #
	#     $$$   Fix parallelization - we are now sending   $$$        #
	#     $$$        uneven data to each processor         $$$        #
	#                                                                 #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	ave, res = divmod(yy.size, nproc)                                 #
	                                                                  #
	count = np.array([ave + 1 if p < res else ave for p in range(nproc)])
	                                                                  #
	cutlocs = np.array([sum(count[:p]) for p in range(nproc)])        #
	                                                                  #
	cpuRow = np.empty(count[iproc])                                   #
	                                                                  #
	comm.Scatterv([yy, count, cutlocs, MPI.DOUBLE], cpuRow, root=0)   #
	                                                                  #
	del ave, res, yray                                                #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	                                                                  #
	PPV, pr, size = [], 0., dens.shape                                #
	                                                                  #
	if GK: PPVParal = []                                              #
	                                                                  #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                 #
	#          $$$   jrs, irs, krs are ray coordinates   $$$          #
	#          $$$      j, i, k are grid coordinates     $$$          #
	#                                                                 #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	for jrs in range (0, len(cpuRow)):                                #
		                                                          #
		yray = cpuRow[jrs] + (z - z[0])/rayvec[2]*rayvec[0]       #
		                                                          #
		pr = pr + 1./len(yy)                                      #
		                                                          #
		if iproc==0: update_progress(pr)                          #
		                                                          #
		rows=np.empty((len(xx), len(frRange)))                    #
		                                                          #
		if GK: rowsParal=np.empty((len(xx), len(frRange)))        #
		                                                          #
		for irs in range (0, len(xx)):                            #
			                                                  #
			xray = xx[irs] + (z - z[0])/rayvec[2]*rayvec[1]   #
			                                                  #
			ILOS=np.zeros(len(frRange))                       #
			                                                  #
			if GK: ILOSParal = np.zeros(len(frRange))         #
			                                                  #
			for f in range (0, len(frRange)):                 #
				                                          #
				I_a, vela, Sline_a, Itotal, LineAbs_a, Sc_a, kext_a, frq = 0., 0., 0., 0., 0., 0., 0., frRange[f]
				                                          #
				if GK: I_aP, Sline_aP, ItotalP, LineAbs_aP = 0., 0., 0., 0.
				                                          #
				jref, iref, fcall = 0, 0, True            #
				                                          #
				for krs in range (0, size[2]):            #
					#* * * * * * * * * * * * * * * * *#
					#  $$ Check if we are inside $$   #
					#  $$        the grid        $$   #
					#* * * * * * * * * * * * * * * * *#
					if np.logical_or(yray[jref] > maxY, yray[jref] < minY) or np.logical_or(xray[iref] > maxX, xray[iref] < minX):
						                          #
						jref, iref = jref + 1, iref + 1
						                          #
						continue                  #
					                                  #
					#* * * * * * * * * * * * * * * * *#
					# $ if not first call, InPoint $  #
					# $      will be Fpoint        $  #
					#* * * * * * * * * * * * * * * * *#
					if fcall: InPoint = np.array([yray[jref], xray[iref], z[krs]])
					                                  #
					dist, FPoint, j, i, k, exitGrid, vLOS, BLOS = distance.distance(rayvec, InPoint, x, y, z, vx, vy, vz, fcall, Bx, By, Bz, GK)
					                                  #
					if exitGrid: break                #
					                                  #
					if fcall: fcall = False           #
					#* * * * * * * * * * * * * * * * *#
					#    j, i, k defined on entry     #
					#     point in cell InPoint       #
					#* * * * * * * * * * * * * * * * *#
					                                  #
					densgp, Tgp, PopRatgp, nlowgp, velb = dens[j, i, k], T[j, i, k], PopRat[j, i, k], nlow[j, i, k], vLOS
					                                  #
					Sline_b, LineAbs_b=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK=False)
					                                  #
					Sc_b, kext_b = dust.dust(frq/c, Tgp, densgp)
					                                  #
					Dthrm = np.sqrt(Kb*Tgp/molmass/c**2) * freqs[low]
					                                  #
					I_b = radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_a, Sc_a, Sc_b, Sline_a, Sline_b, LineAbs_a, LineAbs_b, kext_a, kext_b)
					                                  #
					Itotal = I_b                      #
					                                  #
					if GK == True:                    #
						#  gam = LOSvec * Bvec /  #
						#  (|LOSvector||Bvector|) #
						gam = np.arccos(Blos/np.sqrt(Bx[j, i, k]**2 + By[j, i, k]**2 + Bz[j, i, k]**2))
						                          #
						Sline_bP, LineAbs_bP=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK)
						                          #
						I_bP=radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_aP, Sc_a, Sc_b, Sline_aP, Sline_bP, LineAbs_aP, LineAbs_bP, kext_a, kext_b)
						                          #
						ItotalP = I_bP            #
						                          #
						I_aP, Sline_aP, LineAbs_aP = I_bP, Sline_bP, LineAbs_bP
					                                  #
					I_a, Sline_a, LineAbs_a, Sc_a, kext_a, vela = I_b, Sline_b, LineAbs_b, Sc_b, kext_b, velb
					                                  #
					#* * * * * * * * * * * * * * * * *#
					#    $$ ! update InitPoint ! $$   #
					#* * * * * * * * * * * * * * * * *#
					InPoint = FPoint                  #
					                                  #
					#* * * * * * * * * * * * * * * * *#
					#                                 #
					# !+++++++++++++++++++++++++++++! #
					# ! If fi0 or th0 > 45, we will ! #
					# !  run out of krs iterations  ! #
					# ! before reaching the end of  ! #
					# !  the grid for some rays     ! #
					# ! "while" loop does not work  ! #
					# !    for fi0/th0 < 45.        ! #
					# ! So here we will have an     ! #
					# ! extra step for the same krs ! #
					# !+++++++++++++++++++++++++++++! #
					#                                 #
					#* * * * * * * * * * * * * * * * *#
					if fi0 > 45. or th0 > 45.:        #
						                          #
						dist, FPoint, j, i, k, exitGrid, vLOS, BLOS = distance.distance(rayvec, InPoint, x, y, z, vx, vy, vz, fcall, Bx, By, Bz, GK)
						                          #
						if exitGrid: break        #
						                          #
						densgp, Tgp, PopRatgp, nlowgp = dens[j, i, k], T[j, i, k], PopRat[j, i, k], nlow[j, i, k]
						                          #
						Sline_b, LineAbs_b=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK=False)
						                          #
						Sc_b, kext_b = dust.dust(frq/c, Tgp, densgp)
						                          #
						Dthrm = np.sqrt(Kb*Tgp/molmass/c**2) * freqs[low]
						                          #
						I_b = radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_a, Sc_a, Sc_b, Sline_a, Sline_b, LineAbs_a, LineAbs_b, kext_a, kext_b)
						                          #
						Itotal = I_b              #
						                          #
						if GK == True:            #
							                  #
							gam = np.arccos(Blos/np.sqrt(Bx[j, i, k]**2 + By[j, i, k]**2 + Bz[j, i, k]**2))
							                  #
							Sline_bP, LineAbs_bP=sourcef.sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK)
							                  #
							I_bP=radTran.radTran(frq, vela, velb, dist, freqs[low], Dthrm, I_aP, Sc_a, Sc_b, Sline_aP, Sline_bP, LineAbs_aP, LineAbs_bP, kext_a, kext_b)
							                  #
							ItotalP = I_bP    #
							                  #
							I_aP, Sline_aP, LineAbs_aP = I_bP, Sline_bP, LineAbs_bP
							                  #
						I_a, Sline_a, LineAbs_a, Sc_a, kext_a, vela = I_b, Sline_b, LineAbs_b, Sc_b, kext_b, velb
						                          #
						#* * * * * * * * * * * * *#
						# $ reupdate InitPoint  $ #
						#* * * * * * * * * * * * *#
						InPoint = FPoint          #
				                                          #
				ILOS[f]=Itotal                            #
				                                          #
				if GK: ILOSParal[f]=ItotalP               #
			                                                  #
			rows[irs] = np.array(ILOS).astype(float)          #
			                                                  #
			if GK: rowsParal[irs] = np.array(ILOSParal).astype(float)
			                                                  #
		PPV.append(np.array(rows).astype(float))                  #
		                                                          #
		if GK: PPVParal.append(np.array(rowsParal).astype(float)) #
	                                                                  #
	PPV = np.array(PPV)                                               #
	                                                                  #
	PPV_gathered = np.empty((len(yy), PPV.shape[1], PPV.shape[2]))    #
	                                                                  #
	comm.Gatherv(PPV, [PPV_gathered, (np.array(count)*PPV.shape[1]*PPV.shape[2]).tolist(), (np.array(cutlocs)*PPV.shape[1]*PPV.shape[2]).tolist(), MPI.DOUBLE], root=0)
	                                                                  #
	if GK:                                                            #
		                                                          #
		PPVParal = np.array(PPVParal)                             #
		                                                          #
		PPV_gatheredParal = np.empty((len(yy), PPVParal.shape[1], PPVParal.shape[2]))
		                                                          #
		comm.Gatherv(PPVParal, [PPV_gatheredParal, (np.array(count)*PPVParal.shape[1]*PPVParal.shape[2]).tolist(), (np.array(cutlocs)*PPVParal.shape[1]*PPVParal.shape[2]).tolist(), MPI.DOUBLE], root=0)
	                                                                  #
	if iproc == 0:                                                    #
		                                                          #
		np.save('{}/output_files/PPV'.format(cwd), PPV_gathered)  #
		                                                          #
		if GK: np.save('{}/output_files/PPVParal'.format(cwd), PPV_gatheredParal)
		                                                          #
		if ndims == 2:                                            #
			                                                  #
			PPV = np.hstack((np.fliplr(PPV_gathered), PPV_gathered))
			                                                  #
			np.save('{}/output_files/PPV'.format(cwd), PPV)   #
			                                                  #
			if GK:                                            #
				                                          #
				PPVParal = np.hstack((np.fliplr(PPV_gatheredParal), PPV_gatheredParal))
				                                          #
				np.save('{}/output_files/PPVParal'.format(cwd), PPVParal)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
