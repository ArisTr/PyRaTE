#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#  NAME:                                                                  #
#                                                                         #
#     setup                                                               #
#                                                                         #
#  DESCRIPTION:                                                           #
#                                                                         #
#     This is the setup script for the RT simulation you want to run      #
#     Also load arrays from "export_sim" here to declutter "main"         #
#                                                                         #
#  PARAMETERS:                                                            #
#                                                                         #
#     fi0, th0: sperical coordinates angles (explanation below)           #
#     up, low = transition to consider                                    #
#     Vres: spectral resolution in km/s                                   #
#     npoints: number of frequency points. That number along with the     #
#              spectr_res will determine the bandwidth                    #
#                                                                         #
#  LOGICAL                                                                #
#                                                                         #
#     line: Single line (True) or PPV cube (False)?                       #
#     nonLTE: nonLTE or LTE simulation?                                   #
#     GK: Perform simulations of the GK effect                            #
#     BgCMB: Whether to assume or not a CMB background                    #
#     restart : Restart simulation assuming LevelPopulations have         #
#               already been computed                                     #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#  AUTHOR:                                                                #
#                                                                         #
#  Aris E. Tritsis                                                        #
#  (aris.tritsis@epfl.ch)                                                 #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#  NOTES:                                                                 #
#                                                                         #
#        fi will be needed for cartesian/cylindrical coordiantes          #
#        For cylindrical geometry, fi = 0 (edge on), fi = 90 face on      #
#                                  and th0 = 90. always...                #
#                                                                         #
#                                                                         #
#                             /---------/|        z ^                     #
#                            /         / |          |                     #
#                           /---------/  |     \fi  |                     #
#                           |         |  |      \⌒ |                     #
#                           |         |  |       \  |                     #
#                           |         |  |        \ |                     #
#                           |         | /          \|------------> y      #
#                           |---------|/           / \                    #
#                                                 / ⌣ \                   #
#                                                /  th \                  #
#                                             x V       \                 #
#                                                                         #
#                                                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
import numpy as np                                                        #
import os                                                                 #
import sys                                                                #
import welcome                                                            #
from scipy.constants import c                                             #
import itertools                                                          #
                                                                          #
c = c*1e+2                                                                #
cwd = os.getcwd()                                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
def setup():                                                              #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#     \\                                              //          #
	#      \\                                            //           #
	#       \\       $$  Parameter Input here  $$       //            #
	#       //                                          \\            #
	#      //                                            \\           #
	#     //                                              \\          #
	#                                                                 #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	fi0, th0 = 90., 0.                                                #
	up, low = 1, 0                                                    #
	npoints, Vres = 64, 0.05                                         #
	                                                                  #
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
	#                 Logical parameters....                          #
	                                                                  #
	line, nonLTE, GK, BgCMB = False, True, True, True                 #
	restart = False                                                   #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#     \\                                              //          #
	#      \\                                            //           #
	#       \\     $$  End of Input Parameters  $$      //            #
	#       //                                          \\            #
	#      //                                            \\           #
	#     //                                              \\          #
	#                                                                 #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#              LOAD DATA & COEFFICIENTS                           #
	ndims = int(np.load("simdata/ndims.npy"))                         #
	dens = np.load("simdata/dens.npy")                                #
	mol = np.load("simdata/mol.npy")                                  #
	T = np.load("simdata/T.npy")                                      #
	                                                                  #
	vx = np.load("simdata/velx.npy")                                  #
	x = np.load("simdata/x.npy")                                      #
	vy = np.load("simdata/vely.npy")                                  #
	y = np.load("simdata/y.npy")                                      #
	vz = np.load("simdata/velz.npy")                                  #
	z = np.load("simdata/z.npy")                                      #
	                                                                  #
	Cul = np.load("simdata/Cul.npy")                                  #
	Clu = np.load("simdata/Clu.npy")                                  #
	EinA = np.load("simdata/EinA.npy")                                #
	EinBul = np.load("simdata/EinBul.npy")                            #
	EinBlu = np.load("simdata/EinBlu.npy")                            #
	Ener = np.load("simdata/Ener.npy")                                #
	freqs = np.load("simdata/TranFreq.npy")                           #
	gul = np.load("simdata/gul.npy")                                  #
	tempers = np.load("simdata/tempers.npy")                          #
	molmass = np.load("simdata/molmass.npy")                          #
	numlevels = int(np.load("simdata/numlevels.npy"))                 #
	                                                                  #
	if GK == True:                                                    #
		Bx = np.load("simdata/magx.npy")                          #
		By = np.load("simdata/magy.npy")                          #
		Bz = np.load("simdata/magz.npy")                          #
		CulGK = np.load("simdata/CulGK.npy")                      #
		CluGK = np.load("simdata/CluGK.npy")                      #
		jmjpmp = np.load("simdata/jmjpmp.npy")                    #
	else:                                                             #
		Bx, By, Bz = 0., 0., 0.                                   #
		CulGK, CluGK = 0., 0.                                     #
		jmjpmp = None                                             #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#            $$$  Initialization of additional arrays  $$$        #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	                                                                  #
	Vrange = np.arange(-npoints/2 * Vres + Vres/2, npoints/2 *Vres+Vres/2, Vres)
	                                                                  #
	frRange = freqs[low] * (1 - Vrange * 1e+5/c)                      #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#             $$$  Fix frRange in case of GK effect  $$$          #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	if GK: frRange = freqs[np.where(jmjpmp[:, 2] == low)[0][0]] * (1 - Vrange * 1e+5/c)
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                 $$$    Initial checks    $$$                    #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	if (line==True and ((fi0 !=0 and fi0 !=90) or (th0 !=0 and th0 !=90))):
		                                                          #
		raise SystemExit("Unsupported angle for single line calculations. Sorry :( !!")
	                                                                  #
	if (Vrange[-1] - Vrange[0]) < np.absolute(np.max(vx) - np.min(vx)) * 1e-5:
		                                                          #
		raise SystemExit("Velocity range smaller than velocity spread in the simulation. Consider increasing npoints and/or Vres !!")
	                                                                  #
	if (not np.logical_or(th0 >= 0., th0 <= 90.)) or (not np.logical_or(fi0 >= 0., fi0 <= 90.)):
		                                                          #
		raise SystemExit("fi or theta must be in the range [0., 90.]")
	                                                                  #
	# $$ Small hack to separate between cases based on projection  $$ #
	# $$           angle and dimensionality of grid                $$ #
	#                                                                 #
	#               ndims     fi0        th0                          #
	onaxiscases = ([2, 3], [0., 90.], [0., 90.])                      #
	                                                                  #
	onaxiscases = np.array(list(itertools.product(*onaxiscases)))     #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#             $$$    GK additional checks    $$$                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	if (GK == True and nonLTE == False):                              #
		                                                          #
		raise SystemExit("Setup does not make sense. No GK effect under LTE!")
		                                                          #
	if (GK == True and ndims == 1):                                   #
		                                                          #
		raise SystemExit("GK effect not implemented under 1D spherical symmetry!")
		                                                          #
	if (GK == True and (not np.any(np.sum(onaxiscases - [ndims, th0, fi0], axis = 1) == 0.))):
		                                                          #
		print("\n")                                               #
		print("                            /\ ")                  #
		print("                           /  \ ")                 #
		print("                          / || \ " + "  GK effect not thoroughly tested")
		print("                         /  ||  \ " + " for off-axis calculations")
		print("                        /   ..   \ " + " Proceed with caution !!")
		print("                       /          \ ")             #
		print("                       ------------ ")             #
		print("\n")                                               #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                 $$$    welcome & dir setup    $$$               #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	welcome.welcome()                                                 #
	                                                                  #
	if not os.path.exists('{}/output_files'.format(cwd)):             #
		                                                          #
		os.mkdir('{}/output_files'.format(cwd))                   #
	                                                                  #
	np.save("{}/output_files/Vrange".format(cwd), Vrange)             #
	                                                                  #
	np.save("{}/output_files/frRange".format(cwd), frRange)           #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#  $$$ dimensionality fix for consistency between functions $$$   #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	if ndims == 2:                                                    #
		                                                          #
		dens, mol, T = dens[:, :, np.newaxis], mol[:, :, np.newaxis], T[:, :, np.newaxis]
		                                                          #
		vx, vy, vz = vx[:, :, np.newaxis], vy[:, :, np.newaxis], vz[:, :, np.newaxis]
		                                                          #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                         #
		#       $$$ Fix for 2D cylindrical simulations $$$        #
		#       $$$        z is saved as y....         $$$        #
		#                                                         #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		th0 = 90.                                                 #
		                                                          #
		if GK:                                                    #
			                                                  #
			Bx, By, Bz = Bx[:, :, np.newaxis], By[:, :, np.newaxis], Bz[:, :, np.newaxis]
	if ndims == 1:                                                    #
		                                                          #
		dens, mol, T, vx, vy, vz = dens[:, np.newaxis], mol[:, np.newaxis], T[:, np.newaxis], vx[:, np.newaxis], vy[:, np.newaxis], vz[:, np.newaxis]
		                                                          #
		dens, mol, T = dens[:, :, np.newaxis], mol[:, :, np.newaxis], T[:, :, np.newaxis]
		                                                          #
		vx, vy, vz = vx[:, :, np.newaxis], vy[:, :, np.newaxis], vz[:, :, np.newaxis]
		                                                          #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#                    Time to return                               #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	return fi0, th0, up, low, npoints, Vres, line, nonLTE, GK, BgCMB, restart, ndims, dens, mol, T, vx, x, vy, y, vz, z, Cul, Clu, EinA, EinBul, EinBlu, Ener, freqs, gul, tempers, Bx, By, Bz, CulGK, CluGK, Vrange, frRange, molmass, numlevels, jmjpmp, onaxiscases
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
