#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#  NAME:                                                                  #
#                                                                         #
#     export_sim                                                          #
#                                                                         #
#  DESCRIPTION:                                                           #
#                                                                         #
#     This script exports all data from the simulations in numpy arrays   #
#     It also exports plots and performs checks to verify that assumed    #
#     symmetries/geometries are as expected                               #
#                                                                         #
#     Finally, it exports collisional and Einstein coefficients and       #
#     stores them in a convenient format                                  #
#                                                                         #
#  SIMULATION PARAMETERS:                                                 #
#                                                                         #
#     molecule: which molecule you want to consider                       #
#     noD: whether the simulations includes deuterium or not              #
#     symmetryplane2D: if 2D simulation determine whether one or two      #
#                      quadrants are simulated (same keyword as flash)    #
#     path: path and name of the simulation output considered             #
#     level: AMR level to reconstruct the grid. If string                 #
#            ("maximum"/"minimum") then level= pf.max_level/pf.min_level  #
#            Otherwise should be an integer                               #
#     zoom: Whether to zoom in towards the center of the simulation       #
#     zoomFactor: How much to zoom in (% should be in [0, 1) )            #
#     smooth: Smoothens the grid to avoid duplicate values due to AMR     #
#                                                                         #
#  RADIATIVE PARAMETERS:                                                  #
#                                                                         #
#     nlevels: number of energy levels to consider                        #
#     GK: whether to create collisional coefficints for the GK effect     #
#     GKfact: how much smaller/greater sublevel collisional               #
#             coefficient will be compared to regular levels              #
#                                                                         #
#  AUTHOR:                                                                #
#                                                                         #
#  Aris E. Tritsis                                                        #
#  (aris.tritsis@epfl.ch)                                                 #
                                                                          #
from yt import *                                                          #
import numpy as np                                                        #
from scipy.constants import m_p, parsec, c                                #
from scipy.ndimage import gaussian_filter                                 #
from sympy.physics.wigner import wigner_3j                                #
import os                                                                 #
import sys                                                                #
                                                                          #
m_p, parsec, siny, c = m_p*1e+3, parsec*1e+2, 31556926., c*1e+2           #
a_factor_i, mcr_ions, elec, h = 1.14, 1.69e-9, 4.80320425e-10, 6.6261e-27 #
amu=2.4237981621576                                                       #
cwd = os.getcwd()                                                         #
                                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#        \\                                              //               #
#         \\                                            //                #
#          \\       $$  Parameter input here  $$       //                 #
#          //                                          \\                 #
#         //                                            \\                #
#        //                                              \\               #
#                                                                         #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
                                                                          #
molecule = "CO"                                                           #
noD = True                                                                #
symmetryplane2D = False                                                   #
path = os.path.abspath('DiskMHD_hdf5_chk_1047')                           #
level = "maximum"                                                         #
zoom = False                                                              #
zoomFactor = 0.5                                                          #
smooth = True                                                             #
                                                                          #
nlevels = 4                                                               #
GK, GKfact = True, 1.                                                     #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#        \\                                              //               #
#         \\                                            //                #
#          \\     $$  End of input Parameters  $$      //                 #
#          //                                          \\                 #
#         //                                            \\                #
#        //                                              \\               #
#                                                                         #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
                                                                          #
#        Masses and specNames specific to chemical modeling by            #
#                 Tritsis+2016/2021/2023                                  #
if noD == True:                                                           #
	massesO = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 19.0, 24.0, 24.0, 25.0, 25.0, 26.0, 26.0, 26.0, 26.0, 27.0, 27.0, 27.0, 27.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 29.0, 29.0, 29.0, 30.0, 30.0, 30.0, 30.0, 31.0, 32.0, 32.0, 32.0, 33.0, 36.0, 37.0, 38.0, 38.0, 39.0, 41.0, 42.0, 44.0, 44.0, 45.0, 51.0, 52.0, 1.0, 12.0, 28.0, 30.0, 28.0, 24.0, 32.0, 13.0, 17.0, 30.0, 14.0, 18.0, 44.0, 15.0, 16.0, 27.0, 29.0, 26.0, 51.0, 28.0, 26.0, 15.0, 27.0, 25.0, 17.0, 41.0, 32.0, 16.0, 14.0, 16.0, 2.0, 31.0, 38.0, 37.0, 38.0, 38.0, 37.0])
	                                                                  #
	specNames = np.array(["H+", "H", "H2+", "H3+", "He", "He+", "C+", "C", "CH", "CH+", "CH2+", "CH2", "N", "N+", "CH3", "NH+", "CH3+", "NH", "NH2+", "O", "CH4", "CH4+", "O+", "NH2", "CH5+", "OH", "OH+", "NH3+", "NH3", "H2O", "NH4+", "H2O+", "H3O+", "C2", "C2+", "C2H+", "C2H", "C2H2+", "C2H2", "CN", "CN+", "HCN+", "C2H3+", "HCN", "HNC", "Si+", "C2H4+", "H2NC+", "Si", "N2", "CO+", "HCNH+", "CO", "N2+", "HCO", "N2H+", "HCO+", "H2CO", "H2CO+", "NO", "NO+", "H3CO+", "CH3OH", "O2", "O2+", "CH3OH2+", "C3+", "C3H+", "C2N+", "CNC+", "C3H3+", "CH3CN", "CH3CNH+", "CO2", "CO2+", "HCO2+", "HC3N", "HC3NH+", "GH", "GC", "GCO", "GH2CO", "GSi", "GC2", "GO2", "GCH", "GOH", "GNO", "GCH2", "GH2O", "GCO2", "GCH3", "GCH4", "GHNC", "GHCO", "GC2H2", "GHC3N", "GN2", "GCN", "GNH", "GHCN", "GC2H", "GNH3", "GCH3CN", "GCH3OH", "GNH2", "GN", "GO", "GH2", "GCH2OH", "C3H2", "C3H", "C3H2+", "GC3H2", "GC3H"])
else:                                                                     #
	massesO = np.array([ 1.,1.,2.,3.,4.,4., 12., 12., 13., 13., 14., 14., 14.,14., 15., 15., 15., 15., 16., 16., 16., 16., 16., 16., 17., 17.,17., 17., 17., 18., 18., 18., 19., 24., 24., 25., 25., 26., 26.,26., 26., 27., 27., 27., 27., 28., 28., 28., 28., 28., 28., 28.,28., 28., 29., 29., 29., 30., 30., 30., 30., 31., 32., 32., 32.,33., 36., 37., 38., 38., 39., 41., 42., 44., 44., 45., 51., 52., 2.,2.,3.,4.,4.,5.,6., 14., 14., 15., 16., 15., 16.,16., 17., 18., 16., 16., 17., 18., 16., 17., 18., 17., 18., 19.,20., 17., 18., 19., 20., 17., 18., 18., 19., 20., 21., 22., 18.,18., 18., 19., 20., 18., 19., 20., 19., 20., 19., 20., 21., 22.,19., 20., 20., 21., 22., 26., 26., 27., 26., 27., 28., 28., 28.,29., 30., 28., 28., 29., 30., 31., 32., 29., 30., 29., 30., 31.,30., 30., 30., 31., 32., 31., 32., 32., 33., 34., 33., 34., 35.,35., 34., 35., 36., 34., 34., 34., 35., 36., 35., 36., 37., 36.,37., 38., 38., 40., 41., 42., 42., 43., 44., 43., 43., 44., 45.,44., 45., 46., 46., 52., 53., 54., 55.,3.,1., 12., 28., 30.,28., 24., 32., 13., 17., 30., 14., 18., 44., 15., 16., 27., 29.,26., 51., 28., 26., 15., 27., 25., 17., 41., 32., 16., 14., 16., 2., 31.,2., 31., 32., 14., 18., 15., 16., 19., 20., 16., 17.,18., 17., 18., 19., 20., 28., 30., 27., 28., 52., 16., 28., 26.,18., 19., 20., 42., 43., 44., 33., 34., 35., 33., 34., 35., 36.,17., 18.,3.,4., 32., 33., 32., 33., 34., 38., 37., 38., 39.,38., 39., 39., 39., 38., 37., 39., 40., 38.])
	                                                                  #
	specNames = np.array(["H+", "H", "H2+", "H3+", "He", "He+", "C+", "C", "CH", "CH+", "CH2+", "CH2", "N", "N+", "CH3", "NH+", "CH3+", "NH", "NH2+", "O", "CH4", "CH4+", "O+", "NH2", "CH5+", "OH", "OH+", "NH3+", "NH3", "H2O", "NH4+", "H2O+", "H3O+", "C2", "C2+", "C2H+", "C2H", "C2H2+", "C2H2", "CN", "CN+", "HCN+", "C2H3+", "HCN", "HNC", "Si+", "C2H4+", "H2NC+", "Si", "N2", "CO+", "HCNH+", "CO", "N2+", "HCO", "N2H+", "HCO+", "H2CO", "H2CO+", "NO", "NO+", "H3CO+", "CH3OH", "O2", "O2+", "CH3OH2+", "C3+", "C3H+", "C2N+", "CNC+", "C3H3+", "CH3CN", "CH3CNH+", "CO2", "CO2+", "HCO2+", "HC3N", "HC3NH+", "D+ ", "D", "HD+", "D2+ ", "H2D+", "HD2+", "D3+", "CD", "CD+", "CHD+", "CD2+", "CHD", "CD2", "CH2D", "CHD2", "CD3", "ND+", "CH2D+", "CHD2+", "CD3+", "ND", "NHD+", "ND2+", "CH3D", "CH2D2", "CHD3", "CD4", "CH3D+", "CH2D2+", "CHD3+", "CD4+", "NHD", "ND2", "CH4D+", "CH3D2+", "CH2D3+", "CHD4+", "CD5+", "OD", "OD+", "NH2D+", "NHD2+", "ND3+", "NH2D", "NHD2", "ND3", "HDO", "D2O ", "NH3D+", "NH2D2+", "NHD3+", "ND4+ ", "HDO+", "D2O+", "H2DO+", "HD2O+", "D3O+", "C2D+", "C2D ", "C2HD+", "C2D2+", "C2HD", "C2D2", "DCN+", "C2H2D+", "C2HD2+", "C2D3+", "DCN", "DNC", "C2H3D+", "C2H2D2+", "C2HD3+", "C2D4+", "HDNC+", "D2NC+", "DCNH+", "HCND+", "DCND+", "DCO", "N2D+", "DCO+", "HDCO", "D2CO", "HDCO+", "D2CO+", "H2DCO+", "HD2CO+", "D3CO+", "CH2DOH", "CHD2OH", "CD3OH", "CH3OD", "CH2DOD", "CHD2OD ", "CD3OD", "CH3OHD+", "CH3OD2+", "CH2DOH2+", "CHD2OH2+", "CD3OH2+ ", "CH2DOHD+", "CHD2OHD+", "CD3OHD+", "CH2DOD2+", "CHD2OD2+", "CD3OD2+", "C3D+", "C3H2D+", "C3HD2+", "C3D3+", "CH2DCN", "CHD2CN", "CD3CN", "CH3CND+", "CH2DCNH+", "CHD2CNH+", "CD3CNH+", "CH2DCND+", "CHD2CND+", "CD3CND+", "DCO2+", "DC3N", "DC3NH+", "HC3ND+", "DC3ND+", "HD", "GH", "GC", "GCO", "GH2CO ", "GSi", "GC2", "GO2", "GCH", "GOH", "GNO", "GCH2", "GH2O", "GCO2", "GCH3", "GCH4", "GHNC", "GHCO", "GC2H2", "GHC3N", "GN2", "GCN", "GNH", "GHCN", "GC2H ", "GNH3", "GCH3CN", "GCH3OH", "GNH2", "GN", "GO", "GH2", "GCH2OH", "GD", "GHDCO", "GD2CO", "GCD", "GOD", "GCHD", "GCD2", "GHDO", "GD2O ", "GCH2D", "GCHD2", "GCD3", "GCH3D", "GCH2D2", "GCHD3", "GCD4", "GDNC", "GDCO", "GC2HD", "GC2D2", "GDC3N", "GND ", "GDCN", "GC2D", "GNH2D ", "GNHD2", "GND3", "GCH2DCN", "GCHD2CN", "GCD3CN", "GCH2DOH", "GCHD2OH", "GCD3OH", "GCH3OD", "GCH2DOD", "GCHD2OD", "GCD3OD", "GNHD", "GND2", "GHD", "GD2", "GCHDOH", "GCD2OH", "GCH2OD", "GCHDOD", "GCD2OD", "C3H2  ", "C3H", "C3H2+", "C3HD", "C3D", "C3HD+", "C3D2", "C3D2+", "GC3H2", "GC3H", "GC3HD", "GC3D2", "GC3D"])
                                                                          #
ionTF, species = [], []                                                   #
                                                                          #
for i in range (0, len(specNames)):                                       #
	                                                                  #
	if specNames[i][len(specNames[i])-1] == "+" or specNames[i][len(specNames[i])-1] == "p":
		                                                          #
		ionTF.append(True)                                        #
		                                                          #
	else:                                                             #
		                                                          #
		ionTF.append(False)                                       #
	                                                                  #
	species.append('m{:03}'.format(i+1))                              #
                                                                          #
ionTF, species = np.array(ionTF), np.array(species)                       #
                                                                          #
dictionary=dict(zip(species, specNames))                                  #
                                                                          #
for keys, values in dictionary.items():                                   #
	                                                                  #
	if values == molecule:                                            #
		                                                          #
		sp=keys                                                   #
                                                                          #
mi = int(sp[1:])                                                          #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
                                                                          #
#    Load simulations and perform initial checks that assumed geometry    #
#             & dimensionality are correct                                #
                                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
                                                                          #
pf=load('{}'.format(path))                                                #
                                                                          #
if pf.dimensionality > 1 and pf.geometry=="spherical":                    #
	                                                                  #
	raise SystemExit("Spherical geometry but dimensionality > 1. Not sure what the situation is here")
                                                                          #
if pf.dimensionality > 2 and pf.geometry=="cylindrical":                  #
	                                                                  #
	raise SystemExit("Cylindrical geometry but dimensionality > 2. Not sure what the situation is here")
                                                                          #
if pf.dimensionality < 3 and pf.geometry=="cartesian":                    #
	                                                                  #
	raise SystemExit("Cartesian geometry but dimensionality < 3. Not sure what the situation is here")
#-------------------------------------------------------------------------#
if level == "maximum":                                                    #
	                                                                  #
	level= pf.max_level                                               #
	                                                                  #
elif level == "minimum":                                                  #
	                                                                  #
	try:                                                              #
		                                                          #
		level = pf.min_level                                      #
		                                                          #
	except AttributeError:                                            #
		                                                          #
		level = 0                                                 #
		                                                          #
else:                                                                     #
	                                                                  #
	pass                                                              #
                                                                          #
dims=pf.domain_dimensions*pf.refine_by**level                             #
                                                                          #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
if pf.dimensionality == 1:                                                #
	                                                                  #
	dims=[dims[0], 1, 1]                                              #
	                                                                  #
elif pf.dimensionality == 2:                                              #
	                                                                  #
	dims=[dims[0], dims[1], 1]                                        #
	                                                                  #
else:                                                                     #
	                                                                  #
	dims=[dims[0], dims[1], dims[2]]                                  #
                                                                          #
cube=pf.covering_grid(level, left_edge=pf.domain_left_edge, dims=dims)    #
                                                                          #
dens, mol, velx, vely, velz, T = cube['dens'], cube[sp], cube['velx'], cube['vely'], cube['velz'], cube['temperature']
                                                                          #
dens, mol, velx, vely, velz, T = np.array(dens), np.array(mol), np.array(velx), np.array(vely), np.array(velz), np.array(T)
                                                                          #
mol = mol*dens/m_p/massesO[mi-1]                                          #
                                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#                          $$$ Spherical $$$                              #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
if pf.dimensionality == 1:                                                #
	                                                                  #
	dens, mol, velx, T = dens.reshape(dims[0]), mol.reshape(dims[0]), velx.reshape(dims[0]), T.reshape(dims[0])
	                                                                  #
	x, dx=cube['r'], cube['dr']                                       #
	                                                                  #
	x, dx=np.array(x), np.array(dx)                                   #
	                                                                  #
	x=np.linspace(np.min(x), np.max(x), dims[0])                      #
	                                                                  #
	y, z = x.copy(), x.copy()                                         #
	                                                                  #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#                          $$$ Cylindrical $$$                            #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
elif pf.dimensionality == 2:                                              #
	                                                                  #
	dens, mol, T = dens.reshape(dims[0], dims[1]), mol.reshape(dims[0], dims[1]), T.reshape(dims[0], dims[1])
	                                                                  #
	velx, vely = velx.reshape(dims[0], dims[1]), vely.reshape(dims[0], dims[1])
	                                                                  #
	dens, mol, velx, vely, T = np.rot90(dens), np.rot90(mol), np.rot90(velx), np.rot90(vely), np.rot90(T)
	                                                                  #
	x, dx, y, dy = cube['r'], cube['dr'], cube['z'], cube['dz']       #
	                                                                  #
	x, dx, y, dy = np.array(x), np.array(dx), np.array(y), np.array(dy)
	                                                                  #
	y, x=np.linspace(np.min(y), np.max(y), dims[1]), np.linspace(np.min(x), np.max(x), dims[0])
	                                                                  #
	z = x.copy()                                                      #
	                                                                  #
	#-----------------------------------------------------------------#
	#                 Take care of projections here                   #
	#-----------------------------------------------------------------#
	if symmetryplane2D == True:                                       #
		                                                          #
		dens2, mol2, velx2, vely2, T2, y2 = np.flipud(dens), np.flipud(mol), np.flipud(velx), -np.flipud(vely), np.flipud(T), -y[::-1]
		                                                          #
		dens, mol, T, y = np.vstack((dens, dens2)), np.vstack((mol, mol2)), np.vstack((T, T2)), np.vstack((y, y2))
		                                                          #
		velx, vely = np.vstack((velx, velx2)), np.vstack((vely, vely2))
	                                                                  #
	velz = np.zeros(velx.shape)                                       #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#                           $$$ Cartesian $$$                             #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
if pf.dimensionality == 3:                                                #
	                                                                  #
	dens, mol, T = dens.reshape(dims[0], dims[1], dims[2]), mol.reshape(dims[0], dims[1], dims[2]), T.reshape(dims[0], dims[1], dims[2])
	                                                                  #
	velx, vely, velz = velx.reshape(dims[0], dims[1], dims[2]), vely.reshape(dims[0], dims[1], dims[2]), velz.reshape(dims[0], dims[1], dims[2])
	                                                                  #
	dens, mol, velx, vely, velz, T = np.rot90(dens), np.rot90(mol), np.rot90(velx), np.rot90(vely), np.rot90(velz), np.rot90(T)
	                                                                  #
	x, dx, y, dy, z, dz = cube['x'], cube['dx'], cube['y'], cube['dy'], cube['z'], cube['dz']
	                                                                  #
	x, dx, y, dy, z, dz = np.array(x), np.array(dx), np.array(y), np.array(dy), np.array(z), np.array(dz)
	                                                                  #
	y, x, z=np.linspace(np.min(y), np.max(y), dims[0]), np.linspace(np.min(y), np.max(y), dims[1]), np.linspace(np.min(x), np.max(x), dims[2])
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - #
#                  Non-ideal MHD related arrays                           #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - #
try:                                                                      #
	#-----------------------------------------------------------------#
	#                  $$ B-field components $$                       #
	#-----------------------------------------------------------------#
	magx, magy, magz=cube['magx'], cube['magy'], cube['magz']         #
	                                                                  #
	magx, magy, magz=np.array(magx), np.array(magy), np.array(magz)   #
	                                                                  #
	if pf.dimensionality == 1:                                        #
		                                                          #
		magy = magy.reshape(dims[0])                              #
		                                                          #
		magx, magz = np.zeros(magy.shape), np.zeros(magy.shape)   #
		                                                          #
	elif pf.dimensionality == 2:                                      #
		                                                          #
		magx, magy = magx.reshape(dims[0], dims[1]), magy.reshape(dims[0], dims[1])
		                                                          #
		magx, magy = np.rot90(magx), np.rot90(magy)               #
		                                                          #
		if symmetryplane2D == True:                               #
			                                                  #
			magx2, magy2 = -np.flipud(magx), np.flipud(magy)  #
			                                                  #
			magx, magy = np.vstack((magx, magx2)), np.vstack((magy, magy2))
		                                                          #
		magz = np.zeros(magx.shape)                               #
	else:                                                             #
		                                                          #
		magx, magy, magz=magx.reshape(dims[0], dims[1], dims[2]), magy.reshape(dims[0], dims[1], dims[2]), magz.reshape(dims[0], dims[1], dims[2])
		                                                          #
		magx, magy, magz=np.rot90(magx), np.rot90(magy), np.rot90(magz)
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#           We might need B-field components for GK effect        #
	#                     but not the resistivities                   #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	try:                                                              #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		#                   $$ resistivities $$                   #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		etaPe, etaPa, etaHa=cube['mrpe'], cube['mrpa'], cube['mrha']
		                                                          #
		etaPe, etaPa, etaHa=np.array(etaPe), np.array(etaPa), np.array(etaHa)
		                                                          #
		if pf.dimensionality == 1:                                #
			                                                  #
			etaPe, etaPa, etaHa = etaPe.reshape(dims[0]), etaPa.reshape(dims[0]), etaHa.reshape(dims[0])
			                                                  #
		elif pf.dimensionality == 2:                              #
			                                                  #
			etaPe, etaPa, etaHa=etaPe.reshape(dims[0], dims[1]), etaPa.reshape(dims[0], dims[1]), etaHa.reshape(dims[0], dims[1])
			                                                  #
			etaPe, etaPa, etaHa=np.rot90(etaPe), np.rot90(etaPa), np.rot90(etaHa)
			                                                  #
			if symmetryplane2D == True:                       #
				                                          #
				etaPe2, etaPa2, etaHa2 = np.flipud(etaPe), np.flipud(etaPa), np.flipud(etaHa)
				                                          #
				etaPe, etaPa, etaHa = np.vstack((etaPe, etaPe2)), np.vstack((etaPa, etaPa2)), np.vstack((etaHa, etaHa2))
			                                                  #
		else:                                                     #
			                                                  #
			etaPe, etaPa, etaHa=etaPe.reshape(dims[0], dims[1], dims[2]), etaPa.reshape(dims[0], dims[1], dims[2]), etaHa.reshape(dims[0], dims[1], dims[2])
			                                                  #
			etaPe, etaPa, etaHa=np.rot90(etaPe), np.rot90(etaPa), np.rot90(etaHa)
		                                                          #
		etaPe, etaPa, etaHa=np.array(etaPe), np.array(etaPa), np.array(etaHa)
		                                                          #
		#---------------------------------------------------------#
		#                     $$ Currents $$                      #
		#---------------------------------------------------------#
		Jx, Jy, Jz = cube['curx'], cube['cury'], cube['curz']     #
		                                                          #
		Jx, Jy, Jz = np.array(Jx), np.array(Jy), np.array(Jz)     #
		                                                          #
		if pf.dimensionality == 1:                                #
			                                                  #
			Jz = Jz.reshape(dims[0])                          #
			                                                  #
			Jx, Jy = np.zeros(Jz.shape), np.zeros(Jz.shape)   #
			                                                  #
		elif pf.dimensionality == 2:                              #
			                                                  #
			Jz = Jz.reshape(dims[0], dims[1])                 #
			                                                  #
			Jz = np.rot90(Jz)                                 #
			                                                  #
			if symmetryplane2D == True:                       #
				                                          #
				Jz2 = np.flipud(Jz)                       #
				                                          #
				Jz = np.vstack((Jz, Jz2))                 #
				                                          #
			Jx, Jy = np.zeros(Jz.shape), np.zeros(Jz.shape)   #
		else:                                                     #
			                                                  #
			Jx, Jy, Jz = Jx.reshape(dims[0], dims[1], dims[2]), Jy.reshape(dims[0], dims[1], dims[2]), Jz.reshape(dims[0], dims[1], dims[2])
			                                                  #
			Jx, Jy, Jz = np.rot90(Jx), np.rot90(Jy), np.rot90(Jz)
		                                                          #
		Jx, Jy, Jz = np.array(Jx), np.array(Jy), np.array(Jz)     #
		                                                          #
		Jx, Jy, Jz = Jx * c/4./np.pi, Jy * c/4./np.pi, Jz * c/4./np.pi
	                                                                  #
	except Exception as e:                                            #
		                                                          #
		etaPe, etaPa, etaHa = 0., 0., 0.                          #
	                                                                  #
except Exception as e:                                                    #
	                                                                  #
	pass                                                              #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                                                                         #
#     Check whether the velocities should be changed in case of a         #
#                     non-ideal MHD simulation                            #
#                                                                         #
if molecule[len(molecule)-1] != "+" or not(np.any(etaPe !=0)):            #
	                                                                  #
	dens = dens/m_p/amu                                               #
	                                                                  #
else:                                                                     #
	                                                                  #
	Btot = np.sqrt(magx**2+magy**2+magz**2)                           #
	                                                                  #
	collT = (1./a_factor_i)*((massesO[mi-1]*m_p+2.*m_p)/dens)*(1./mcr_ions)
	                                                                  #
	dens = dens/m_p/amu                                               #
	                                                                  #
	fact = c*m_p*massesO[mi-1]/(collT*elec)                           #
	                                                                  #
	JparalX, JparalY, JparalZ = (Jx*magx+Jy*magy+Jz*magz)/Btot**2 * magx, (Jx*magx+Jy*magy+Jz*magz)/Btot**2 * magy, (Jx*magx+Jy*magy+Jz*magz)/Btot**2 * magz
	                                                                  #
	JperpX, JperpY, JperpZ = Jx - JparalX, Jy - JparalY, Jz - JparalZ #
	                                                                  #
	d1 = c*etaPe*JperpX + c*etaPa*JparalX + c*etaHa*(Jy*magz/Btot - Jz*magy/Btot)
	                                                                  #
	d2 = c*etaPe*JperpY + c*etaPa*JparalY + c*etaHa*(Jz*magx/Btot - Jx*magz/Btot)
	                                                                  #
	d3 = c*etaPe*JperpZ + c*etaPa*JparalZ + c*etaHa*(Jx*magy/Btot - Jy*magx/Btot)
	                                                                  #
	#-----------------------------------------------------------------#
	#            ***********   Determinants   ***********             #
	#-----------------------------------------------------------------#
	detC = -fact*(fact**2+magx**2) - magz*(fact*magz - magx*magy) - magy*(magz*magx+fact*magy)
	                                                                  #
	detX = d1*(fact**2+magx**2) - magz*(-fact*d2-magx*d3) - magy*(-magx*d2+fact*d3)
	                                                                  #
	detY = -fact*(-fact*d2-magx*d3) - d1*(fact*magz-magx*magy) - magy*(-magz*d3-d2*magy)
	                                                                  #
	detZ = -fact*(-fact*d3+magx*d2) - magz*(-magz*d3-magy*d2) + d1*(magx*magz+fact*magy)
	                                                                  #
	#-----------------------------------------------------------------#
	#              ***********   Velocities   ***********             #
	#-----------------------------------------------------------------#
	vdrX = detX/detC ; vdrY = detY/detC ; vdrZ = detZ/detC            #
	                                                                  #
	velx, vely, velz = velx - vdrX, vely - vdrY, velz- vdrZ           #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#              Taking care of zoom here                                   #
if zoom == True:                                                          #
	                                                                  #
	size = dens.shape                                                 #
	                                                                  #
	ind = int(size[0]* zoomFactor/2)                                  #
	                                                                  #
	if pf.dimensionality == 1:                                        #
		                                                          #
		ind1, ind2 = 0, ind                                       #
		                                                          #
		x = x[ind1:ind2]                                          #
	else:                                                             #
		                                                          #
		ind1, ind2 = ind, -ind                                    #
		                                                          #
	dens, velx, vely, velz, T, mol, x = dens[ind1:ind2], velx[ind1:ind2], vely[ind1:ind2], velz[ind1:ind2], T[ind1:ind2], mol[ind1:ind2]
	                                                                  #
	try:                                                              #
		                                                          #
		magx, magy, magz = magx[ind1:ind2], magy[ind1:ind2], magz[ind1:ind2]
		                                                          #
		try:                                                      #
			                                                  #
			etaPe, etaPa, etaHa = etaPe[ind1:ind2], etaPa[ind1:ind2], etaHa[ind1:ind2]
			                                                  #
			Jx, Jy, Jz = Jx[ind1:ind2], Jy[ind1:ind2], Jz[ind1:ind2]
			                                                  #
		except NameError:                                         #
			                                                  #
			pass                                              #
			                                                  #
	except NameError:                                                 #
			                                                  #
		pass                                                      #
		                                                          #
	if pf.dimensionality >= 2:                                        #
		                                                          #
		y = y[ind1:ind2]                                          #
		                                                          #
		ind = int(size[1]* zoomFactor/2)                          #
		                                                          #
		if pf.dimensionality == 2:                                #
			                                                  #
			ind1, ind2 = 0, ind                               #
			                                                  #
		else:                                                     #
			                                                  #
			ind1, ind2 = ind, -ind                            #
			                                                  #
		dens, T, mol, x  = dens[:, ind1:ind2], T[:, ind1:ind2], mol[:, ind1:ind2], x[ind1:ind2]
		                                                          #
		velx, vely, velz = velx[:, ind1:ind2], vely[:, ind1:ind2], velz[:, ind1:ind2]
		                                                          #
		try:                                                      #
			                                                  #
			magx, magy, magz = magx[:, ind1:ind2], magy[:, ind1:ind2], magz[:, ind1:ind2]
			                                                  #
			try:                                              #
				                                          #
				etaPe, etaPa, etaHa = etaPe[:, ind1:ind2], etaPa[:, ind1:ind2], etaHa[:, ind1:ind2]
				                                          #
				Jx, Jy, Jz = Jx[:, ind1:ind2], Jy[:, ind1:ind2], Jz[:, ind1:ind2]
				                                          #
			except NameError:                                 #
				                                          #
				pass                                      #
				                                          #
		except NameError:                                         #
			pass                                              #
		                                                          #
	if pf.dimensionality == 3:                                        #
		                                                          #
		ind = int(size[2]* zoomFactor/2)                          #
		                                                          #
		dens, T, mol, z = dens[:, :, ind:-ind], T[:, :, ind:-ind], mol[:, :, ind:-ind], z[ind:-ind]
		                                                          #
		velx, vely, velz = velx[:, :, ind:-ind], vely[:,:,ind:-ind], velz[:, :, ind:-ind]
		                                                          #
		try:                                                      #
			                                                  #
			magx, magy, magz = magx[:, :, ind:-ind], magy[:, :, ind:-ind], magz[:, :, ind:-ind]
			                                                  #
			try:                                              #
				                                          #
				etaPe, etaPa, etaHa = etaPe[:, :, ind:-ind], etaPa[:, :, ind:-ind], etaHa[:, :, ind:-ind]
				                                          #
				Jx, Jy, Jz = Jx[:, :, ind:-ind], Jy[:, :, ind:-ind], Jz[:, :, ind:-ind]
				                                          #
			except NameError:                                 #
				                                          #
				pass                                      #
				                                          #
		except NameError:                                         #
			                                                  #
			pass                                              #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
                                                                          #
import matplotlib.pyplot as plt                                           #
import matplotlib.ticker as ticker                                        #
from mpl_toolkits.axes_grid1 import make_axes_locatable                   #
import matplotlib                                                         #
from matplotlib import ticker                                             #
tick_locator = ticker.MaxNLocator(nbins=4)                                #
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter         #
from pylab import *                                                       #
from matplotlib.ticker import MultipleLocator                             #
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid        #
import matplotlib.gridspec as gridspec                                    #
plt.rcParams.update({'font.size': 15})                                    #
matplotlib.rcParams['pdf.fonttype'] = 42                                  #
plt.rcParams['ps.fonttype'] = 42                                          #
                                                                          #
if pf.dimensionality == 1:                                                #
	                                                                  #
	if (x.shape!=mol.shape) or (x.shape!=velx.shape) or (x.shape!=T.shape) or (x.shape!=dens.shape): 
		                                                          #
		print("\n")                                               #
		print(" "*35+"The shape of matrices differs!")            #
		print(" "*35+"Density length is {}".format(dens.shape))   #
		print(" "*35+"Molecule length is {}".format(mol.shape))   #
		print(" "*35+"Velocity length is {}".format(velx.shape))  #
		print(" "*35+"Temperature length is {}".format(T.shape))  #
		print(" "*35+"radius length is {}".format(x.shape))       #
	                                                                  #
	else:                                                             #
		print("\n")                                               #
		print(" "*35+"All arrays have the same size")             #
	                                                                  #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	#               Now check for NaN values                          #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	Nans=np.array([np.isnan(dens).any(), np.isnan(mol).any(), np.isnan(velx).any(), np.isnan(T).any()])
	                                                                  #
	if np.isnan(Nans).any()==True:                                    #
		                                                          #
		print("\n")                                               #
		print(" "*35+"Found NaN values in arrays!")               #
		                                                          #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	#       Now make a multipanel plot and return so that             #
	#         the user can check if something is wrong                #
	#                   with the input data                           #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	fig=plt.figure(figsize=(14, 10))                                  #
	ax1=fig.add_subplot(221)   #top left                              #
	ax2=fig.add_subplot(222)   #top right                             #
	ax3=fig.add_subplot(223)   #bottom left                           #
	ax4=fig.add_subplot(224)   #bottom right                          #
	#- - -  - - - - - - - - - - - - - - - - - - - - - - - - - - -     #
	ax1.semilogy(x/parsec, dens, 'k', linewidth=2.0)                  #
	ax1.set_title('Density')                                          #
	ax1.set_xlabel('pc', fontsize=22)                                 #
	ax1.set_ylabel(r"$\mathregular{n_{H_2}}$" " "r"$\mathregular{(cm^{-3})}$", fontsize=22)
	ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))      #
	start, end = ax1.get_xlim()                                       #
	ax1.xaxis.set_ticks(np.linspace(start, end, 3))                   #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	ax2.plot(x/parsec, velx*1e-5, 'k', linewidth=2.0)                 #
	ax2.set_title('Velocity')                                         #
	ax2.set_xlabel('pc', fontsize=22)                                 #
	ax2.set_ylabel('km ' r"$\mathregular{s^{-1}}$", fontsize=22)      #
	ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))      #
	start, end = ax2.get_xlim()                                       #
	ax2.xaxis.set_ticks(np.linspace(start, end, 3))                   #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	ax3.plot(x/parsec, mol, 'k', linewidth=2.0)                       #
	ax3.set_title('Molecular Number Density')                         #
	ax3.set_ylabel(r"$\mathregular{n_{mol}/n_{H_2}}$", fontsize=22)   #
	ax3.set_xlabel('pc', fontsize=22)                                 #
	ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))      #
	start, end = ax3.get_xlim()                                       #
	ax3.xaxis.set_ticks(np.linspace(start, end, 3))                   #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	ax4.plot(x/parsec, T, 'k', linewidth=2.0)                         #
	ax4.set_title('Temperature')                                      #
	ax4.set_xlabel('pc', fontsize=22)                                 #
	ax4.set_ylabel('K', fontsize=22)                                  #
	start, end = ax4.get_xlim()                                       #
	ax4.xaxis.set_ticks(np.linspace(start, end, 3))                   #
	fig.subplots_adjust(top = 0.95, bottom = 0.08, right = 0.985, left = 0.05, wspace = 0.25, hspace = 0.30)
	plt.show()                                                        #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                             Cylindrical Part                            #
#               First check that everything has the same dimensions       #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
elif pf.dimensionality == 2:                                              #
	                                                                  #
	if (dens.shape!=mol.shape) or (dens.shape!=velx.shape) or (dens.shape!=T.shape) or (dens.shape!=vely.shape):
		                                                          #
		print("\n")                                               #
		print(" "*35+"The shape of matrices differs!")            #
		print(" "*35+"Density length is {}".format(dens.shape))   #
		print(" "*35+"Molecule length is {}".format(mol.shape))   #
		print(" "*35+"Velocity length is {}".format(velx.shape))  #
		print(" "*35+"Temperature length is {}".format(T.shape))  #
		print(" "*35+"radius length is {}".format(x.shape))       #
		                                                          #
	else:                                                             #
		print("\n")                                               #
		print(" "*35+"All arrays have the same size")             #
	                                                                  #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	#                    Now check for NaN values                     #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	Nans=np.array([np.isnan(dens).any(), np.isnan(mol).any(), np.isnan(velx).any(), np.isnan(T).any(), np.isnan(vely).any()])
	                                                                  #
	if np.isnan(Nans).any()==True:                                    #
		                                                          #
		print("\n")                                               #
		print(" "*35+"Found NaN values in arrays!")               #
		                                                          #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	#       Now make a multipanel plot and return so that             #
	#         the user can check if something is wrong                #
	#                   with the input data                           #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	fig=plt.figure(figsize=(14, 10))                                  #
	ax1=fig.add_subplot(221)   #top left                              #
	ax2=fig.add_subplot(222)   #top right                             #
	ax3=fig.add_subplot(223)   #bottom left                           #
	ax4=fig.add_subplot(224)   #bottom right                          #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im1=ax1.imshow(np.log10(dens), cmap='hot')                        #
	divider1 = make_axes_locatable(ax1)                               #
	cax1 = divider1.append_axes("right", size="5%", pad=0.0)          #
	cbar1 = plt.colorbar(im1, cax=cax1)                               #
	ax1.set_title('Density')                                          #
	ax1.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])                #
	ax1.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax1.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
	ax1.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])                           #
	ax1.set_xlabel('r [pc]', fontsize = 22)                           #
	ax1.set_ylabel('z [pc]', fontsize = 22)                           #
	ax1.set_ylim(0, len(dens)-1)                                      #
	ax1.set_xlim(0, len(dens[0])-1)                                   #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im2=ax2.imshow(velx*1e-5, cmap='hot')                             #
	divider2 = make_axes_locatable(ax2)                               #
	cax2 = divider2.append_axes("right", size="5%", pad=0.0)          #
	cbar2 = plt.colorbar(im2, cax=cax2)                               #
	ax2.set_title('Velocity-r')                                       #
	ax2.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])                #
	ax2.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax2.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
	ax2.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])                           #
	ax2.set_xlabel('r [pc]', fontsize = 22)                           #
	ax2.set_ylabel('z [pc]', fontsize = 22)                           #
	ax2.set_ylim(0, len(dens)-1)                                      #
	ax2.set_xlim(0, len(dens[0])-1)                                   #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im3=ax3.imshow(mol, cmap='hot')                                   #
	divider3 = make_axes_locatable(ax3)                               #
	cax3 = divider3.append_axes("right", size="5%", pad=0.0)          #
	cbar3 = plt.colorbar(im3, cax=cax3)                               #
	ax3.set_title('Molecular Number Density')                         #
	ax3.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])                #
	ax3.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax3.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
	ax3.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])                           #
	ax3.set_xlabel('r [pc]', fontsize = 22)                           #
	ax3.set_ylabel('z [pc]', fontsize = 22)                           #
	ax3.set_ylim(0, len(dens)-1)                                      #
	ax3.set_xlim(0, len(dens[0])-1)                                   #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im4=ax4.imshow(vely*1e-5, cmap='hot')                             #
	divider4 = make_axes_locatable(ax4)                               #
	cax4 = divider4.append_axes("right", size="5%", pad=0.0)          #
	cbar4 = plt.colorbar(im4, cax=cax4)                               #
	ax4.set_title('Velocity-z')                                       #
	ax4.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])                #
	ax4.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax4.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
	ax4.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])                           #
	ax4.set_xlabel('r [pc]', fontsize = 22)                           #
	ax4.set_ylabel('z [pc]', fontsize = 22)                           #
	ax4.set_ylim(0, len(dens)-1)                                      #
	ax4.set_xlim(0, len(dens[0])-1)                                   #
	fig.subplots_adjust(top = 0.95, bottom = 0.08, right = 0.985, left = 0.05, wspace = 0.25, hspace = 0.30)
	plt.show()                                                        #
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
	try:                                                              #
		fig=plt.figure(figsize=(14, 10))                          #
		ax1=fig.add_subplot(221)   #top left                      #
		ax2=fig.add_subplot(222)   #top right                     #
		ax3=fig.add_subplot(223)   #bottom left                   #
		ax4=fig.add_subplot(224)   #bottom right                  #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im1=ax1.imshow(magx, cmap='hot')                          #
		divider1 = make_axes_locatable(ax1)                       #
		cax1 = divider1.append_axes("right", size="5%", pad=0.0)  #
		cbar1 = plt.colorbar(im1, cax=cax1)                       #
		ax1.set_title('Bx')                                       #
		ax1.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])        #
		ax1.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax1.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
		ax1.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])  #
		ax1.set_xlabel('r [pc]', fontsize = 22)                   #
		ax1.set_ylabel('z [pc]', fontsize = 22)                   #
		ax1.set_ylim(0, len(dens)-1)                              #
		ax1.set_xlim(0, len(dens[0])-1)                           #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im2=ax2.imshow(magy, cmap='hot')                          #
		divider2 = make_axes_locatable(ax2)                       #
		cax2 = divider2.append_axes("right", size="5%", pad=0.0)  #
		cbar2 = plt.colorbar(im2, cax=cax2)                       #
		ax2.set_title('By')                                       #
		ax2.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])        #
		ax2.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax2.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
		ax2.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])  #
		ax2.set_xlabel('r [pc]', fontsize = 22)                   #
		ax2.set_ylabel('z [pc]', fontsize = 22)                   #
		ax2.set_ylim(0, len(dens)-1)                              #
		ax2.set_xlim(0, len(dens[0])-1)                           #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im3=ax3.imshow(Jz, cmap='hot')                            #
		divider3 = make_axes_locatable(ax3)                       #
		cax3 = divider3.append_axes("right", size="5%", pad=0.0)  #
		cbar3 = plt.colorbar(im3, cax=cax3)                       #
		ax3.set_title('Jz')                                       #
		ax3.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])        #
		ax3.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax3.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
		ax3.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])  #
		ax3.set_xlabel('r [pc]', fontsize = 22)                   #
		ax3.set_ylabel('z [pc]', fontsize = 22)                   #
		ax3.set_ylim(0, len(dens)-1)                              #
		ax3.set_xlim(0, len(dens[0])-1)                           #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im4=ax4.imshow(etaPe, cmap='hot')                         #
		divider4 = make_axes_locatable(ax4)                       #
		cax4 = divider4.append_axes("right", size="5%", pad=0.0)  #
		cbar4 = plt.colorbar(im4, cax=cax4)                       #
		ax4.set_title(r'$\mathregular{\eta_\perp}$')              #
		ax4.set_xticks([len(x)*0, len(x)*2/4.0, len(x)-1])        #
		ax4.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax4.set_xticklabels([round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*2/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])
		ax4.set_yticklabels([round(y[int(len(y)*0)]/parsec, 2), round(y[int(len(y)*1/4)]/parsec, 2), round(y[int(len(x)*2/4)]/parsec, 0), round(y[int(len(y)*3/4)]/parsec, 2), round(y[int(len(y)-1)]/parsec, 2)])  #
		ax4.set_xlabel('r [pc]', fontsize = 22)                   #
		ax4.set_ylabel('z [pc]', fontsize = 22)                   #
		ax4.set_ylim(0, len(dens)-1)                              #
		ax4.set_xlim(0, len(dens[0])-1)                           #
		fig.subplots_adjust(top = 0.95, bottom = 0.08, right = 0.985, left = 0.05, wspace = 0.25, hspace = 0.30)
		plt.show()                                                #
	except Exception as e:                                            #
		pass                                                      #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                             Cartesian Part                              #
#               First check that everything has the same dimensions       #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
else:                                                                     #
	                                                                  #
	if (dens.shape!=mol.shape) or (dens.shape!=velx.shape) or (dens.shape!=T.shape) or (dens.shape!=vely.shape) or (dens.shape!=velz.shape):
		                                                          #
		print("\n")                                               #
		print(" "*35+"The shape of matrices differs!")            #
		print(" "*35+"Density length is {}".format(dens.shape))   #
		print(" "*35+"Molecule length is {}".format(mol.shape))   #
		print(" "*35+"Velocity length is {}".format(velx.shape))  #
		print(" "*35+"Temperature length is {}".format(T.shape))  #
		print(" "*35+"radius length is {}".format(x.shape))       #
		                                                          #
	else:                                                             #
		print("\n")                                               #
		print(" "*35+"All arrays have the same size")             #
	                                                                  #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	#                    Now check for NaN values                     #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	Nans=np.array([np.isnan(dens).any(), np.isnan(mol).any(), np.isnan(velx).any(), np.isnan(T).any(), np.isnan(vely).any(), np.isnan(velz).any()])
	                                                                  #
	if np.isnan(Nans).any()==True:                                    #
		                                                          #
		print("\n")                                               #
		print(" "*35+"Found NaN values in arrays!")               #
		                                                          #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	#       Now make a multipanel plot and return so that             #
	#         the user can check if something is wrong                #
	#                   with the input data                           #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	fig=plt.figure(figsize=(14, 10))                                  #
	ax1=fig.add_subplot(221)   #top left                              #
	ax2=fig.add_subplot(222)   #top right                             #
	ax3=fig.add_subplot(223)   #bottom left                           #
	ax4=fig.add_subplot(224)   #bottom right                          #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im1=ax1.imshow(np.log10(dens[:, :, dens.shape[2]//2]), cmap='hot')#
	divider1 = make_axes_locatable(ax1)                               #
	cax1 = divider1.append_axes("right", size="5%", pad=0.0)          #
	cbar1 = plt.colorbar(im1, cax=cax1)                               #
	ax1.set_title('Density')                                          #
	ax1.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
	ax1.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax1.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax1.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax1.set_xlabel('r [pc]', fontsize = 22)                           #
	ax1.set_ylabel('z [pc]', fontsize = 22)                           #
	ax1.set_ylim(0, len(dens)-1)                                      #
	ax1.set_xlim(0, len(dens)-1)                                      #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im2=ax2.imshow(velx[:, :, velx.shape[2]//2]*1e-5, cmap='hot')     #
	divider2 = make_axes_locatable(ax2)                               #
	cax2 = divider2.append_axes("right", size="5%", pad=0.0)          #
	cbar2 = plt.colorbar(im2, cax=cax2)                               #
	ax2.set_title('Velocity-r')                                       #
	ax2.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
	ax2.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax2.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax2.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax2.set_xlabel('r [pc]', fontsize = 22)                           #
	ax2.set_ylabel('z [pc]', fontsize = 22)                           #
	ax2.set_ylim(0, len(dens)-1)                                      #
	ax2.set_xlim(0, len(dens)-1)                                      #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im3=ax3.imshow(mol[:, :, mol.shape[2]//2], cmap='hot')            #
	divider3 = make_axes_locatable(ax3)                               #
	cax3 = divider3.append_axes("right", size="5%", pad=0.0)          #
	cbar3 = plt.colorbar(im3, cax=cax3)                               #
	ax3.set_title('Molecular Number Density')                         #
	ax3.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
	ax3.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax3.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax3.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax3.set_xlabel('r [pc]', fontsize = 22)                           #
	ax3.set_ylabel('z [pc]', fontsize = 22)                           #
	ax3.set_ylim(0, len(dens)-1)                                      #
	ax3.set_xlim(0, len(dens)-1)                                      #
	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
	im4=ax4.imshow(vely[:, :, vely.shape[2]//2]*1e-5, cmap='hot')     #
	divider4 = make_axes_locatable(ax4)                               #
	cax4 = divider4.append_axes("right", size="5%", pad=0.0)          #
	cbar4 = plt.colorbar(im4, cax=cax4)                               #
	ax4.set_title('Velocity-z')                                       #
	ax4.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
	ax4.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
	ax4.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax4.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])                           #
	ax4.set_xlabel('r [pc]', fontsize = 22)                           #
	ax4.set_ylabel('z [pc]', fontsize = 22)                           #
	ax4.set_ylim(0, len(dens)-1)                                      #
	ax4.set_xlim(0, len(dens)-1)                                      #
	fig.subplots_adjust(top = 0.95, bottom = 0.08, right = 0.985, left = 0.05, wspace = 0.25, hspace = 0.30)
	plt.show()                                                        #
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
	try:                                                              #
		fig=plt.figure(figsize=(14, 10))                          #
		ax1=fig.add_subplot(221)   #top left                      #
		ax2=fig.add_subplot(222)   #top right                     #
		ax3=fig.add_subplot(223)   #bottom left                   #
		ax4=fig.add_subplot(224)   #bottom right                  #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im1=ax1.imshow(magx[:, :, magx.shape[2]//2], cmap='hot')  #
		divider1 = make_axes_locatable(ax1)                       #
		cax1 = divider1.append_axes("right", size="5%", pad=0.0)  #
		cbar1 = plt.colorbar(im1, cax=cax1)                       #
		ax1.set_title('Bx')                                       #
		ax1.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
		ax1.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax1.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax1.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax1.set_xlabel('r [pc]', fontsize = 22)                   #
		ax1.set_ylabel('z [pc]', fontsize = 22)                   #
		ax1.set_ylim(0, len(dens)-1)                              #
		ax1.set_xlim(0, len(dens)-1)                              #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im2=ax2.imshow(magy[:, :, magy.shape[2]//2], cmap='hot')  #
		divider2 = make_axes_locatable(ax2)                       #
		cax2 = divider2.append_axes("right", size="5%", pad=0.0)  #
		cbar2 = plt.colorbar(im2, cax=cax2)                       #
		ax2.set_title('By')                                       #
		ax2.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
		ax2.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax2.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax2.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax2.set_xlabel('r [pc]', fontsize = 22)                   #
		ax2.set_ylabel('z [pc]', fontsize = 22)                   #
		ax2.set_ylim(0, len(dens)-1)                              #
		ax2.set_xlim(0, len(dens)-1)                              #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im3=ax3.imshow(Jz[:, :, Jz.shape[2]//2], cmap='hot')      #
		divider3 = make_axes_locatable(ax3)                       #
		cax3 = divider3.append_axes("right", size="5%", pad=0.0)  #
		cbar3 = plt.colorbar(im3, cax=cax3)                       #
		ax3.set_title('Jz')                                       #
		ax3.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
		ax3.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax3.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax3.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax3.set_xlabel('r [pc]', fontsize = 22)                   #
		ax3.set_ylabel('z [pc]', fontsize = 22)                   #
		ax3.set_ylim(0, len(dens)-1)                              #
		ax3.set_xlim(0, len(dens)-1)                              #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
		im4=ax4.imshow(etaPe[:, :, etaPe.shape[2]//2], cmap='hot')#
		divider4 = make_axes_locatable(ax4)                       #
		cax4 = divider4.append_axes("right", size="5%", pad=0.0)  #
		cbar4 = plt.colorbar(im4, cax=cax4)                       #
		ax4.set_title(r'$\mathregular{\eta_\perp}$')              #
		ax4.set_xticks([len(x)*0, len(x)*1/4.0, len(x)*2/4.0, len(x)*3/4.0, len(x)-1])
		ax4.set_yticks([len(y)*0, len(y)*1/4.0, len(y)*2/4.0, len(y)*3/4.0, len(y)-1])
		ax4.set_xticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax4.set_yticklabels([round(x[int(len(x)*0)]/parsec, 2), round(x[int(len(x)*1/4)]/parsec, 2), round(x[int(len(x)*2/4)]/parsec, 0), round(x[int(len(x)*3/4)]/parsec, 2), round(x[int(len(x)-1)]/parsec, 2)])  #
		ax4.set_xlabel('r [pc]', fontsize = 22)                   #
		ax4.set_ylabel('z [pc]', fontsize = 22)                   #
		ax4.set_ylim(0, len(dens)-1)                              #
		ax4.set_xlim(0, len(dens)-1)                              #
		fig.subplots_adjust(top = 0.95, bottom = 0.08, right = 0.985, left = 0.05, wspace = 0.25, hspace = 0.30)
		plt.show()                                                #
	except Exception as e:                                            #
		pass                                                      #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#                 $$  Load coefficient data  $$                           #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
                                                                          #
mole = molecule.lower() + ".dat"                                          #
                                                                          #
path = cwd + '/lambda/'                                                   #
                                                                          #
if not (os.path.exists(path+mole)):                                       #
	                                                                  #
	raise SystemExit("Cannot find file: ", path+mole)                 #
                                                                          #
f=open(path+mole)                                                         #
                                                                          #
txt=[]                                                                    #
                                                                          #
for line in iter(f):                                                      #
	                                                                  #
	txt.append(line)                                                  #
	                                                                  #
f.close()                                                                 #
                                                                          #
Cul_p, Cul_o, EinA, TranFreq, Ener, gul, EnerL = [], [], [], [], [], [], []
                                                                          #
para = 0                                                                  #
                                                                          #
for i in range(0, len(txt)):                                              #
	                                                                  #
	if (txt[i][0:6]== "!LEVEL"):                                      #
		                                                          #
		m, LessThanMaxLevel = i+1, True                           #
		                                                          #
		while LessThanMaxLevel:                                   #
			                                                  #
			temp = np.fromstring(txt[m], dtype='float', sep=' ')
			                                                  #
			EnerL.append(temp[0]); Ener.append(temp[1]) ; gul.append(temp[2])
			                                                  #
			m = m+1                                           #
			                                                  #
			if temp[0] == nlevels:                            #
				                                          #
				LessThanMaxLevel = False                  #
	                                                                  #
	if (txt[i][0:29] == "!TRANS + UP + LOW + EINSTEINA"):             #
		                                                          #
		m, LessThanMaxLevel = i+1, True                           #
		                                                          #
		while LessThanMaxLevel:                                   #
			                                                  #
			temp = np.fromstring(txt[m], dtype='float', sep=' ')
			                                                  #
			EinA.append(temp[3]); TranFreq.append(temp[4])    #
			                                                  #
			m = m+1                                           #
			                                                  #
			if temp[1] == nlevels:                            #
				                                          #
				LessThanMaxLevel = False                  #
	                                                                  #
	if (txt[i][0:29] == "!TRANS + UP + LOW + COLLRATES") and para == 0:
		                                                          #
		m, LessThanMaxLevel, para = i+1, True, i                  #
		                                                          #
		tempersP = np.fromstring(txt[i -1 ], dtype='float', sep=' ')
		                                                          #
		while LessThanMaxLevel:                                   #
			                                                  #
			temp = np.fromstring(txt[m], dtype='float', sep=' ')
			                                                  #
			Cul_p.append(temp)                                #
			                                                  #
			m = m+1                                           #
			                                                  #
			if temp[1] > nlevels:                             #
				                                          #
				LessThanMaxLevel = False                  #
	                                                                  #
	if (txt[i][0:29] == "!TRANS + UP + LOW + COLLRATES") and i!=para: #
		                                                          #
		m, LessThanMaxLevel = i+1, True                           #
		                                                          #
		tempersO = np.fromstring(txt[i -1 ], dtype='float', sep=' ')
		                                                          #
		while LessThanMaxLevel:                                   #
			                                                  #
			temp = np.fromstring(txt[m], dtype='float', sep=' ')
			                                                  #
			Cul_o.append(temp)                                #
			                                                  #
			m = m+1                                           #
			                                                  #
			if temp[1] > nlevels:                             #
				                                          #
				LessThanMaxLevel = False                  #
	                                                                  #
Cul_p, Cul_o, EinA, TranFreq, Ener, gul, EnerL = np.array(Cul_p), np.array(Cul_o), np.array(EinA), np.array(TranFreq), np.array(Ener), np.array(gul), np.array(EnerL)
                                                                          #
if not np.all(Cul_o):                                                     #
	                                                                  #
	Cul = Cul_p.copy()                                                #
	                                                                  #
else:                                                                     #
	                                                                  #
	try:                                                              #
		                                                          #
		if np.all(tempersP == tempersO):                          #
			                                                  #
			Cul = 0.7*Cul_p + 0.3*Cul_o                       #
			                                                  #
		else:                                                     #
			                                                  #
			print("\n")                                       #
			                                                  #
			raise SystemExit("Temperature mismatch between collisional partners")
	                                                                  #
	except NameError:                                                 #
		                                                          #
		Cul = Cul_p                                               #
                                                                          #
Cul, TranFreq = Cul[:-1], TranFreq*1e+9                                   #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#             $$  Calculate Einstein B & Clu $$                           #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
if GK:                                                                    #
	#         !! This requires extra care  !!                         #
	                                                                  #
	GKgul = []                                                        #
	                                                                  #
	for j in range (0, nlevels):                                      #
		                                                          #
		for m in range (0, j+1):                                  #
			                                                  #
			if m == 0:                                        #
				                                          #
				GKgul.append(1.)                          #
				                                          #
			else:                                             #
				                                          #
				GKgul.append(2.)                          #
				                                          #
	GKgul = np.array(GKgul)                                           #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#     !! Deal with Einstein & collisional coefficients !!         #
	#                                                                 #
	#        Deguchi & Watson quote Townes & Schawlow for:            #
	#  Ajm->j'm' = Ajj' * [(j+1)**2 -m**2]/[(2j+1)(j+1)] for Dm = 0   # #((j+1)**2 -m**2) / ((2.*j+1.)*(j+1.)) )
	# Ajm->j'm' = Ajj' * [(j-|m|+1)(j-|m|)]/[2*(2j+1)(j+1)] for Dm = 0# #((j+m+1)*(j+m))/(2.*(2.*j+1.)*(j+1.)))
	#          Here jj = lower level, j = upper level                 #
	#                                                                 #
	#   Corrent relations for up to level J = 3 are given in Table    #
	#                      4.2 in following:                          #
	# https://refubium.fu-berlin.de/bitstream/handle/fub188/7773/06_chapter4.pdf?sequence=7&isAllowed=y
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	# $$$ However even these relations break down for some reason $$$ #
	# $$$   for j > 3. More correctly we multiply here with the   $$$ #
	# $$$               the Winger 3j symbol                      $$$ #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	GKEinA, jmjpmp, GKCul, GKTranFreq, jmjpmpCul = [], [], [], [], [] #
	                                                                  #
	for jj in range (0, nlevels-1):                                   #
		                                                          #
		for mm in range (0, jj+1):                                #
			                                                  #
			j = jj+1                                          #
			                                                  #
			for m in range (0, j+1):                          #
				                                          #
				if np.absolute(m - mm) == 0:              #
					                                  #
					GKTranFreq.append(TranFreq[jj])   #
					                                  #
					#GKEinA.append( gul[j] * EinA[jj] * (j**2-m**2)/(j*(2.*j - 1.) * (2.*j + 1.)) )
					GKEinA.append( gul[j] * EinA[jj] * float(wigner_3j(j, 1, jj, -m, (m-mm), mm))**2 )
					                                  #
					jmjpmp.append([j, m, jj, mm])     #
					                                  #
				elif np.absolute(m - mm) == 1:            #
					                                  #
					GKTranFreq.append(TranFreq[jj])   #
					                                  #
					#GKEinA.append( gul[j] * EinA[jj] * (j*(j-1)+(2*j-1)*m+m**2)/(2*j*(2*j-1)*(2*j+1)))
					GKEinA.append( gul[j] * EinA[jj] * float(wigner_3j(j, 1, jj, -m, (m-mm), mm))**2 )
					                                  #
					jmjpmp.append([j, m, jj, mm])     #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	# $$$ No need to multiply here with Winger 3j symbol. However $$$ #
	# $$$   we need to divide with (2.*Jlower+1) here to ensure   $$$ #
	# $$$ that the way the Statistical Equilibrium Equations are  $$$ #
	# $$$   written, we lose/receive from/to each energy level    $$$ #
	# $$$    the same amount as in the case where GK = False      $$$ #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	for i in range (0, len(Cul)):                                     #
		                                                          #
		ju, jl = int(round(Cul[i, 1]))-1, int(round(Cul[i, 2]))-1 #
		                                                          #
		for mu in range (0, ju+1):                                #
			                                                  #
			for ml in range (0, jl+1):                        #
				                                          #
				jmjpmpCul.append([ju, mu, jl, ml])        #
				                                          #
				DJ = ju - jl                              #
				                                          #
				if ml == 0:                               #
					# gul[ju] * (ju**2-mu**2)/(ju*(2.*ju - 1.) * (2.*ju + 1.)) ))) # float(wigner_3j(ju, 1, jl, -mu, (mu-ml), ml)**2) )))
					GKCul.append(np.concatenate((Cul[i, :3], Cul[i, 3:] / (2.*jl+1) )))
				else:                                     #
					#gul[ju] * (ju*(ju-1)+(2*ju-1)*mu+mu**2)/(2*ju*(2*ju-1)*(2*ju+1)) ))) # float(wigner_3j(ju, 1, jl, -mu, (mu-ml), ml)**2) )))
					GKCul.append(np.concatenate((Cul[i, :3], Cul[i, 3:] / (2.*jl+1) )))
	                                                                  #
	gul[:] = 1.                                                       #
	                                                                  #
	GKEinA, GKCul, jmjpmp, GKTranFreq, jmjpmpCul = np.array(GKEinA), np.array(GKCul), np.array(jmjpmp), np.array(GKTranFreq), np.array(jmjpmpCul)
	                                                                  #
	# $$ Rename GK coefficients for consistency with rest of code $$  #
	EinA, Cul, gul, TranFreq = GKEinA, GKCul, GKgul, GKTranFreq       #
                                                                          #
#           Following work for both GK = True and GK = False              #
EinBul = EinA*c**2/(2*h*TranFreq**3)                                      #
                                                                          #
for i in range (0, len(Ener)):                                            #
	                                                                  #
	Ener[i] = Ener[i] * c * h                                         #
                                                                          #
Clu, EinBlu = [], []                                                      #
                                                                          #
if GK == False:                                                           #
	                                                                  #
	for i in range (0, len(EinBul)):                                  #
		                                                          #
		EinBlu.append(gul[i+1]/gul[i] * EinBul[i])                #
		                                                          #
	for i in range (0, len(Cul)):                                     #
		                                                          #
		Clu.append(np.concatenate((Cul[i, :3], gul[int(Cul[i][1])-1]/gul[int(Cul[i][2])-1] * Cul[i, 3:])))
		                                                          #
else:                                                                     #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#   $$$    Statistical weights not included here. Makes    $$$    #
	#   $$$    more sense to me to include these explicitly    $$$    #
	#   $$$         in detailed balance equations              $$$    #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	for i in range (0, len(EinBul)):                                  #
		                                                          #
		EinBlu.append(EinBul[i])                                  #
		                                                          #
	for i in range (0, len(Cul)):                                     #
		                                                          #
		Clu.append(np.concatenate((Cul[i, :3], Cul[i, 3:])))      #
                                                                          #
EinBlu, Clu = np.array(EinBlu), np.array(Clu)                             #
                                                                          #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#          $$  Create txt file with detailed balance Eqs $$               #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
detailedBalEqs = open('detailedBalEqs','w+')                              #
                                                                          #
#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-=#
#                                                                         #
#  $$ Cleanest way of tracking detailed balance is to completely $$       #
#   $$ distinguish when no GK effect. Some code duplication is $$         #
#                 $$ unavoidable but it's ok! $$                          #
#                                                                         #
#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-=#
if GK == False:                                                           #
	                                                                  #
	newline = ""                                                      #
	                                                                  #
	for i in range (nlevels):                                         #
		                                                          #
		newline = newline + "n{}, ".format(i)                     #
		                                                          #
		                                                          #
	newline = newline[:-2] + " = pd" + "\n"                           #
	                                                                  #
	detailedBalEqs.write(newline)                                     #
	#-----------------------------------------------------------------#
	newline = "eq1 = "                                                #
	                                                                  #
	for i in range (nlevels):                                         #
		                                                          #
		newline = newline + "n{} + ".format(i)                    #
		                                                          #
	newline = newline[:-2] + "-molgp" + "\n"                          #
	                                                                  #
	detailedBalEqs.write(newline)                                     #
	#-----------------------------------------------------------------#
	                                                                  #
	for j in range (nlevels-1):                                       #
		                                                          #
		newline = "eq{} = densgp * (".format(j+2)                 # open nH2 parenthesis
		#                                                         #
		#   $$$  Collisional "Gainings" from upper levels  $$$    #
		#                                                         #
		for i in range (len(Cul)):                                #
			                                                  #
			if int(round(Cul[i, 2])) == j+1:                  #
				                                          #
				newline = newline + "Cul[{}, tmin] * n{} + ".format(i, int(round(Cul[i, 1])-1 ) )
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		#                                                         #
		#     $$$  Collisional "Losses" to upper levels  $$$      #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		newline = newline[:-2] + "- n{} * (".format(j)            # open nj paranthesis
		                                                          #
		for i in range (len(Clu)):                                #
			                                                  #
			if int(round(Clu[i, 2])) == j+1:                  #
				                                          #
				#    CexpF = np.exp(-h* DE * c/(Kb*Tgp))  #
				newline = newline + "Clu[{}, tmin] * CexpF[{}] + ".format(i, i)
			                                                  #
		newline = newline[:-2] + " + "                            #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-#
		#                                                         #
		#     $$$  Collisional "Losses" to lower levels  $$$      #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		for i in range (len(Cul)):                                #
			                                                  #
			if int(round(Cul[i, 1])) == j+1:                  #
				                                          #
				newline = newline + "Cul[{}, tmin] + ".format(i)
			                                                  #
		newline = newline[:-2] + ") + "                           # close nj parenthesis
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-#
		#                                                         #
		#    $$$  Collisional "Gainings" from lower levels  $$$   #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		for i in range (len(Clu)):                                #
			                                                  #
			if int(round(Clu[i, 1])) == j+1:                  #
				                                          #
				newline = newline + "Clu[{}, tmin] * n{} * CexpF[{}] + ".format(i, int(round(Clu[i, 2])-1 ), i)
			                                                  #
		newline = newline[:-2] + ") + "                           # close nH2 parenthesis
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-#
		#                                                         #
		#             $$$  Spontaneous "Gainings"  $$$            #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		newline = newline + "EinA[{}] * n{}".format(j, j+1)       #
		                                                          #
		#--------------------------------------------------- -----#
		#                                                         #
		#             $$$  Spontaneous "Losses"  $$$              #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		if j > 0:                                                 #
			                                                  #
			newline = newline + "- EinA[{}] * n{}".format(j-1, j)
			                                                  #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		#                                                         #
		#     Stimulated "Gainings/Losses" from/to higher level   #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		if j < nlevels-1:                                         #
			                                                  #
			newline = newline+"+ (EinBul[{}] * n{} - EinBlu[{}] * n{}) * NomFact[{}] * (SCMB[{}] * beta[{}] + (1.-beta[{}]) / ( n{}*gul[{}]/(n{}*gul[{}]) -1) )".format(j, j+1, j, j, j, j, j, j, j, j+1, j+1, j)
		                                                          #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		#                                                         #
		#     Stimulated "Gainings/Losses" from/to lower level    #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		if j > 0:                                                 #
			                                                  #
			newline = newline+"- (EinBul[{}] * n{} - EinBlu[{}] * n{}) * NomFact[{}] * (SCMB[{}] * beta[{}] + (1.-beta[{}]) / ( n{}*gul[{}]/(n{}*gul[{}]) -1) )".format(j-1, j, j-1, j-1, j-1, j-1, j-1, j-1, j-1, j, j, j-1)
		                                                          #
		newline = newline + "\n"                                  #
		                                                          #
		detailedBalEqs.write(newline)                             #
else:                                                                     #
	                                                                  #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#    $$$ Code below will produce N+1 Eqs for N unknowns $$$       #
	#   $$$ which is of course redundant! Remember to remove $$$      #
	#      $$$  one when copying one to "populations.py"  $$$         #
	#                                                                 #
	#-----------------------------------------------------------------#
	#                                                                 #
	#   $$$ For collisional processes between sublevels only $$$      #
	# consider m != 0 with m = 0 (e.g. ignore m = +/-1 with m = +/-2) #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	print("\n")                                                       #
	print(" "*13 +"*"*89)                                             #
	print(" "*13 +"*"+" "*87+"*")                                     #
	print(" "*13 +"*" + " "*35 + "! Modeling GK effect !" + " "*30+"*")
	print(" "*13+"* Variable Ujj reserved for transitions with \u0394m = 1 and Rjj for transitions with \u0394m = 0 *")
	print(" "*13 +"*"+" "*87+"*")                                     #
	print(" "*13 +"*"*89)                                             #
	print("\n")                                                       #
	                                                                  #
	newline = ""                                                      #
	                                                                  #
	uniquelevels = np.array([list(x) for x in set(tuple(x) for x in np.concatenate((jmjpmp[:, newaxis:2], jmjpmp[:, 2:newaxis])))])
	                                                                  #
	uniquelevels = uniquelevels[np.argsort(uniquelevels[:, 0] *0.8 + uniquelevels[:, 1]*0.2)]
	                                                                  #
	for i in range (0, len(uniquelevels)):                            #
		                                                          #
		newline = newline + "n{}{}, ".format(uniquelevels[i][0], uniquelevels[i][1])
		                                                          #
	newline = newline[:-2] + " = pd" + "\n"                           #
	                                                                  #
	detailedBalEqs.write(newline)                                     #
	#-----------------------------------------------------------------#
	newline = "eq1 = "                                                #
	                                                                  #
	for i in range (0, len(uniquelevels)):                            #
		                                                          #
		newline = newline + "{} * n{}{} + ".format(gul[i], uniquelevels[i][0], uniquelevels[i][1])
		                                                          #
	newline = newline.replace("1.0 * ", "")                           #
	                                                                  #
	newline = newline[:-2] + "-molgp" + "\n"                          #
	                                                                  #
	detailedBalEqs.write(newline)                                     #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                 #
	#               $$ Time to deal with this mess $$                 #
	#                                                                 #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	upperl, lowerl, upperlC, lowerlC = jmjpmp[:, newaxis:2], jmjpmp[:, 2:newaxis], jmjpmpCul[:, newaxis:2], jmjpmpCul[:, 2:newaxis]
	                                                                  #
	uniquelower = np.array([list(x) for x in set(tuple(x) for x in lowerl)])
	                                                                  #
	uniquelower = uniquelower[np.argsort(uniquelower[:, 0] *0.8 + uniquelower[:, 1]*0.2)]
	                                                                  #
	uniqueupper = np.array([list(x) for x in set(tuple(x) for x in upperl)])
	                                                                  #
	uniqueupper = uniqueupper[np.argsort(uniqueupper[:, 0] *0.8 + uniqueupper[:, 1]*0.2)]
	                                                                  #
	for j in range (0, len(uniquelevels)):                            #
		                                                          #
		newline = "eq{} = ".format(j+2)                           #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-#
		#                                                         #
		#             $$$  Spontaneous "Gainings"  $$$            #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-#
		for l in range (0, len(lowerl)):                          #
			                                                  #
			if np.all(lowerl[l] == uniquelevels[j]):          #
				                                          #
				if upperl[l, 1] == 1. and uniquelevels[j, 1] == 0.:
					                                  #
					newline = newline + "2. * n{}{} * EinA[{}] + ".format(upperl[l, 0], upperl[l, 1], l)
					                                  #
				else:                                     #
					                                  #
					newline = newline + "n{}{} * EinA[{}] + ".format(upperl[l, 0], upperl[l, 1], l)
					                                  #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-#
		#                                                         #
		#             $$$  Spontaneous "Losses"  $$$              #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-#
		if j > 0:                                                 #
			                                                  #
			if uniquelevels[j, 0] < np.max(uniquelevels[:, 0]):
				                                          #
				newline = newline[:-2]                    #
				                                          #
			newline = newline + "- n{}{} * (".format(uniquelevels[j][0], uniquelevels[j][1])
			                                                  #
			for l in range (0, len(upperl)):                  #
				                                          #
				if np.all(upperl[l] == uniquelevels[j]):  #
					                                  #
					if upperl[l, 1] == 0. and lowerl[l, 1] == 1.:
						                          #
						newline = newline + "2. * EinA[{}] + ".format(l)
						                          #
					else:                             #
						                          #
						newline = newline + "EinA[{}] + ".format(l)
						                          #
				                                          #
			newline = newline[:-2] + ") +"                    #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-#
		#                                                         #
		#     $$$  Collisional "Gainings" from upper levels $$$   #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=--=-#
		newline = newline + "densgp * ( "                         # open nH2 parenthesis
		                                                          #
		for l in range (0, len(lowerlC)):                         #
			                                                  #
			if np.all(lowerlC[l] == uniquelevels[j]):         #
				                                          #
				if upperlC[l, 1] == 0:                    #
					                                  #
					newline = newline + "+ n{}{} * Cul[{}, tmin]".format(upperlC[l, 0], upperlC[l, 1], l)
					                                  #
				else:                                     #
					                                  #
					newline = newline + "+ 2. * n{}{} * Cul[{}, tmin]".format(upperlC[l, 0], upperlC[l, 1], l)
					                                  #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		#                                                         #
		#     $$$  Collisional "Losses" to upper levels  $$$      #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		newline = newline + "- n{}{} * (".format(uniquelevels[j][0], uniquelevels[j][1])  # open njm paranthesis
		                                                          #
		for l in range (0, len(lowerlC)):                         #
			                                                  #
			if np.all(lowerlC[l] == uniquelevels[j]):         #
				                                          #
				if upperlC[l, 1] == 0:                    #
					                                  #
					newline = newline + " + Clu[{}, tmin] * CexpF[{}]".format(l, l)
					                                  #
				else:                                     #
					                                  #
					newline = newline + " + 2. * Clu[{}, tmin] * CexpF[{}]".format(l, l)
					                                  #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-#
		#                                                         #
		#     $$$  Collisional "Losses" to lower levels  $$$      #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		for l in range (0, len(upperlC)):                         #
			                                                  #
			if np.all(upperlC[l] == uniquelevels[j]):         #
				                                          #
				if lowerlC[l, 1] == 0:                    #
					                                  #
					newline = newline + " + Cul[{}, tmin]".format(l)
					                                  #
				else:                                     #
					                                  #
					newline = newline + " + 2. * Cul[{}, tmin]".format(l)
					                                  #
		                                                          #
		if uniquelevels[j, 0] <= np.max(uniquelevels[:, 0]):      #
			                                                  #
			newline = newline + ")"                           # close njm parenthesis
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-#
		#                                                         #
		#    $$$  Collisional "Gainings" from lower levels  $$$   #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		for l in range (0, len(upperlC)):                         #
			                                                  #
			if np.all(upperlC[l] == uniquelevels[j]):         #
				                                          #
				if lowerlC[l][1] == 0:                    #
					                                  #
					newline = newline + "+ n{}{} * Clu[{}, tmin] * CexpF[{}]".format(lowerlC[l][0], lowerlC[l][1], l, l)
					                                  #
				else:                                     #
					                                  #
					newline = newline + "+ 2. * n{}{} * Clu[{}, tmin] * CexpF[{}]".format(lowerlC[l][0], lowerlC[l][1], l, l)
					                                  #
		                                                          #
		newline = newline.replace("+densgp *-", "-densgp *")      #
		                                                          #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-#
		#                                                         #
		# $$ Collisional "Gainings"/"Losses" from/to sublevels $$ #
		#                                                         #
		#=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-=-=-=#
		Cnsdr = ""                                                #
		                                                          #
		for l in range (0, len(upperlC)):                         #
			                                                  #
			#   First check for levels with m = 0. We must    #
			#      multiply with a factor of 2 here           #
			if uniquelevels[j, 1] == 0:                       #
				                                          #
				c = "{}".format(upperlC[l].tolist())      #
				                                          #
				if ((uniquelevels[j, 0] == upperlC[l, 0] and uniquelevels[j, 1] != upperlC[l, 1]) and (upperlC[l, 0] - lowerlC[l, 0] == 1) and (c not in Cnsdr)):
					                                  #
					fi = np.where(np.sum(np.equal(lowerlC, lowerlC[l]), axis = 1) == 2)[0][0]
					                                  #
					newline = newline + "+ 2.* CulGK[{}, tmin] * n{}{} - 2. * CluGK[{}, tmin] * n{}{}".format(fi, upperlC[l, 0], upperlC[l, 1], fi, uniquelevels[j, 0], uniquelevels[j, 1])
					                                  #
					Cnsdr = Cnsdr + "{}".format(upperlC[l].tolist())
		                                                          #
		Cnsdr = ""                                                #
		                                                          #
		for l in range (0, len(upperlC)):                         #
			                                                  #
			if uniquelevels[j, 1] != 0:                       #
				                                          #
				c = "{}".format(upperlC[l].tolist())      #
				                                          #
				if ((uniquelevels[j, 0] == upperlC[l, 0] and uniquelevels[j, 1] != upperlC[l, 1]) and (upperlC[l, 0] - lowerlC[l, 0] == 1) and (c not in Cnsdr)):
					                                  #
					fi = np.where(np.sum(np.equal(lowerlC, lowerlC[l]), axis = 1) == 2)[0][0]
					                                  #
					if upperlC[l][1] == 0:            #
						                          #
						newline = newline + "+ CulGK[{}, tmin] * n{}{} - CluGK[{}, tmin] * n{}{}".format(fi, upperlC[l, 0], upperlC[l, 1], fi, uniquelevels[j, 0], uniquelevels[j, 1])
						                          #
					else:                             #
						                          #
						newline = newline + "+ 2. * CulGK[{}, tmin] * n{}{} - 2. * CluGK[{}, tmin] * n{}{}".format(fi, upperlC[l, 0], upperlC[l, 1], fi, uniquelevels[j, 0], uniquelevels[j, 1])
						                          #
					Cnsdr = Cnsdr + "{}".format(upperlC[l].tolist())
		                                                          #
		newline = newline + ")"                                   # close nH2 parenthesis
		                                                          #
		newline = newline.replace(" ( +", " (")                   #
		#.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.#
		#                                                         #
		#    $$$ Spontaneous and Collisional taken care of! $$$   #
		# $$$ Time to do stimulated excitation/de-excitation $$$  #
		#                                                         #
		#.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.#
		for l in range (0, len(upperl)):                          #
			                                                  #
			#        Induced from/to upper levels             #
			#           i >j and DM = 0 first                 #
			if ((upperl[l][0] - uniquelevels[j][0]) == 1 and (upperl[l][1] - uniquelevels[j][1]) == 0 and uniquelevels[j][1] == jmjpmp[l, 3]):   #  DJ = 1, Dm = 0., 1-case for 1 each m
				                                          #
				newline = newline + "+ R[{}] * EinBul[{}] * (n{}{} - n{}{})".format(uniquelevels[j][0], l, upperl[l][0], upperl[l][1], uniquelevels[j][0], uniquelevels[j][1])
				                                          #
			#               Dm !=0 now                        #
			elif ((upperl[l][0] - uniquelevels[j][0]) == 1 and abs(upperl[l][1] - uniquelevels[j][1]) == 1 and uniquelevels[j][1] == jmjpmp[l, 3]):
				                                          #
				if uniquelevels[j][1] == 0:               #
					                                  #  for instance n00 gains 2 * Bul * n11 and loses 2 * Blu * n00 -> 3/4. * U = 3/2. * U
					newline = newline + "+ U[{}] * 2. * EinBul[{}] * (n{}{} - n{}{})".format(uniquelevels[j][0], l, upperl[l][0], upperl[l][1], uniquelevels[j][0], uniquelevels[j][1])
					                                  #
				else:                                     #
					                                  #  for instance n11 gains/loses 1 * Bul/Blu from/to n20 or n22
					newline = newline + "+ U[{}] * EinBul[{}] * (n{}{} - n{}{})".format(uniquelevels[j][0], l, upperl[l][0], upperl[l][1], uniquelevels[j][0], uniquelevels[j][1])
					                                  #
				                                          #
		#.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.#
		for l in range (0, len(lowerl)):                          #
			                                                  #
			#        Induced from/to lower levels             #
			#           i <j and DM = 0 first                 #
			if ((uniquelevels[j][0] - lowerl[l][0]) == 1 and (lowerl[l][1] - uniquelevels[j][1]) == 0 and uniquelevels[j][1] == jmjpmp[l, 1]):
				                                          #
				newline = newline + "- R[{}] * EinBul[{}] * (n{}{} - n{}{})".format(lowerl[l][0], l, uniquelevels[j][0], uniquelevels[j][1], lowerl[l][0], lowerl[l][1])
				                                          #
			#               Dm !=0 now                        #
			elif ((uniquelevels[j][0] - lowerl[l][0]) == 1 and abs(lowerl[l][1] - uniquelevels[j][1]) == 1 and uniquelevels[j][1] == jmjpmp[l, 1]):
				                                          #
				if uniquelevels[j][1] == 0:               #
					                                  #
					newline = newline + "- U[{}] * 2. * EinBul[{}] * (n{}{} - n{}{})".format(lowerl[l][0], l, uniquelevels[j][0], uniquelevels[j][1], lowerl[l][0], lowerl[l][1])
					                                  #
				else:                                     #
					                                  #
					newline = newline + "- U[{}] * EinBul[{}] * (n{}{} - n{}{})".format(lowerl[l][0], l, uniquelevels[j][0], uniquelevels[j][1], lowerl[l][0], lowerl[l][1])
					                                  #
		#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
		#   Just a comment in the code to keep track of which     #
		#  equation refers to which level in "populations.py"     #
		newline = newline + " #  [{}, {}]".format(uniquelevels[j][0], uniquelevels[j][1])
		                                                          #
		newline = newline + "\n"                                  #
		                                                          #
		detailedBalEqs.write(newline)                             #
		                                                          #
                                                                          #
detailedBalEqs.close()                                                    #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
print((" "*15), ("="*78))                                                 #
print((" "*30),"Time for the file you gave me is", round(np.array(pf.current_time)/siny*1e-6, 2), "Myrs")
print((" "*30), ("-"*42))                                                 #
print((" "*20), "The molecule under consideration is", molecule, "or", sp, "in code language")
print((" "*15), ("="*78))                                                 #
print("\n")                                                               #
                                                                          #
#                          Exporting data                                 #
workdir = os.getcwd()                                                     #
                                                                          #
if not os.path.exists('{}/simdata'.format(workdir)):                      #
	                                                                  #
	os.mkdir('{}/simdata'.format(workdir))                            #
	                                                                  #
                                                                          #
if smooth == True:                                                        #
	dens = gaussian_filter(dens, sigma = 2**(level-1))                #
	mol = gaussian_filter(mol, sigma = 2**(level-1))                  #
	velx = gaussian_filter(velx, sigma = 2**(level-1))                #
	vely = gaussian_filter(vely, sigma = 2**(level-1))                #
	velz = gaussian_filter(velz, sigma = 2**(level-1))                #
	T = gaussian_filter(T, sigma = 2**(level-1))                      #
	try:                                                              #
		magx = gaussian_filter(magx, sigma = 2**(level-1))        #
		magy = gaussian_filter(magy, sigma = 2**(level-1))        #
		magz = gaussian_filter(magz, sigma = 2**(level-1))        #
	except NameError:                                                 #
		pass                                                      #
                                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#    $$     Fucking python and it's order of things       $$              #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
dens, mol, T = np.flipud(dens), np.flipud(mol), np.flipud(T)              #
velx, vely, velz = np.flipud(velx), np.flipud(vely), np.flipud(velz)      #
                                                                          #
np.save("simdata/dens", dens)                                             #
np.save("simdata/mol", mol)                                               #
                                                                          #
np.save("simdata/velx", velx)                                             #
np.save("simdata/vely", vely)                                             #
np.save("simdata/velz", velz)                                             #
                                                                          #
np.save("simdata/T", T)                                                   #
np.save("simdata/x", x)                                                   #
np.save("simdata/y", y)                                                   #
np.save("simdata/z", z)                                                   #
                                                                          #
np.save("simdata/ndims", pf.dimensionality)                               #
try:                                                                      #
	magx, magy, magz = np.flipud(magx), np.flipud(magy), np.flipud(magz)
	np.save("simdata/magx", magx)                                     #
	np.save("simdata/magy", magy)                                     #
	np.save("simdata/magz", magz)                                     #
except NameError:                                                         #
	pass                                                              #
                                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                                                                         #
#             $$$  Save Radiative coefficient data  $$$                   #
#                                                                         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
gul = gul.astype(float)                                                   #
np.save("simdata/Cul", Cul)                                               #
np.save("simdata/Clu", Clu)                                               #
np.save("simdata/EinA", EinA)                                             #
np.save("simdata/EinBul", EinBul)                                         #
np.save("simdata/EinBlu", EinBlu)                                         #
np.save("simdata/Ener", Ener)                                             #
np.save("simdata/TranFreq", TranFreq)                                     #
np.save("simdata/gul", gul)                                               #
np.save("simdata/tempers", tempersP)                                      #
np.save("simdata/molmass", massesO[mi-1]*m_p)                             #
np.save("simdata/numlevels", nlevels)                                     #
#    EnerL is not needed anywhere. However, save for every event          #
np.save("simdata/EnerL", EnerL)                                           #
if GK == True:                                                            #
	CulGK = Cul * GKfact                                              #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#  $$$ Here CluGK != Clu * GKfact since statistical weights $$$   #
	#  $$$  already encoded in Clu but for GKClu are explicitly $$$   #
	#  $$$      added in the detailed balance equations         $$$   #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	CluGK = Cul * GKfact                                              #
	CulGK[:, 0:3] = Cul[:, 0:3]                                       #
	CluGK[:, 0:3] = Clu[:, 0:3]                                       #
	np.save("simdata/CulGK", CulGK)                                   #
	np.save("simdata/CluGK", CluGK)                                   #
	np.save("simdata/jmjpmp", jmjpmp)                                 #
	np.save("simdata/jmjpmpCul", jmjpmp)                              #
                                                                          #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
