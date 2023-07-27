#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#  NAME:                                                                  #
#                                                                         #
#     postprocess                                                         #
#                                                                         #
#  DESCRIPTION:                                                           #
#                                                                         #
#     Post process line RT simulations to:                                #
#        1. Change spectral resolution                                    #
#        2. Change spatial resolution                                     #
#        3. Add noise                                                     #
#        4. Convert to different units                                    #
#        5. Export data to fits format                                    #
#                                                                         #
# Useful reading:                                                         #
# https://hera.ph1.uni-koeln.de/ftpspace/simonr/Pablo/Radioastronomy.pdf  #
#                                                                         #
#  SIMULATION PARAMETERS:                                                 #
#                                                                         #
#     SpectralRes: Desired spectral resolution in km/s. Should be >= Vres #
#     SpatialRes: The desired spatial resolution in pc. Should be >= dx   #
#     SNR: Desired SNR (if 0 no noise added)                              #
#     units: Convert to radio-astronomy units (cgs, Ta or Tmb)            #
#                                                                         #
#  AUTHOR:                                                                #
#                                                                         #
#  Aris E. Tritsis                                                        #
#  (aris.tritsis@epfl.ch)                                                 #
                                                                          #
import numpy as np                                                        #
from numpy.random import standard_normal                                  #
from astropy.io import fits                                               #
from astropy.wcs import WCS                                               #
from astropy import units as u                                            #
from astropy.convolution import Gaussian2DKernel, convolve                #
from scipy import ndimage                                                 #
from scipy.constants import parsec, c, h, k                               #
import os                                                                 #
                                                                          #
parsec, c, h, Kb = parsec*1e+2, c*1e+2, h*1.e+7, k*1.e+7                  #
fwhm = 2.355                                                              #
                                                                          #
cwd = os.getcwd()                                                         #
PPV, x, Vrange, frange = np.load('{}/output_files/PPV.npy'.format(cwd)), np.load("{}/simdata/x.npy".format(cwd))/parsec, np.load('{}/output_files/Vrange.npy'.format(cwd)), np.load('{}/output_files/frRange.npy'.format(cwd))
Vres = Vrange[1] - Vrange[0]                                              #
molecule, freqs, up, low = "CO", np.load("simdata/TranFreq.npy"), 1, 0    #
nonLTE, BgCMB = True, True                                                #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
                                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#                 User defined variables                                  #
SpatialRes, SpectralRes = 0.01, 0.01                                      #
SNR = 30.                                                                 #
units, beam_size = "Tmb", 32                                              #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#              Spatial resolution change                                  #
                                                                          #
dx = x[1] - x[0]                                                          #
                                                                          #
if (SpatialRes/dx) == 1:                                                  #
	                                                                  #
	pass                                                              #
	                                                                  #
else:                                                                     #
	                                                                  #
	sigma = (SpatialRes/dx)/fwhm                                      #
	                                                                  #
	kernel = Gaussian2DKernel(x_stddev=sigma)                         #
	                                                                  #
	for i in range (0, len(PPV[0, 0])):                               #
		                                                          #
		PPV[:, :, i] = convolve(PPV[:, :, i], kernel, boundary ="extend")
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#             Spectral resolution change                                  #
                                                                          #
if (SpectralRes/Vres) == 1:                                               #
	                                                                  #
	pass                                                              #
	                                                                  #
else:                                                                     #
	                                                                  #
	sigma = (SpectralRes/Vres)/fwhm                                   #
	                                                                  #
	for i in range (0, len(PPV)):                                     #
		                                                          #
		for j in range (0, len(PPV[i])):                          #
			                                                  #
			PPV[i, j, :] = ndimage.gaussian_filter(PPV[i, j, :], sigma=sigma, mode='nearest')
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                         Add noise based on SNR                          #
if SNR == 0:                                                              #
	                                                                  #
	pass                                                              #
	                                                                  #
else:                                                                     #
	                                                                  #
	gamma = 10**(SNR/10.)                                             #
	                                                                  #
	P = np.sum(abs(PPV[len(PPV)//2, len(PPV[0])//2, :])**2)/len(PPV[len(PPV)//2, len(PPV[0])//2, :])
	                                                                  #
	N0=P/gamma                                                        #
	                                                                  #
	for i in range (0, len(PPV)):                                     #
		                                                          #
		for j in range (0, len(PPV[i])):                          #
			                                                  #
			noise = np.sqrt(N0/2.)*standard_normal(len(PPV[len(PPV)//2, len(PPV[0])//2, :]))
			                                                  #
			PPV[i, j, :] = PPV[i, j, :] + noise               #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                       Change units (Jy/Teff/Ta)                         #
if units == "cgs":                                                        #
	                                                                  #
	pass                                                              #
	                                                                  #
elif units == "Ta" or units == "Tmb":                                     #
	                                                                  #
	wavel = (c/freqs[low])                                            # [cm]
	                                                                  #
	beam_size=beam_size/3600.*np.pi/180.                              # [sr]
	                                                                  #
	D = 1.22 * wavel/beam_size                                        # [cm]
	                                                                  #
	Omega_mb = 1.133 * beam_size ** 2.                                # [sr]
	                                                                  #
	Ageo =  np.pi/4. * D**2                                           # [cm^2]
	                                                                  #
	#    Constant antenna efficiency assumed here ittaA = 0.5         #
	A_eff = 0.5* Ageo                                                 # [cm^2]
	                                                                  #
	Omega_A = wavel**2 / A_eff                                        # [cm^2 sr^-1]
	                                                                  #
	PPV = 0.5*PPV*wavel**2/Kb                                         # [K]
	                                                                  #
	if units == "Tmb":                                                #
		                                                          #
		itta_mb = Omega_mb/Omega_A                                # [dimensionless]
		                                                          #
		PPV = PPV/itta_mb                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
hdu = fits.PrimaryHDU(PPV)                                                #
                                                                          #
hdul = fits.HDUList([hdu])                                                #
                                                                          #
hdr = hdul[0].header                                                      #
                                                                          #
hdr.set('BUNIT', "   {}".format(units))                                   #
hdr.set('LINE', "   {}".format(molecule))                                 #
                                                                          #
hdr.set('CDELT1', SpatialRes)                                             #
hdr.set('CDELT2', SpatialRes)                                             #
hdr.set('CDELT3', SpectralRes)                                            #
                                                                          #
hdr.set('CUNIT1', "   pc")                                                #
hdr.set('CUNIT2', "   pc")                                                #
hdr.set('CUNIT3', "   km s-1")                                            #
                                                                          #
hdr.set('RESTFRQ', "   {} / [Hz] Line rest frequency".format(freqs[low])) #
                                                                          #
hdr.set('COMMENT', "non-LTE parameter set to {}".format(nonLTE))          #
hdr.set('COMMENT', "Background radiation CMB set to {}".format(BgCMB))    #
hdr.set('COMMENT', "Axis order: 2 3 1")                                   #
hdr.set('COMMENT', "Your own simulated PPV cube. Enjoy :)")               #
                                                                          #
hdul.writeto('{}/output_files/MyMockObs{}.fits'.format(cwd, molecule))    #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
