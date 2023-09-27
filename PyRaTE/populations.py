#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#  NAME:                                                                     #
#                                                                            #
#  populations.py                                                            #
#                                                                            #
#                                                                            #
#  DESCRIPTION:                                                              #
#                                                                            #
#  Python script for calculating the population densities of E-levels        #
#  through a Lambda iteration                                                #
#                                                                            #
#  A. If nonLTE = False: Solve detailed balance assuming escape probability  #
#                        is 1 everywhere                                     #
#  B. If nonLTE = True:  Start with the Boltzman distribution and do small   #
#                        corrections.                                        #
#                                                                            #
#                        If GK = True initial betas are also set to 1.       #
#                        gammas = angle between magnetic field and vectors   #
#                                 used to calculate the integrals            #
#                        gammasPA = angle between the magnetic field and     #
#                                 principle axes                             #
#                        Phis = angle between the vectors used to calculate  #
#                               the integrals and the y-axis (normally theta)#
#                                                                            #
#  1. Assume a Boltzman distribution.                                        #
#  2. Compute dtau_l                                                         #
#  3. Compute tau_l as the sum of dtau_l for dv_ij < Dtherm                  #
#                                                                            #
#         With "dv_ij < Dtherm" we basically assume a                        #
#         step function for the profile.    ^                                #
#                                           |  f(v)                          #
#              1/2v_th Dv<v_th          ____|____                            #
#     f(v) = {                          |   |   |                            #
#              0 otherwise       _______|   |   |________                    #
#                                                                            #
#  4. Compute pd from tau_l                                                  #
#  5. Circle back and check if (PopRat_b-PopRat_a) < tollerance              #
#                                                                            #
#  PARAMETERS:                                                               #
#                                                                            #
#     Input : All arrays from "export_sim"                                   #
#     Output : LevPops, tline                                                #
#                                                                            #
#  COMMENT:                                                                  #
#                                                                            #
#     fsolve can also be replaced with "scipy.optimize.root". Methods "lm"   #
#     and "hybr" seem to be working fine and lead to less descrepancies      #
#     between GK effect and simple case                                      #
#                                                                            #
#  AUTHOR:                                                                   #
#                                                                            #
#  Aris E. Tritsis                                                           #
#  (aris.tritsis@epfl.ch)                                                    #
#                                                                            #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
from numba import jit                                                        #
import numpy as np                                                           #
from scipy.constants import m_p, c, h, k                                     #
from scipy.optimize import fsolve                                            #
from scipy.integrate import nquad                                            #
import sys                                                                   #
                                                                             #
#- - - - - - - - - - - - - -Convert to cgs- - - - - - - - - - - - - - - - - -#
m_p, c, h, Kb = m_p*1e+3, c*1e+2, h*1.e+7, k*1.e+7                           #
amu, fwhm = 2.4237981621576, 2.35482                                         #
Tcmb = 2.7255                                                                #
#                   Weights for LAMBDA iteration                             #
weight1, weight2 = 0.3, 0.7                                                  #
#    Take a "mean" optical depth for fiducial case or min (if mean= False)   #
mean = True                                                                  #
                                                                             #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#                                                                            #
#       $$$         On axis vectors in the cell for GK      $$$              #
#       $$$     These will be used for "Integration" and    $$$              #
#       $$$           for interpolating betas....           $$$              #
#                                                                            #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
rayvecYp, rayvecYm = np.array([1., 0., 0.]), np.array([-1., 0., 0.])         #
rayvecXp, rayvecXm = np.array([0., 1., 0.]), np.array([0., -1., 0.])         #
rayvecZp, rayvecZm = np.array([0., 0., 1.]), np.array([0., 0., -1.])         #
rayvecsA = np.array([rayvecYp, rayvecYm, rayvecXp, rayvecXm, rayvecZp, rayvecZm])
Intlim, IntNorm = 2.*np.pi, 1./(4.* np.pi)                                   #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                                                                            #
#                    $$$  Define Auxiliary Arrays  $$$                       #
#                                                                            #
#  DESCRIPTION:                                                              #
#                                                                            #
#  Define some auxiliary arrays to declutter "populations"                   #
#                                                                            #
#  PARAMETERS:                                                               #
#     Input : freqs, EinA, T, Ener, Cul                                      #
#     Output : NomFact, SCMB/CexpF, Dtherm                                   #
#                                                                            #
def auxiliaryAr(freqs, EinA, nlevels, BgCMB = None, T = None, Ener = None, Cul = None):
	                                                                     #
	if not T:                                                            #
		#            $$$ Not temperature dependent $$$               #
		NomFact, DkFact = [], []                                     #
		                                                             #
		for p in range (0, nlevels-1):                               #
			                                                     #
			NomFact.append(2. * h*freqs[p]**3/c**2)              #
			                                                     #
			DkFact.append(c**2/(8.*np.pi*freqs[p]**2)*EinA[p])   #
			                                                     #
		NomFact, DkFact = np.array(NomFact), np.array(DkFact)        #
		                                                             #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		SCMB = []                                                    #
		                                                             #
		for p in range (0, nlevels-1):                               #
			                                                     #
			if BgCMB == True:                                    #
				                                             #
				SCMB.append( 1. /(np.exp(h*freqs[p]/(Kb*Tcmb)) - 1))
				                                             #
			else:                                                #
				                                             #
				SCMB.append( 0.)                             #
				                                             #
		SCMB = np.array(SCMB)                                        #
		                                                             #
		return NomFact, DkFact, SCMB                                 #
	else:                                                                #
		#               $$$ Temperature dependent $$$                #
		CexpF = []                                                   #
		                                                             #
		for p in range (0, len(Cul)):                                #
			                                                     #
			CexpF.append(np.exp( - (Ener[int(round(Cul[p, 1]) -1)] - Ener[int(round(Cul[p, 2]) -1)])/(Kb*T)))
			                                                     #
		CexpF = np.array(CexpF)                                      #
		                                                             #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		Dtherm = np.sqrt(8.*Kb*T*np.log(2.)/(amu*m_p*c**2))*c/fwhm   #
		                                                             #
		return CexpF, Dtherm                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#    \\      _____ _     _            _       _       //                     #
#     \\    |  ___(_) __| |_   _  ___(_) __ _| |     //                      #
#      \\   | |_  | |/ _` | | | |/ __| |/ _` | |    //                       #
#      //   |  _| | | (_| | |_| | (__| | (_| | |    \\                       #
#     //    |_|   |_|\__,_|\__,_|\___|_|\__,_|_|     \\                      #
#    //                                               \\                     #
#                                                                            #
#             $$$  Detailed Balance Equations  $$$                           #
#                                                                            #
def eqsF(pd, densgp, molgp, gul, Cul, Clu, EinA, EinBul, EinBlu, beta, tmin, CexpF, NomFact, DkFact, SCMB):
	                                                                     #
	n0, n1 = pd                                                          #
	                                                                     #
	eq1 = n0 + n1 -molgp                                                 #
	                                                                     #
	eq2 = densgp * (Cul[0, tmin] * n1 - n0 * (Clu[0, tmin] * CexpF[0]  ) ) + EinA[0] * n1+ (EinBul[0] * n1 - EinBlu[0] * n0) * NomFact[0] * (SCMB[0] * beta[0] + (1.-beta[0]) / ( n0*gul[1]/(n1*gul[0]) -1) )
	                                                                     #
	return eq1, eq2                                                      #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#                                                                            #
#                      $$$ nonLTE case from here on $$$                      #
#                                                                            #
def tauF(pds_a, Dtherm, densgp, mol, gul, vx, vy, vz, dx, dy, dz, index, i, k, ndims, Cul, Clu, EinA, EinBul, EinBlu, tmin, CexpF, NomFact, DkFact, SCMB):
	                                                                     #
	dummy, counter = False, 0                                            #
	                                                                     #
	rtols = np.geomspace(1e-7, 5e-6, len(pds_a))                         #
	                                                                     #
	meanf = 0.25 if ndims == 2 else 0.16667                              #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	while (dummy==False):                                                #
		                                                             #
		pds_ap = pds_a/mol[index, i, k]                              #
		                                                             #
		t_line = []                                                  #
		                                                             #
		popdiffs = pds_ap[0:-1] * gul[1:]/gul[:-1] - pds_ap[1:]      #
		                                                             #
		dtl = []                                                     #
		                                                             #
		for n in range (0, len(DkFact)):                             #
			                                                     #
			dtl.append( DkFact[n] * mol / Dtherm / np.sqrt(np.pi) * popdiffs[n])
			                                                     #
		dtl = np.array(dtl, dtype=np.float64)                        #
		                                                             #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                            #
		#     Now compare which are one thermal linewidth away.      #
		#          Do this towards all 6 directions.                 #
		#                                                            #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		for n in range (0, len(dtl)):                                #
			                                                     #
			#            $$$ SPHERICAL CASE $$$                  #
			if ndims == 1:                                       #
				                                             #
				tlineXp = dtl[n, index+np.where(abs(vx[index, i, k]-vx[index+1:, i, k]) < Dtherm )[0], i, k].sum() * dx
				                                             #
				tlineXp = tlineXp + dtl[n, index, i, k].sum() * dx/2.
				                                             #
				tlineXm, tlineYp, tlineYm, tlineZp, tlineZm = np.nan, np.nan, np.nan, np.nan, np.nan
			#           $$$ CYLINDRICAL CASE $$$                 #
			if ndims > 1:                                        #
				                                             #
				tlineXp = dtl[n, index, i + np.where(abs(vx[index, i, k]-vx[index, i+1:, k]) < Dtherm )[0], k].sum() * dx
				                                             #
				tlineXp = tlineXp + dtl[n, index, i, k].sum() * dx /2.
				                                             #
				tlineYp = dtl[n, index + np.where(abs(vy[index, i, k]-vy[index+1:, i, k]) < Dtherm )[0], i, k].sum() * dy
				                                             #
				tlineYp = tlineYp + dtl[n, index, i, k].sum() * dy /2.
				                                             #
				tlineXm, tlineYm, tlineZp, tlineZm = np.nan, np.nan, np.nan, np.nan
				                                             #
			#            $$$ CARTESIAN CASE $$$                  #
			if ndims > 2:                                        #
				                                             #
				tlineXm = dtl[n, index, np.where(abs(vx[index, i, k]-vx[index, :i, k]) < Dtherm )[0], k].sum() * dx
				                                             #
				tlineXm = tlineXm + dtl[n, index, i, k].sum() * dx/2.
				                                             #
				tlineYm = dtl[n, np.where(abs(vy[index, i, k]-vy[:index, i, k]) < Dtherm )[0], i, k].sum() * dy
				                                             #
				tlineYm = tlineYm + dtl[n, index, i, k].sum() * dy/2.
				                                             #
				tlineZp = dtl[n, index, i, k+np.where(abs(vz[index, i, k]-vz[index, i, k+1:]) < Dtherm )[0]].sum() * dz
				                                             #
				tlineZm = dtl[n, index, i, np.where(abs(vz[index, i, k]-vz[index, i, :k]) < Dtherm )[0]].sum() * dz
				                                             #
				tlineZp = tlineZp + dtl[n, index, i, k].sum() * dz/2.
				                                             #
				tlineZm = tlineZm + dtl[n, index, i, k].sum() * dz/2.
				                                             #
			if mean:                                             #
				                                             #
				t_line.append(1./ ( meanf * np.nansum ([1./tlineXp, 1./tlineXm, 1./tlineYp, 1./tlineYm, 1./tlineZp, 1./tlineZm])))
				                                             #
			else:                                                #
				                                             #
				t_line.append(np.nanmin((tlineXp, tlineXm, tlineYp, tlineYm, tlineZp, tlineZm)))
				                                             #
		t_line = np.array(t_line)                                    #
		                                                             #
		t_line = t_line.flatten()                                    #
		                                                             #
		bet0 = (1.- np.exp(-t_line))/t_line                          #
		                                                             #
		bet0[[not elem for elem in np.isfinite(bet0)]] = 1.          #
		                                                             #
		bet0[bet0 > 1.] = 1.                                         #
		                                                             #
		pds_b = fsolve(eqsF, pds_a, args=(densgp, mol[index, i, k], gul, Cul, Clu, EinA, EinBul, EinBlu, bet0, tmin, CexpF, NomFact, DkFact, SCMB))
		                                                             #
		check = np.absolute(np.absolute(pds_b-pds_a)/pds_a)          #
		                                                             #
		if np.all(check<=rtols):                                     #
			                                                     #
			dummy=True                                           #
			                                                     #
		if counter>=250:                                             #
			                                                     #
			raise SystemExit("No convergence! Last two populations computed were {} and {}. Change the tollerance or initial guess".format(pds_a, pds_b))
		                                                             #
		counter=counter+1                                            #
		                                                             #
		pds_a = pds_b * weight1 + pds_a * weight2                    #
		                                                             #
	return pds_a, t_line                                                 #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                      ____ _  __                                            #
#                     / ___| |/ /                                            #
#                    | |  _| ' /                                             #
#                    | |_| | . \                                             #
#                     \____|_|\_\                                            #
#                                                                            #
def GKangles(Bvec):                                                          #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#                                                                    #
	#  $$$   Calculate angles between B-field and principal axes  $$$    #
	#                                                                    #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	                                                                     #
	return np.arccos(np.dot(rayvecsA, Bvec))                             #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
@jit(nopython=True)                                                          #
def InterpTau(Tline, omega):                                                 #
	                                                                     #
	cos_theta = np.dot(rayvecsA, omega)                                  #
	                                                                     #
	inds = np.where(cos_theta > 0)[0]                                    #
	                                                                     #
	invtau = np.sum(cos_theta[inds] * 1./Tline[inds])/np.sum(cos_theta[inds])
	                                                                     #
	beta = (1. - np.exp(-1./invtau)) * invtau                            #
	                                                                     #
	if beta > 1.: beta = 1.                                              #
	                                                                     #
	return beta                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
def SK(pd, jmjpmp, DkFact, NomFact, both):                                   #
	                                                                     #
	n00, n10, n11, n20, n21, n22, n30, n31, n32, n33 = pd                #
	                                                                     #
	S, k = [], []                                                        #
	                                                                     #
	for p in range (0, len(jmjpmp)):                                     #
		                                                             #
		gu = 1. + (jmjpmp[p, 1] / jmjpmp[p, 1] if jmjpmp[p, 1] else 0.)
		                                                             #
		gl = 1. + (jmjpmp[p, 3] / jmjpmp[p, 3] if jmjpmp[p, 3] else 0.)
		                                                             #
		nu, nl = "n{}{}".format(jmjpmp[p][0], jmjpmp[p][1]), "n{}{}".format(jmjpmp[p][2], jmjpmp[p][3])
		                                                             #
		nu, nl = vars()[nu], vars()[nl]                              #
		                                                             #
		k.append(DkFact[p] * np.max((gu, gl)) * (nl - nu))           #
		                                                             #
		if both:                                                     #
			                                                     #
			S.append(NomFact[p] * 1./ ((nl / nu ) -1.))          #
			                                                     #
		else:                                                        #
			                                                     #
			S = None                                             #
			                                                     #
	return np.array(S), np.array(k)                                      #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
@jit(nopython=True)                                                          #
def SparalsInt(theta, phi, S, k, TlinePa, SCMB, b, jmjpmp, Bvec, cos2):      #
	                                                                     #
	ux = np.cos(theta) * np.sin(phi)                                     #
	                                                                     #
	uy = np.sin(theta) * np.sin(phi)                                     #
	                                                                     #
	uz = np.cos(phi)                                                     #
	                                                                     #
	omega = np.array([uy, ux, uz])                                       #
	                                                                     #
	gamma = np.arccos(np.dot(omega, Bvec))                               #
	                                                                     #
	beta = InterpTau(TlinePa, omega)                                     #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                    #
	#  $$$ In the following: b -> lower level J                          #
	#                        g -> direction for beta (xp, xm, yp...)     #
	#                                                                    #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	tempN, tempD = 0., 0.                                                #
	                                                                     #
	for p in range (0, len(jmjpmp)):                                     #
		                                                             #
		if (jmjpmp[p, 1] - jmjpmp[p, 3]) == 0. and jmjpmp[p, 2] == b:#
			                                                     #
			tempN, tempD = tempN + np.sin(gamma)**2 * S[p] * k[p], tempD + np.sin(gamma)**2 * k[p]
			                                                     #
		elif (jmjpmp[p, 1] - jmjpmp[p, 3]) != 0. and jmjpmp[p, 2] == b:
			                                                     #
			tempN, tempD = tempN + 0.5 * np.cos(gamma)**2 * S[p] * k[p], tempD + 0.5 * np.cos(gamma)**2 * k[p]
			                                                     #
	if cos2:                                                             #
		                                                             #
		return np.cos(gamma)**2 * (tempN/tempD * (1. - beta) + SCMB * beta) * np.sin(phi)
		                                                             #
	else:                                                                #
		                                                             #
		return (tempN/tempD * (1. - beta) + SCMB * beta) * np.sin(phi)
	                                                                     #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
@jit(nopython=True)                                                          #
def SperpsInt(theta, phi, Sperp, SCMB, TlinePe):                             #
	                                                                     #
	ux = np.cos(theta) * np.sin(phi)                                     #
	                                                                     #
	uy = np.sin(theta) * np.sin(phi)                                     #
	                                                                     #
	uz = np.cos(phi)                                                     #
	                                                                     #
	omega = np.array([uy, ux, uz])                                       #
	                                                                     #
	beta = InterpTau(TlinePe, omega)                                     #
	                                                                     #
	return (Sperp * (1. - beta) + SCMB * beta) * np.sin(phi)             #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
def GKEm(pd, NomFact, DkFact, SCMB, TlinePe, TlinePa, jmjpmp, Bvec):         #
	                                                                     #
	S, k = SK(pd, jmjpmp, DkFact, NomFact, True)                         #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                    #
	#           $$$ Sparal/Sperp together with R & U $$$                 #
	#                                                                    #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	                                                                     #
	U, R = [], []                                                        #
	                                                                     #
	for b in range (0, len(TlinePe)):                                    #
		                                                             #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		#                                                            #
		#     $$$ For NomFact, SCMB we do not mind if ind $$$        #
		#     $$$  absolutely correct since these depend  $$$        #
		#     $$$          only on frequency and          $$$        #
		#     $$$ [1, 0, 0, 0]/[1, 1, 0, 0] have same f0  $$$        #
		#                                                            #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		ind = np.where(jmjpmp[:, 2] == b)[0][0]                      #
		                                                             #
		tempN, tempD = 0., 0.                                        #
		                                                             #
		for p in range (0, len(jmjpmp)):                             #
			                                                     #
			if (jmjpmp[p, 1] - jmjpmp[p, 3]) != 0. and jmjpmp[p, 2] == b:
				                                             #
				tempN, tempD = tempN + S[p]*k[p], tempD + k[p]
		                                                             #
		Sperp = tempN/tempD                                          #
		                                                             #
		Sp, error = nquad(SparalsInt, [(0, Intlim), (0, np.pi)], args=(S, k, TlinePa[b], SCMB[ind], b, jmjpmp, Bvec, False))
		                                                             #
		SpInt, error = nquad(SparalsInt, [(0, Intlim), (0, np.pi)], args=(S, k, TlinePa[b], SCMB[ind], b, jmjpmp, Bvec, True))
		                                                             #
		Uperp, error = nquad(SperpsInt, [(0, Intlim), (0, np.pi)], args=(Sperp, SCMB[ind], TlinePe[b]))
		                                                             #
		U.append((Uperp + SpInt) * IntNorm) ; R.append((Sp - SpInt) * IntNorm)
		                                                             #
	U, R = np.array(U) * 1.5, np.array(R) * 3.                           #
	                                                                     #
	return R, U                                                          #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
def eqsGK(pd, densgp, molgp, Cul, Clu, EinA, EinBul, EinBlu, TlinePe, TlinePa, tmin, CexpF, NomFact, DkFact, SCMB, CulGK, CluGK, jmjpmp, Bvec):
	                                                                     #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#          U corresponds to => Dm = 1 and R to => Dm = 0             #
	#      Bx, By, Bz here are only for grid point index, i, k           #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	R, U = GKEm(pd, NomFact, DkFact, SCMB, TlinePe, TlinePa, jmjpmp, Bvec)
	                                                                     #
	n00, n10, n11, n20, n21, n22, n30, n31, n32, n33 = pd                #
	                                                                     #
	eq1 = n00 + n10 + 2.0 * n11 + n20 + 2.0 * n21 + 2.0 * n22 + n30 + 2.0 * n31 + 2.0 * n32 + 2.0 * n33 -molgp
	                                                                     #
	eq2 = n10 * EinA[0] + 2. * n11 * EinA[1] + densgp * ( n10 * Cul[0, tmin]+ 2. * n11 * Cul[1, tmin]+ n20 * Cul[2, tmin]+ 2. * n21 * Cul[3, tmin]+ 2. * n22 * Cul[4, tmin]+ n30 * Cul[11, tmin]+ 2. * n31 * Cul[12, tmin]+ 2. * n32 * Cul[13, tmin]+ 2. * n33 * Cul[14, tmin]- n00 * ( Clu[0, tmin] * CexpF[0] + 2. * Clu[1, tmin] * CexpF[1] + Clu[2, tmin] * CexpF[2] + 2. * Clu[3, tmin] * CexpF[3] + 2. * Clu[4, tmin] * CexpF[4] + Clu[11, tmin] * CexpF[11] + 2. * Clu[12, tmin] * CexpF[12] + 2. * Clu[13, tmin] * CexpF[13] + 2. * Clu[14, tmin] * CexpF[14]))+ R[0] * EinBul[0] * (n10 - n00)+ U[0] * 2. * EinBul[1] * (n11 - n00) #  [0, 0]
	                                                                     #
	eq3 = n20 * EinA[2] + 2. * n21 * EinA[3] - n10 * (EinA[0] ) +densgp * ( n20 * Cul[5, tmin]+ 2. * n21 * Cul[7, tmin]+ 2. * n22 * Cul[9, tmin]+ n30 * Cul[15, tmin]+ 2. * n31 * Cul[17, tmin]+ 2. * n32 * Cul[19, tmin]+ 2. * n33 * Cul[21, tmin]- n10 * ( Clu[5, tmin] * CexpF[5] + 2. * Clu[7, tmin] * CexpF[7] + 2. * Clu[9, tmin] * CexpF[9] + Clu[15, tmin] * CexpF[15] + 2. * Clu[17, tmin] * CexpF[17] + 2. * Clu[19, tmin] * CexpF[19] + 2. * Clu[21, tmin] * CexpF[21] + Cul[0, tmin])+ n00 * Clu[0, tmin] * CexpF[0]+ 2.* CulGK[0, tmin] * n11 - 2. * CluGK[0, tmin] * n10)+ R[1] * EinBul[2] * (n20 - n10)+ U[1] * 2. * EinBul[3] * (n21 - n10)- R[0] * EinBul[0] * (n10 - n00) #  [1, 0]
	                                                                     #
	eq4 = n20 * EinA[4] + n21 * EinA[5] + n22 * EinA[6] - n11 * (EinA[1] ) +densgp * ( n20 * Cul[6, tmin]+ 2. * n21 * Cul[8, tmin]+ 2. * n22 * Cul[10, tmin]+ n30 * Cul[16, tmin]+ 2. * n31 * Cul[18, tmin]+ 2. * n32 * Cul[20, tmin]+ 2. * n33 * Cul[22, tmin]- n11 * ( Clu[6, tmin] * CexpF[6] + 2. * Clu[8, tmin] * CexpF[8] + 2. * Clu[10, tmin] * CexpF[10] + Clu[16, tmin] * CexpF[16] + 2. * Clu[18, tmin] * CexpF[18] + 2. * Clu[20, tmin] * CexpF[20] + 2. * Clu[22, tmin] * CexpF[22] + Cul[1, tmin])+ n00 * Clu[1, tmin] * CexpF[1]+ CulGK[0, tmin] * n10 - CluGK[0, tmin] * n11)+ U[1] * EinBul[4] * (n20 - n11)+ R[1] * EinBul[5] * (n21 - n11)+ U[1] * EinBul[6] * (n22 - n11)- U[0] * EinBul[1] * (n11 - n00) #  [1, 1]
	                                                                     #
	eq5 = n30 * EinA[7] + 2. * n31 * EinA[8] - n20 * (EinA[2] + 2. * EinA[4] ) +densgp * ( n30 * Cul[23, tmin]+ 2. * n31 * Cul[26, tmin]+ 2. * n32 * Cul[29, tmin]+ 2. * n33 * Cul[32, tmin]- n20 * ( Clu[23, tmin] * CexpF[23] + 2. * Clu[26, tmin] * CexpF[26] + 2. * Clu[29, tmin] * CexpF[29] + 2. * Clu[32, tmin] * CexpF[32] + Cul[2, tmin] + Cul[5, tmin] + 2. * Cul[6, tmin])+ n00 * Clu[2, tmin] * CexpF[2]+ n10 * Clu[5, tmin] * CexpF[5]+ 2. * n11 * Clu[6, tmin] * CexpF[6]+ 2.* CulGK[5, tmin] * n21 - 2. * CluGK[5, tmin] * n20+ 2.* CulGK[5, tmin] * n22 - 2. * CluGK[5, tmin] * n20)+ R[2] * EinBul[7] * (n30 - n20)+ U[2] * 2. * EinBul[8] * (n31 - n20)- R[1] * EinBul[2] * (n20 - n10)- U[1] * 2. * EinBul[4] * (n20 - n11) 
	                                                                     #
	eq6 = n30 * EinA[9] + n31 * EinA[10] + n32 * EinA[11] - n21 * (EinA[3] + EinA[5] ) +densgp * ( n30 * Cul[24, tmin]+ 2. * n31 * Cul[27, tmin]+ 2. * n32 * Cul[30, tmin]+ 2. * n33 * Cul[33, tmin]- n21 * ( Clu[24, tmin] * CexpF[24] + 2. * Clu[27, tmin] * CexpF[27] + 2. * Clu[30, tmin] * CexpF[30] + 2. * Clu[33, tmin] * CexpF[33] + Cul[3, tmin] + Cul[7, tmin] + 2. * Cul[8, tmin])+ n00 * Clu[3, tmin] * CexpF[3]+ n10 * Clu[7, tmin] * CexpF[7]+ 2. * n11 * Clu[8, tmin] * CexpF[8]+ CulGK[5, tmin] * n20 - CluGK[5, tmin] * n21+ 2. * CulGK[5, tmin] * n22 - 2. * CluGK[5, tmin] * n21)+ U[2] * EinBul[9] * (n30 - n21)+ R[2] * EinBul[10] * (n31 - n21)+ U[2] * EinBul[11] * (n32 - n21)- U[1] * EinBul[3] * (n21 - n10)- R[1] * EinBul[5] * (n21 - n11) #  [2, 1]
	                                                                     #
	eq7 = n31 * EinA[12] + n32 * EinA[13] + n33 * EinA[14] - n22 * (EinA[6] ) +densgp * ( n30 * Cul[25, tmin]+ 2. * n31 * Cul[28, tmin]+ 2. * n32 * Cul[31, tmin]+ 2. * n33 * Cul[34, tmin]- n22 * ( Clu[25, tmin] * CexpF[25] + 2. * Clu[28, tmin] * CexpF[28] + 2. * Clu[31, tmin] * CexpF[31] + 2. * Clu[34, tmin] * CexpF[34] + Cul[4, tmin] + Cul[9, tmin] + 2. * Cul[10, tmin])+ n00 * Clu[4, tmin] * CexpF[4]+ n10 * Clu[9, tmin] * CexpF[9]+ 2. * n11 * Clu[10, tmin] * CexpF[10]+ CulGK[5, tmin] * n20 - CluGK[5, tmin] * n22+ 2. * CulGK[5, tmin] * n21 - 2. * CluGK[5, tmin] * n22)+ U[2] * EinBul[12] * (n31 - n22)+ R[2] * EinBul[13] * (n32 - n22)+ U[2] * EinBul[14] * (n33 - n22)- U[1] * EinBul[6] * (n22 - n11) #  [2, 2]
	                                                                     #
	eq8 = - n30 * (EinA[7] + 2. * EinA[9] ) +densgp * ( - n30 * ( Cul[11, tmin] + Cul[15, tmin] + 2. * Cul[16, tmin] + Cul[23, tmin] + 2. * Cul[24, tmin] + 2. * Cul[25, tmin])+ n00 * Clu[11, tmin] * CexpF[11]+ n10 * Clu[15, tmin] * CexpF[15]+ 2. * n11 * Clu[16, tmin] * CexpF[16]+ n20 * Clu[23, tmin] * CexpF[23]+ 2. * n21 * Clu[24, tmin] * CexpF[24]+ 2. * n22 * Clu[25, tmin] * CexpF[25]+ 2.* CulGK[23, tmin] * n31 - 2. * CluGK[23, tmin] * n30+ 2.* CulGK[23, tmin] * n32 - 2. * CluGK[23, tmin] * n30+ 2.* CulGK[23, tmin] * n33 - 2. * CluGK[23, tmin] * n30)- R[2] * EinBul[7] * (n30 - n20)- U[2] * 2. * EinBul[9] * (n30 - n21) #  [3, 0]
	                                                                     #
	eq9 = - n31 * (EinA[8] + EinA[10] + EinA[12] ) +densgp * ( - n31 * ( Cul[12, tmin] + Cul[17, tmin] + 2. * Cul[18, tmin] + Cul[26, tmin] + 2. * Cul[27, tmin] + 2. * Cul[28, tmin])+ n00 * Clu[12, tmin] * CexpF[12]+ n10 * Clu[17, tmin] * CexpF[17]+ 2. * n11 * Clu[18, tmin] * CexpF[18]+ n20 * Clu[26, tmin] * CexpF[26]+ 2. * n21 * Clu[27, tmin] * CexpF[27]+ 2. * n22 * Clu[28, tmin] * CexpF[28]+ CulGK[23, tmin] * n30 - CluGK[23, tmin] * n31+ 2. * CulGK[23, tmin] * n32 - 2. * CluGK[23, tmin] * n31+ 2. * CulGK[23, tmin] * n33 - 2. * CluGK[23, tmin] * n31)- U[2] * EinBul[8] * (n31 - n20)- R[2] * EinBul[10] * (n31 - n21)- U[2] * EinBul[12] * (n31 - n22) #  [3, 1]
	                                                                     #
	eq10 = - n32 * (EinA[11] + EinA[13] ) +densgp * ( - n32 * ( Cul[13, tmin] + Cul[19, tmin] + 2. * Cul[20, tmin] + Cul[29, tmin] + 2. * Cul[30, tmin] + 2. * Cul[31, tmin])+ n00 * Clu[13, tmin] * CexpF[13]+ n10 * Clu[19, tmin] * CexpF[19]+ 2. * n11 * Clu[20, tmin] * CexpF[20]+ n20 * Clu[29, tmin] * CexpF[29]+ 2. * n21 * Clu[30, tmin] * CexpF[30]+ 2. * n22 * Clu[31, tmin] * CexpF[31]+ CulGK[23, tmin] * n30 - CluGK[23, tmin] * n32+ 2. * CulGK[23, tmin] * n31 - 2. * CluGK[23, tmin] * n32+ 2. * CulGK[23, tmin] * n33 - 2. * CluGK[23, tmin] * n32)- U[2] * EinBul[11] * (n32 - n21)- R[2] * EinBul[13] * (n32 - n22) #  [3, 2]
	                                                                     #
	return eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10             #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#                                                                            #
#     $$$  Calculate optical depths correspondin to // and _|_  $$$          #
#     $$$     Verified multiple times that GK/DW formalisms     $$$          #
#     $$$         are equivalent for a 2-level molecule         $$$          #
#                                                                            #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
def GKOd(pd, DkFact, gammas, jmjpmp, jlevels):                               #
	                                                                     #
	S, dtau = SK(pd, jmjpmp, DkFact, 0, False)                           #
	                                                                     #
	dtauPerp, dtauParal = [], []                                         #
	                                                                     #
	for t in range (len(jlevels)-1):                                     #
		                                                             #
		tempPerp, tempParal = 0., np.zeros(6)                        #
		                                                             #
		for p in range (0, len(jmjpmp)):                             #
			                                                     #
			if (jmjpmp[p, 1] - jmjpmp[p, 3]) !=0 and jmjpmp[p, 2] == jlevels[t]:
				                                             #
				tempPerp = tempPerp + dtau[p]                #
				                                             #
			elif (jmjpmp[p, 1] - jmjpmp[p, 3]) ==0 and jmjpmp[p, 2] == jlevels[t]:
				                                             #
				for g in range (len(gammas)):                #
					                                     #
					tempParal[g] = tempParal[g] + dtau[p] * np.sin(gammas[g]) **2
					                                     #
		dtauPerp.append(tempPerp)                                    #
		                                                             #
		for g in range (len(gammas)):                                #
			                                                     #
			tempParal[g] = tempParal[g] + tempPerp * 0.5 * np.cos(gammas[g]) ** 2
			                                                     #
		dtauParal.append(tempParal)                                  #
	                                                                     #
	dtauPerp, dtauParal  = 0.5 * np.array(dtauPerp), np.array(dtauParal) #
	                                                                     #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                    #
	#     \\                                                      //     #
	#      \\      $$      dtauPerp shape: (J-LEVELS)      $$    //      #
	#       \\     $$          dtauParal shape:            $$   //       #
	#       //     $$           (J-LEVELS, 6)              $$   \\       #
	#      //      $$     6: Yp, Ym, Xp, Xm, Zp, Zm        $$    \\      #
	#     //                                                      \\     #
	#                                                                    #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	return dtauPerp, dtauParal                                           #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
def tauGK(pd_a, Dtherm, densgp, mol, vx, vy, vz, dx, dy, dz, index, i, k, ndims, Cul, Clu, EinA, EinBul, EinBlu, tmin, CexpF, NomFact, DkFact, SCMB, CulGK, CluGK, jmjpmp, gammas, normTau, Bvec):
	                                                                     #
	dummy, counter = False, 0                                            #
	                                                                     #
	jlevels = np.unique(np.concatenate((jmjpmp[:, 0], jmjpmp[:, 2])))    #
	                                                                     #
	rtols = np.geomspace(1e-8, 5e-7, len(pd_a))                          #
	                                                                     #
	indsXp = i + np.where(abs(vx[index, i, k]-vx[index, i+1:, k]) < Dtherm )[0]
	                                                                     #
	indsXm = np.where(abs(vx[index, i, k]-vx[index, :i, k]) < Dtherm )[0]#
	                                                                     #
	indsYp = index + np.where(abs(vy[index, i, k]-vy[index+1:, i, k]) < Dtherm )[0]
	                                                                     #
	indsYm = np.where(abs(vy[index, i, k]-vy[:index, i, k]) < Dtherm )[0]#
	                                                                     #
	if ndims > 2:                                                        #
		                                                             #
		indsZp = k+np.where(abs(vz[index, i, k]-vz[index, i, k+1:]) < Dtherm )[0]
		                                                             #
		indsZm = np.where(abs(vz[index, i, k]-vz[index, i, :k]) < Dtherm )[0]
		                                                             #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	#         $$$       Cylindrical case now       $$$                   #
	else:                                                                #
		indsXm2 = np.where(abs(vx[index, i, k] + np.fliplr(vx)[index, :, k] ) < Dtherm )[0]
		                                                             #
		indsYm2 = np.where(abs(vy[index, i, k] + np.flipud(vx)[:, i, k] ) < Dtherm )[0]
		                                                             #
		# $$$  z (i.e. y) vel - component now, this is tricky  $$$   #
		# $$$  setting i here in "k-location" is not a mistake $$$   #
		indsZp = np.where(abs(vz[index, :, i]) < Dtherm )[0]         #
		                                                             #
		indsZm = indsZp                                              #
	#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
	while (dummy==False):                                                #
		                                                             #
		pds_ap = pd_a/mol[index, i, k]                               #
		                                                             #
		TlinePe, TlinePa = [], []                                    #
		                                                             #
		dtlPerp, dtlParal = GKOd(pds_ap, DkFact, gammas, jmjpmp, jlevels)
		                                                             #
		for n in range (len(dtlPerp)):                               #
			                                                     #
			#* * * * * * * * * * * * * * * * * * * * * * * * * * #
			#                                                    #
			#            $$$ Finalize Tau _|_ first $$$          #
			#                                                    #
			#* * * * * * * * * * * * * * * * * * * * * * * * * * #
			JTauPe = dtlPerp[n] * normTau                        #
			                                                     #
			tl_Xp, tl_Xm = JTauPe[index, indsXp, k].sum() * dx, JTauPe[index, indsXm, k].sum() * dx
			                                                     #
			tl_Xp, tl_Xm = tl_Xp + JTauPe[index, i, k] * dx/2., tl_Xm + JTauPe[index, i, k] * dx/2.
			                                                     #
			#* * * * * * * * * * * * * * * * * * * * * * * * * * #
			tl_Yp, tl_Ym = JTauPe[indsYp, i, k].sum() * dy, JTauPe[indsYm, i, k].sum() * dy
			                                                     #
			tl_Yp, tl_Ym = tl_Yp + JTauPe[index, i, k].sum() * dy/2., tl_Ym + JTauPe[index, i, k].sum() * dy/2.
			                                                     #
			#* * * * * * * * * * * * * * * * * * * * * * * * * * #
			tl_Zp, tl_Zm = JTauPe[index, i, indsZp].sum() * dz, JTauPe[index, i, indsZm].sum() * dz
			                                                     #
			tl_Zp, tl_Zm = tl_Zp + JTauPe[index, i, k] * dz/2., tl_Zm + JTauPe[index, i, k] * dz/2.
			                                                     #
			if ndims == 2:                                       #
				                                             #
				tl_Ym, tl_Xm = tl_Ym + JTauPe[indsYm2, i, k].sum() * dy, JTauPe[index, indsXm2, k].sum() * dx
				                                             #
			TlinePe.append([tl_Yp, tl_Ym, tl_Xp, tl_Xm, tl_Zp, tl_Zm])
			#* * * * * * * * * * * * * * * * * * * * * * * * * * #
			#                                                    #
			#              $$$ Finalize Tau // now $$$           #
			#                                                    #
			#* * * * * * * * * * * * * * * * * * * * * * * * * * #
			#            Y+, Y- (rayvecsA[0:2])                  #
			JTauPap, JTauPam = dtlParal[n, 0] * normTau, dtlParal[n, 1] * normTau
			                                                     #
			tl_Yp, tl_Ym = JTauPap[indsYp, i, k].sum() * dy, JTauPam[indsYm, i, k].sum() * dy
			                                                     #
			tl_Yp, tl_Ym = tl_Yp + JTauPap[index, i, k].sum() * dy/2., tl_Ym + JTauPam[index, i, k].sum() * dy/2.
			#----------------------------------------------------#
			#            X+, X- (rayvecsA[2:4])                  #
			JTauPap, JTauPam = dtlParal[n, 2] * normTau, dtlParal[n, 3] * normTau
			                                                     #
			tl_Xp, tl_Xm = JTauPap[index, indsXp, k].sum() * dy, JTauPam[index, indsXm, k].sum() * dy
			                                                     #
			tl_Xp, tl_Xm = tl_Xp + JTauPap[index, i, k].sum() * dy/2., tl_Xm + JTauPam[index, i, k].sum() * dy/2.
			#----------------------------------------------------#
			#            Z+, Z- (rayvecsA[4:6])                  #
			JTauPap, JTauPam = dtlParal[n, 4] * normTau, dtlParal[n, 5] * normTau
			                                                     #
			tl_Zp, tl_Zm = JTauPap[index, i, indsZp].sum() * dz, JTauPam[index, i, indsZm].sum() * dz
			                                                     #
			tl_Zp, tl_Zm = tl_Zp + JTauPap[index, i, k].sum() * dz/2., tl_Zm + JTauPam[index, i, k].sum() * dz/2.
			                                                     #
			if ndims == 2:                                       #
				                                             #
				tl_Ym, tl_Xm = tl_Ym + JTauPap[indsYm2, i, k].sum() * dy, JTauPap[index, indsXm2, k].sum() * dx
				                                             #
			TlinePa.append([tl_Yp, tl_Ym, tl_Xp, tl_Xm, tl_Zp, tl_Zm])
		                                                             #
		TlinePe, TlinePa = np.array(TlinePe), np.array(TlinePa)      #
		                                                             #
		pd_b = fsolve(eqsGK, pd_a, args=(densgp, mol[index, i, k], Cul, Clu, EinA, EinBul, EinBlu, TlinePe, TlinePa, tmin, CexpF, NomFact, DkFact, SCMB, CulGK, CluGK, jmjpmp, Bvec))
		                                                             #
		check = np.absolute(np.absolute(pd_b-pd_a)/pd_a)             #
		                                                             #
		if np.all(check<=rtols):                                     #
			                                                     #
			dummy=True                                           #
			                                                     #
		if counter>=250:                                             #
			                                                     #
			raise SystemExit("No convergence! Last two populations computed were {} and {}. Change tollerance and/or initial guess".format(pds_a, pds_b))
			                                                     #
		counter=counter+1                                            #
		                                                             #
		pd_a = pd_b * weight1 + pd_a * weight2                       #
		                                                             #
	return pd_a, TlinePe                                                 #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
def populations(nonLTE, BgCMB, ndims, dens, mol, T, vx, dx, vy, dy, vz, dz, y, chopped_ys, Cul, Clu, EinA, EinBul, EinBlu, Ener, freqs, gul, tempers, GK, Bx, By, Bz, numlevels, CulGK=None, CluGK=None, jmjpmp = None):
	                                                                     #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#    $$$  Initialize betas, initial guesses and method of sol  $$$   #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	nlevels = len(EinA) + 1                                              #
	                                                                     #
	NomFact, DkFact, SCMB = auxiliaryAr(freqs, EinA, nlevels, BgCMB)     #
	                                                                     #
	njm, nbetas = nlevels, nlevels - 1                                   #
	                                                                     #
	if GK == True:                                                       #
		                                                             #
		DkFact, NomFact = DkFact * 3., NomFact/2.                    #
		                                                             #
		njm, nbetas = np.sum(range(numlevels+1)), numlevels - 1      #
		                                                             #
		SCMB = SCMB * NomFact                                        #
		                                                             #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	                                                                     #
	LevPops, TLINE = [], []                                              #
	                                                                     #
	size = np.array(dens.shape)                                          #
	                                                                     #
	if ndims==2: size[2] = 1                                             #
	#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
	for j in range (0, len(chopped_ys)):                                 #
		                                                             #
		index=np.where(y==chopped_ys[j])[0][0]                       #
		                                                             #
		temp, tempL = [], []                                         #
		                                                             #
		for i in range (size[1]):                                    #
			                                                     #
			temp2, tempL2 = [], []                               #
			                                                     #
			FirstCall = True                                     #
			                                                     #
			for k in range (size[2]):                            #
				                                             #
				bet0 = np.ones(nbetas)                       #
				                                             #
				if FirstCall: pd_a, FirstCall = np.log10(np.logspace(1.5/njm, 0.5/njm, njm)) * mol[index, i, k], False
				                                             #
				tmin = np.argmin(np.absolute(tempers - T[index, i, k])) + 3
				                                             #
				CexpF, Dtherm = auxiliaryAr(freqs, EinA, nlevels, BgCMB, T[index, i, k], Ener, Cul)
				                                             #
				t_line = np.zeros(nbetas)                    #
				# * * * * * * * * * * * * * * * * * * * * * *#
				#                                            #
				# $$$ some code duplication considering $$$  #
				# $$$    differently the two cases but  $$$  #
				# $$$         better keep it clean      $$$  #
				#                                            #
				# * * * * * * * * * * * * * * * * * * * * * *#
				densgp, molgp = dens[index, i, k], mol[index, i, k]
				                                             #
				if not GK:                                   #
					                                     #
					pd_a=fsolve(eqsF, pd_a, args=(densgp, molgp, gul, Cul, Clu, EinA, EinBul, EinBlu, bet0, tmin, CexpF, NomFact, DkFact, SCMB))
					                                     #
					if nonLTE == True:                   #
						                             #
						pd_a, t_line = tauF(pd_a, Dtherm, densgp, mol, gul, vx, vy, vz, dx, dy, dz, index, i, k, ndims, Cul, Clu, EinA, EinBul, EinBlu, tmin, CexpF, NomFact, DkFact, SCMB)
						                             #
				# * * * * * * * * * * * * * * * * * * * * * *#
				else:                                        #
					                                     #
					Bvec =  np.array([By[index, i, k], Bx[index, i, k], Bz[index, i, k]])
					                                     #
					Bvec = Bvec/np.linalg.norm(Bvec)     #
					                                     #
					gammas = GKangles(Bvec)              #
					                                     #
					# * * * * * * * * * * * * * * * * * *#
					#                                    #
					#  $$ 6.705e-9 below is such that $$ #
					#  $$  beta = (1- e^tau)/tau ~ 1  $$ #
					#  $$    i.e. we start from LTE   $$ #
					#                                    #
					# * * * * * * * * * * * * * * * * * *#
					TlinePe, TlinePa = np.ones((nbetas, 6)) * 6.705e-9, np.ones((nbetas, 6)) * 6.705e-9
					                                     #
					pd_a = fsolve(eqsGK, pd_a, args=(densgp, molgp, Cul, Clu, EinA, EinBul, EinBlu, TlinePe, TlinePa, tmin, CexpF, NomFact, DkFact, SCMB, CulGK, CluGK, jmjpmp, Bvec))
					                                     #
					if nonLTE == True:                   #
						                             #
						normTau = mol / Dtherm / np.sqrt(np.pi)
						                             #
						pd_a, t_line = tauGK(pd_a, Dtherm, densgp, mol, vx, vy, vz, dx, dy, dz, index, i, k, ndims, Cul, Clu, EinA, EinBul, EinBlu, tmin, CexpF, NomFact, DkFact, SCMB, CulGK, CluGK, jmjpmp, gammas, normTau, Bvec)
					                                     #
				temp2.append(pd_a) ; tempL2.append(t_line)   #
				                                             #
			temp.append(temp2) ; tempL.append(tempL2)            #
			                                                     #
		LevPops.append(temp) ; TLINE.append(tempL)                   #
		                                                             #
	LevPops, TLINE = np.array(LevPops), np.array(TLINE)                  #
	                                                                     #
	if GK: TLINE = np.mean(TLINE, axis = 4)                              #
	                                                                     #
	return LevPops, TLINE                                                #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
