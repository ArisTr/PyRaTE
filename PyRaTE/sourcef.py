#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#  NAME:                                                                     #
#                                                                            #
#  sourcef.py                                                                #
#                                                                            #
#                                                                            #
#  DESCRIPTION:                                                              #
#                                                                            #
#  Python script for calculating the continuum and line source function      #
#                                                                            #
#  PARAMETERS:                                                               #
#                                                                            #
#     Input : frq, PopRatgp, nlowgp, EinA, grat                              #
#     Output : S_c, k_ext, S_line, Line_abs                                  #
#                                                                            #
#  AUTHOR:                                                                   #
#                                                                            #
#  Aris E. Tritsis                                                           #
#  (aris.tritsis@epfl.ch)                                                    #
#                                                                            #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
import numpy as np                                                           #
from scipy.constants import h, c                                             #
                                                                             #
#- - - - - - - - - - - - - -Convert to cgs- - - - - - - - - - - - - - - - - -#
h, c = h*1.e+7, c*1.e+2                                                      #
                                                                             #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                                                                            #
#    $$$ Calculate line emission and line absorption coefficients $$$        #
#                                                                            #
def sourcef(frq, PopRatgp, nlowgp, EinBlu, grat, gam, jmjpmp, GK):           #
	                                                                     #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                    #
	#  $$$ This first "if statement" is trully the GK = False case $$$   #
	#                                                                    #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	if not GK and not np.any(jmjpmp):                                    #
		                                                             #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
		                                                             #
		S_line = 2. * h*frq**3/c**2 * 1./((grat/PopRatgp) - 1.)      #
		                                                             #
		#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
		Line_abs = nlowgp * h * frq * EinBlu * (1. - PopRatgp / grat) / (4.*np.pi)
		                                                             #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                    #
	#  $$$ This is where the GK = True. Two subcases for // and _|_ $$$  #
	#                                                                    #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	else:                                                                #
		                                                             #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                            #
		#  $$$       Under LTE (or in large/small tau)       $$$     #
		#  $$$   nupm = 2. * nu such that with statistical   $$$     #
		#  $$$    weights k[1] = 2*k[0] and S[1] = S[0]      $$$     #
		#  $$$ which will give you S// = S_|_ for all gammas $$$     #
		#                                                            #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		S, k = [], []                                                #
		                                                             #
		for i in range (0, len(jmjpmp)):                             #
			                                                     #
			gu = 1. + (jmjpmp[i, 1] / jmjpmp[i, 1] if jmjpmp[i, 1] else 0.)
			                                                     #
			gl = 1. + (jmjpmp[i, 3] / jmjpmp[i, 3] if jmjpmp[i, 3] else 0.)
			                                                     #
			nu, nl = np.sum(jmjpmp[i, 0:2]), np.sum(jmjpmp[i, 2:]) - jmjpmp[i, 2]
			                                                     #
			nu, nl = PopRatgp[nu], PopRatgp[nl]                  #
			                                                     #
			S.append(h*frq**3/c**2 * 1./ (nl / nu  -1.))         #
			                                                     #
			k.append(3. * nl * h * frq * EinBlu[i] * np.max((gu, gl)) * (1. - nu/nl ) / (4.*np.pi))
			                                                     #
		S, k = np.array(S), np.array(k)                              #
		                                                             #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                            #
		#             Calculate Sperp, kperp                         #
		#                                                            #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		if GK == False:                                              #
			                                                     #
			Line_abs, S_line = 0., 0.                            #
			                                                     #
			for i in range (0, len(jmjpmp)):                     #
				                                             #
				if abs(jmjpmp[i, 3] - jmjpmp[i, 1]) == 1.:   #
					                                     #
					S_line, Line_abs = S_line + S[i]*k[i], Line_abs + k[i]
					                                     #
			S_line = S_line/Line_abs                             #
			                                                     #
			Line_abs = 0.5 * Line_abs                            #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                            #
		#            Calculate Sparal, kparal                        #
		#                                                            #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		else:                                                        #
			                                                     #
			Line_abs, S_line = 0., 0.                            #
			                                                     #
			for i in range (0, len(jmjpmp)):                     #
				                                             #
				if abs(jmjpmp[i, 3] - jmjpmp[i, 1]) == 0.:   #
					                                     #
					S_line, Line_abs = S_line + S[i]*k[i] * np.sin(gam)**2, Line_abs + k[i] * np.sin(gam)**2
					                                     #
				else:                                        #
					                                     #
					S_line, Line_abs = S_line + 0.5 * S[i]*k[i] * np.cos(gam)**2, Line_abs + 0.5 * k[i] * np.cos(gam)**2
					                                     #
			S_line = S_line/Line_abs                             #
			                                                     #
	return S_line, Line_abs                                              #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
