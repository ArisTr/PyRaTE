#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#  NAME:                                                                     #
#                                                                            #
#  rad_tran_eq_integration.py                                                #
#                                                                            #
#                                                                            #
#  DESCRIPTION:                                                              #
#                                                                            #
#  Python script for computing the line-of-sight intensity. We neend to      #
#  Integrate the radiative trnasfer equation:                                #
#                                                                            #
#           dI/ds=-k_line*I+k_line*S+k_cont*S_cont*e^(t_cont)  (eq. 7-9)     #
#                                                                            #
#  Specifically, we need to compute the quantity:                            #
#                                                                            #
#   I_b={[e^(-t_cont_b)-p]*I_a+p*S_line_a+q*S_line_b+S_k}/(1+q)  (eq. 7-10)  #
#                                                                            #
#  where a and b denote two grid points. Other quantities are define as:     #
#                                                                            #
#          q=t_line_b/(1+e^(-t_line_b))                                      #
#          p=q*e^(-t_line_b-t_cont_b)                                        #
#  S_k=e^(-t_cont_b)*integral_a^b(k_cont*S_cont*e^(integral(k_cont ds')) ds) #
#          t_line_b=integral_a^b(k_line ds)                                  #
#                                                                            #
#   The last equation depends on the population densities N1 & N2 and the    #
#      Doppler-shifted frequency v'. Einstein's coefficients do not vary     #
#        from grid point to grid point. We thus need to compute:             #
#                                                                            #
#      t_line_b=N1*B12(hv0/4pi)[1-(N2g1/N1g2)]*integral_a^b(f(v) ds)         #
#                                                                            #
#              which for a Gaussian-like profile reduces to:                 #
# t_line_b=N1*B12(hv0/4pi)[1-(N2g1/N1g2)]*((sb-sa)/(vlos_b-vlos_a))*(P_b-P_a)#
#                                                                            #
#                     where P is the error function                          #
#                                                                            #
#                                                                            #
#  PARAMETERS:                                                               #
#                                                                            #
#  Input:  LOS velocity, v0 (central velocity of line)                       #
#                                                                            #
#  Output: I_b                                                               #
#                                                                            #
#                                                                            #
#  AUTHOR:                                                                   #
#                                                                            #
#  Aris E. Tritsis                                                           #
#  (aris.tritsis@epfl.ch)                                                    #
#                                                                            #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
import numpy as np                                                           #
from scipy.constants import c                                                #
from numba import jit                                                        #
                                                                             #
#- - - - - - - - - - - - - -Convert to cgs- - - - - - - - - - - - - - - - - -#
c=c*1.e+2                                                                    #
                                                                             #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#                                                                            #
#          /\                                                                #
#         /  \                                                               #
#        / || \                                                              #
#       /  ||  \                                                             #
#      /   ..   \            ALL OTHER SUVROUTINES ARE HERE                  #
#     /          \           TO GET "radTran" WHAT IT NEEDS!                 #
#     ------------                                                           #
#                                                                            #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
@jit(nopython=True)                                                          #
def radTran(freq, vel_a, vel_b, dist, freq_0, Dfreq_thermal, I_a, S_c_a, S_c_b, S_line_a, S_line_b, Line_abs_a, Line_abs_b, k_ext_a, k_ext_b):
	                                                                     #
	Qb = np.exp(-(freq - freq_0 - vel_b/c * freq_0)**2/Dfreq_thermal**2) / (np.sqrt(np.pi) * Dfreq_thermal)
	                                                                     #
	Qa = np.exp(-(freq - freq_0 - vel_a/c * freq_0)**2/Dfreq_thermal**2) / (np.sqrt(np.pi) * Dfreq_thermal)
	                                                                     #
	t_line_b= 0.5*(Line_abs_a*Qa+Line_abs_b*Qb)*dist                     #
	                                                                     #
	q=t_line_b/(1.+np.exp(-t_line_b))                                    #
	                                                                     #
	t_cont_b=dist*(k_ext_a+k_ext_b)/2.                                   #
	                                                                     #
	p=q*np.exp(-t_line_b-t_cont_b)                                       #
	                                                                     #
	S_c = np.exp(-t_cont_b) * dist * 0.5 * (S_c_b * k_ext_b + S_c_a * k_ext_a ) * np.exp(dist*(k_ext_a+k_ext_b)/2.)
	                                                                     #
	I_b=((np.exp(-t_cont_b) -p)*I_a+p*S_line_a+q*S_line_b+S_c)/(1.+q)    #
	                                                                     #
	return I_b                                                           #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
