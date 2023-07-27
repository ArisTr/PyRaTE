#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#  NAME:                                                                     #
#                                                                            #
#  render3D.py                                                               #
#                                                                            #
#                                                                            #
#  DESCRIPTION:                                                              #
#                                                                            #
#  Python script for volume rendering of a 2D simulation into 3D             #
#                                                                            #
#  PARAMETERS:                                                               #
#                                                                            #
#     Input : r, z, qtor                                                     #
#     Input : typeQ (parameter type, accepted values: "scalar", "vectorR"    #
#                    "vectorZ")                                              #
#     Output : render3Dq (rendered quantity)                                 #
#              Add if np.max(x) + 3. * dx >= radiusf:                        #
#                                                                            #
#  AUTHOR:                                                                   #
#                                                                            #
#  Aris E. Tritsis                                                           #
#  (aris.tritsis@epfl.ch)                                                    #
#                                                                            #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
import numpy as np                                                           #
from scipy import interpolate                                                #
import warnings                                                              #
                                                                             #
warnings.filterwarnings("ignore")                                            #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
def radius(x, y):                                                            #
	                                                                     #
	return np.sqrt(x**2+y**2)                                            #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
                                                                             #
def render3D(qtor, r, z, typeQ, inteType = "linear"):                        #
	                                                                     #
	qtor, r = qtor[:, :, 0], r - r[0]                                    #
	                                                                     #
	X, Y = np.meshgrid(r, r)                                             #
	                                                                     #
	qnt3D = []                                                           #
	                                                                     #
	for i in range (0, len(qtor)):                                       #
		                                                             #
		temp=interpolate.griddata(r, qtor[i, :], radius(X, Y), method= inteType)
		                                                             #
		if typeQ == "scalar":                                        #
			                                                     #
			temp = np.vstack((np.flipud(temp), temp))            #
			                                                     #
		elif typeQ == "vectorR":                                     #
			                                                     #
			temp = temp * np.sin(np.arctan2(Y, X))               #
			                                                     #
			temp[:, 0] = qtor[i, :]                              #
			                                                     #
			temp = np.vstack((-np.flipud(temp), temp))           #
			                                                     #
		elif typeQ == "vectorZ":                                     #
			                                                     #
			temp = np.vstack((np.flipud(temp), temp))            #
			                                                     #
		qnt3D.append(temp)                                           #
	                                                                     #
	qnt3D = np.array(qnt3D)                                              #
	                                                                     #
	del X, Y, temp, qtor                                                 #
	                                                                     #
	qnt3D = np.moveaxis(qnt3D, 2, 1)                                     #
	                                                                     #
	return qnt3D                                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
def cntCart(qtor, x, typeQ, inteType = "linear"):                            #
	                                                                     #
	dx = x[1] - x[0]                                                     #
	                                                                     #
	qtor = qtor[:, :, 0]                                                 #
	                                                                     #
	Xc, Zc = np.meshgrid(x, np.concatenate((-x[::-1], x)))               #
	                                                                     #
	qnt3D = []                                                           #
	                                                                     #
	for j in range (0, len(qtor)):                                       #
		                                                             #
		prof1d = qtor[j, :]                                          #
		                                                             #
		f = interpolate.interp1d(x, prof1d, kind = inteType, bounds_error = False, fill_value = (prof1d[0], prof1d[-1]))
		                                                             #
		temp = []                                                    #
		                                                             #
		for i in range (0, len(Xc)):                                 #
			                                                     #
			temp2 = []                                           #
			                                                     #
			for k in range (0, len(Xc[0])):                      #
				                                             #
				radius = np.sqrt(Xc[i, k]**2 + Zc[i, k]**2)  #
				                                             #
				#radiusf = np.sqrt((Xc[i, k] - dx/2.)**2 + (abs(Zc[i, k])-dx/2.)**2)
				                                             #
				if typeQ == "vectorR":                       #
					                                     #
					theta = np.arctan2(Zc[i, k], Xc[i, k])
					                                     #
					temp2.append(f(radius) * np.sin(theta))
					                                     #
				else:                                        #
					                                     #
					temp2.append(f(radius))              #
					                                     #
			temp.append(temp2)                                   #
			                                                     #
		temp = np.array(temp)                                        #
		                                                             #
		temp = np.flipud(np.rot90(temp))                             #
		                                                             #
		qnt3D.append(temp)                                           #
		                                                             #
	qnt3D = np.array(qnt3D)                                              #
	                                                                     #
	return qnt3D                                                         #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
