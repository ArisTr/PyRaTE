#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#  NAME:                                                                     #
#                                                                            #
#  axisRearange.py                                                           #
#                                                                            #
#                                                                            #
#  DESCRIPTION:                                                              #
#                                                                            #
#  Python script for rearranging axes based on projection angle/geometry     #
#  so that the axis of integration ends up at the end                        #
#                                                                            #
#  PARAMETERS:                                                               #
#                                                                            #
#     Input : ndim, fi0, th0, dens, mol, vx, vy, vz, T, PopRat, nlow         #
#     Output : dens, mol, vLOS, T, PopRat, nlow, LOSaxis                     #
#                                                                            #
#  AUTHOR:                                                                   #
#                                                                            #
#  Aris E. Tritsis                                                           #
#  (aris.tritsis@epfl.ch)                                                    #
#                                                                            #
import numpy as np                                                           #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
def axisRearange(ndim, line, fi0, th0, dens, mol, vx, vy, vz, T, PopRat, nlow, x, y, z, Bx, By, Bz, GK):
	                                                                     #
	size = dens.shape                                                    #
	                                                                     #
	if line == True:                                                     #
		                                                             #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		#                                                            #
		# $$ Spherical case, we don't care about projection angle $$ #
		#                                                            #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		Blos = None                                                  #
		                                                             #
		if ndim == 1:                                                #
			                                                     #
			dens, mol, T = np.hstack((dens[::-1], dens)), np.hstack((mol[::-1], mol)), np.hstack((T[::-1], T))
			                                                     #
			PopRat, nlow = np.hstack((PopRat[::-1], PopRat)), np.hstack((nlow[::-1], nlow))
			                                                     #
			vLOS, LOSaxis = np.hstack((-vx[::-1], vx)), np.hstack((-x[::-1], x))
			                                                     #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		#                                                            #
		#  $$ Cylindircal case, 2 cases based on projection angle $$ #
		#                                                            #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		elif ndim == 2:                                              #
			                                                     #
			if fi0 == 90.:                                       #
				                                             #
				dens, mol, T = dens[:, 0, 0], mol[:, 0, 0], T[:, 0, 0]
				                                             #
				PopRat, nlow = PopRat[:, 0, 0], nlow[:, 0, 0]#
				                                             #
				vLOS, LOSaxis = vy[:, 0, 0], y               #
				                                             #
				if GK:                                       #
					                                     #
					Bx, By, Bz = Bx[:, 0, 0], By[:, 0, 0], Bz[:, 0, 0]
					                                     #
					Blos = By                            #
					                                     #
			elif fi0 == 0.:                                      #
				                                             #
				dens, mol, T = dens[len(dens)//2, :, 0], mol[len(mol)//2, :, 0], T[len(T)//2, :, 0]
				                                             #
				PopRat, nlow = PopRat[len(PopRat)//2, :, 0], nlow[len(nlow)//2, :, 0]
				                                             #
				vLOS = vx[len(vx)//2, :, 0]                  #
				                                             #
				# * * * * * * * * * * * * * * * * * * * * * *#
				dens, mol, T, nlow = np.hstack((dens[::-1], dens)), np.hstack((mol[::-1], mol)), np.hstack((T[::-1], T)), np.hstack((nlow[::-1], nlow))
				                                             #
				if not GK: PopRat = np.hstack((PopRat[::-1], PopRat))
				                                             #
				vLOS, LOSaxis = np.hstack((-vLOS[::-1], vLOS)), np.hstack((-x[::-1], x))
				                                             #
				if GK:                                       #
					                                     #
					temp = []                            #
					                                     #
					for jm in range (0, len(PopRat[0])): #
						                             #
						lvpd = PopRat[:, jm]         #
						                             #
						temp.append(np.hstack((lvpd[::-1], lvpd)))
						                             #
					temp = np.array(temp)                #
					                                     #
					PopRat = np.moveaxis(temp, 0, 1)     #
					                                     #
					Bx, By, Bz = Bx[len(Bx)//2, :, 0], By[len(By)//2, :, 0], Bz[len(Bz)//2, :, 0]
					                                     #
					Bx, By, Bz = np.hstack((-Bx[::-1], Bx)), np.hstack((By[::-1], By)), np.hstack((Bz[::-1], Bz))
					                                     #
					Blos = Bx                            #
					                                     #
		                                                             #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		#                                                            #
		#  $$ Cartesian case, 3 cases based on projection angle $$   #
		#                                                            #
		#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
		else:                                                        #
			                                                     #
			#----------------------------------------------------#
			#           Integrate along z axis                   #
			#----------------------------------------------------#
			if fi0 == 0.:                                        #
				                                             #
				j, i = dens.shape[0]//2, dens.shape[1]//2    #
				                                             #
				dens, mol, T = dens[j, i, :], mol[j, i, :], T[j, i, :]
				                                             #
				PopRat, nlow = PopRat[j, i, :], nlow[j, i, :]#
				                                             #
				vLOS, LOSaxis = vz[j, i, :], z               #
				                                             #
				if GK:                                       #
					                                     #
					Bx, By, Bz = Bx[j, i, :], By[j, i, :], Bz[j, i, :]
					                                     #
					Blos = Bz                            #
			else:                                                #
				                                             #
				#--------------------------------------------#
				#        Integrate along x axis              #
				#--------------------------------------------#
				if th0 == 0.:                                #
					                                     #
					j, k = dens.shape[0]//2, dens.shape[2]//2
					                                     #
					dens, mol, T = dens[j, :, k], mol[j, :, k], T[j, :, k]
					                                     #
					PopRat, nlow = PopRat[j, :, k], nlow[j, :, k]
					                                     #
					vLOS, LOSaxis = vx[j, :, k], x       #
					                                     #
					if GK:                               #
						                             #
						Bx, By, Bz = Bx[j, :, k], By[j, :, k], Bz[j, :, k]
						                             #
						Blos = Bx                    #
				#--------------------------------------------#
				#        Integrate along y axis              #
				#--------------------------------------------#
				else:                                        #
					                                     #
					i, k = dens.shape[1]//2, dens.shape[2]//2
					                                     #
					dens, mol, T = dens[:, i, k], mol[:, i, k], T[:, i, k]
					                                     #
					PopRat, nlow = PopRat[:, i, k], nlow[:, i, k]
					                                     #
					vLOS, LOSaxis = vy[:, i, k], y       #
					                                     #
					if GK:                               #
						                             #
						Bx, By, Bz = Bx[:, i, k], By[:, i, k], Bz[:, i, k]
						                             #
						Blos = By                    #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                            #
		#           $$$ End of single line == True $$$               #
		#                                                            #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		return dens, mol, vLOS, T, PopRat, nlow, LOSaxis, Bx, By, Bz, Blos
	else:                                                                #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                            #
		#        $$$ line == False, only trully applies to 3D $$$    #
		#                                                            #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		#                                                            #
		if ndim == 2:                                                #
			                                                     #
			vLOS, LOSaxis, RowAxis = vx, x, y                    #
			                                                     #
			if GK:                                               #
				                                             #
				Blos = Bx                                    #
				                                             #
			else:                                                #
				                                             #
				Blos = 0.                                    #
				                                             #
		elif ndim == 3:                                              #
			                                                     #
			#----------------------------------------------------#
			#           Integrate along z axis                   #
			#----------------------------------------------------#
			if fi0 == 0.:                                        #
				                                             #
				LOSaxis, vLOS, RowAxis = z, vz, y            #
				                                             #
				if GK:                                       #
					                                     #
					Blos = Bz                            #
					                                     #
				else:                                        #
					                                     #
					Blos = 0.                            #
					                                     #
			else:                                                #
				                                             #
				#--------------------------------------------#
				#        Integrate along x axis              #
				#--------------------------------------------#
				if th0 == 0.:                                #
					                                     #
					dens, mol, T = np.moveaxis(dens, 1, 2), np.moveaxis(mol, 1, 2), np.moveaxis(T, 1, 2)
					                                     #
					PopRat, nlow, vx = np.moveaxis(PopRat, 1, 2), np.moveaxis(nlow, 1, 2), np.moveaxis(vx, 1, 2)
					                                     #
					LOSaxis, vLOS, RowAxis = x, vx, y    #
					                                     #
					if GK:                               #
						                             #
						Blos =  np.moveaxis(Bx, 1, 2)#
						                             #
					else:                                #
						                             #
						Blos = 0.                    #
						                             #
				#--------------------------------------------#
				#        Integrate along y axis              #
				#--------------------------------------------#
				else:                                        #
					                                     #
					dens, mol, T = np.moveaxis(dens, 0, 2), np.moveaxis(mol, 0, 2), np.moveaxis(T, 0, 2)
					                                     #
					PopRat, nlow, vy = np.moveaxis(PopRat, 0, 2), np.moveaxis(nlow, 0, 2), np.moveaxis(vy, 0, 2)
					                                     #
					LOSaxis, vLOS, RowAxis = y, vy, x    #
					                                     #
					if GK:                               #
						                             #
						Blos =  np.moveaxis(By, 0, 2)#
						                             #
					else:                                #
						                             #
						Blos = 0.                    #
						                             #
	                                                                     #
	return dens, mol, vLOS, T, PopRat, nlow, LOSaxis, RowAxis, Blos      #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
