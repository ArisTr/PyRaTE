#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#  NAME:                                                                     #
#                                                                            #
#  distance.py                                                               #
#                                                                            #
#                                                                            #
#  DESCRIPTION:                                                              #
#                                                                            #
#  Python script for calculating the distance "travelled" by a ray inside    #
#  the grid based on geometry and angle                                      #
#                                                                            #
#  This script will also return the vLOS and BLOS based on projection angle  #
#                                                                            #
#  PARAMETERS:                                                               #
#                                                                            #
#     Input : rayvec, InitPoint, x, y, z, vx, vy, vz, Bx, By, Bz, GK         #
#     Output : dist, vLOS                                                    #
#              BLOS, FPoint                                                  #
#                                                                            #
#  InitPoint: Entry point in the grid/cell                                   #
#  dist: Distance in the cell along the LOS                                  #
#  vLOS, BLOS: velocity and B-field along the LOS                            #
#  Fpoint: Exit point in the cell                                            #
#                                                                            #
#  AUTHOR:                                                                   #
#                                                                            #
#  Aris E. Tritsis                                                           #
#  (aris.tritsis@epfl.ch)                                                    #
#                                                                            #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
import numpy as np                                                           #
import sys                                                                   #
import warnings                                                              #
                                                                             #
warnings.filterwarnings("ignore")                                            #
XZpUnitV = np.array([1., 0., 0.])  # Normal vector to the XZ plane           #
YZpUnitV = np.array([0., 1., 0.])  # Normal vector to the YZ plane           #
YXpUnitV = np.array([0., 0., 1.])  # Normal vector to the YX plane           #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
def distance(rayvec, InitPoint, x, y, z, vx, vy, vz, fcall, Bx, By, Bz, GK): #
	                                                                     #
	dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]                   #
	                                                                     #
	#   $$      Already a unit vector but ensure for any case      $$    #
	rayvec = rayvec/np.sqrt(np.dot(rayvec, rayvec))                      #
	                                                                     #
	exitGrid = False                                                     #
	                                                                     #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                    #
	#   $$ To avoid rounding errors define a tolerance for the min $$    #
	#   $$ distance within a cell...This means that this script is $$    #
	#   $$   accurate to the extend that we don't hit a face of a  $$    #
	#   $$       cell closer than 1e-10*dx to the next x(i)        $$    #
	#   $$                                                         $$    #
	#   $$  !! Note that for super high-res sims we might need to  $$    #
	#   $$  !!    Increase this number to a higher value           $$    #
	                                                                     #
	toly, tolx, tolz = 1e-10*dy, 1e-10*dx, 1e-10*dz                      #
	                                                                     #
	tol = np.sqrt(toly**2+tolx**2+tolz**2)                               #
	                                                                     #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	#                                                                    #
	#   $$ Special care is needed for when we first enter the grid $$    #
	#  $$ In this scenario we may need to properly update InitPoint $$   #
	#  $$ If this is indeed the case, we will find the actual entry $$   #
	#  $$  point based on rayvec and minumum distance with "minus"  $$   #
	#  $$                faces of the cube                          $$   #
	#                                                                    #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	if fcall:                                                            #
		                                                             #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		# $$ Following is to work out where exactly our ray hits  $$ #
		# $$  the cell in order to define the faces of the cell   $$ #
		# $$ If ray hits closer to x[i] then the x-coordinate of  $$ #
		# $$  of the "minus" face will simply be x[i] and the     $$ #
		# $$  "plus" face will be at x[i] + dx. If on the other   $$ #
		# $$  hand it hits closer to x[i+1] then the "minus" face $$ #
		# $$           should be at x[i+1-1] = x[i]               $$ #
		# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
		indy, indx, indz = np.argmin( np.absolute(y - InitPoint[0]) ), np.argmin( np.absolute(x - InitPoint[1]) ), np.argmin( np.absolute(z - InitPoint[2]) )
		                                                             #
		DY, DX, DZ = InitPoint[0] - y[indy], InitPoint[1] - x[indx], InitPoint[2] - z[indz]
		                                                             #
		if np.absolute(DY) < toly: DY = 0.                           #
		                                                             #
		if np.absolute(DX) < tolx: DX = 0.                           #
		                                                             #
		if np.absolute(DZ) < tolz: DZ = 0.                           #
		                                                             #
		j = indy if DY >= 0. else indy-1                             #
		                                                             #
		i = indx if DY >= 0. else indx-1                             #
		                                                             #
		k = indz if DZ >= 0. else indz-1                             #
		                                                             #
		#    $$  Points in minus and "plus" faces of the cell  $$    #
		PointMinusFace = np.array([y[j], x[i], z[k]]) - np.array([dy/2., dx/2., dz/2.])
		                                                             #
		temp = []                                                    #
		                                                             #
		for vec in [XZpUnitV, YZpUnitV, YXpUnitV]:                   #
			                                                     #
			temp.append( -(np.dot(vec, InitPoint - PointMinusFace)) / np.dot(vec, rayvec) )
			                                                     #
		temp = np.array(temp)                                        #
		                                                             #
		dists, points = [], []                                       #
		                                                             #
		for t in range (len(temp)):                                  #
			                                                     #
			p = InitPoint + temp[t] * rayvec                     #
			                                                     #
			dist = np.sqrt( (p[0] - InitPoint[0])**2 + (p[1] - InitPoint[1])**2 + (p[2] - InitPoint[2])**2)
			                                                     #
			dists.append(dist); points.append(p)                 #
			                                                     #
		dists, points = np.array(dists), np.array(points)            #
		                                                             #
		points, dists = points[np.isfinite(dists)], dists[np.isfinite(dists)]
		                                                             #
		InitPoint = points[np.argmin(dists)]                         #
		                                                             #
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
	else:                                                                #
		#  $$ If not on "first call" we already are on a cell's $$   #
		#  $$ face but still need to figure out where exactly   $$   #
		cellfaces = np.array([y - dy/2., x - dx/2., z - dz/2.])      #
		                                                             #
		j = np.argmin(np.absolute(np.round(cellfaces[0] - InitPoint[0], -10)))
		                                                             #
		i = np.argmin(np.absolute(np.round(cellfaces[1] - InitPoint[1], -10)))
		                                                             #
		k = np.argmin(np.absolute(np.round(cellfaces[2] - InitPoint[2], -10)))
		                                                             #
		PointMinusFace = np.array([cellfaces[0][j], cellfaces[1][i], cellfaces[2][k]])
		                                                             #
	PointPlusFace = PointMinusFace + np.array([dy, dx, dz])              #
	                                                                     #
	#    $$   Find value of at the point of intersection   $$$           #
	                                                                     #
	temp = []                                                            #
	                                                                     #
	for vec in [XZpUnitV, YZpUnitV, YXpUnitV]:                           #
		                                                             #
		temp.append( -(np.dot(vec, InitPoint - PointPlusFace)) / np.dot(vec, rayvec) )
		                                                             #
	temp = np.array(temp)                                                #
	                                                                     #
	#   $$ Find the coordinates of the point of intersection $$          #
	dists, points = [], []                                               #
	                                                                     #
	for t in range (len(temp)):                                          #
		                                                             #
		p = InitPoint + temp[t] * rayvec                             #
		                                                             #
		dist = np.sqrt( (p[0] - InitPoint[0])**2 + (p[1] - InitPoint[1])**2 + (p[2] - InitPoint[2])**2)
		                                                             #
		dists.append(dist); points.append(p)                         #
	                                                                     #
	dists, points = np.array(dists), np.array(points)                    #
	                                                                     #
	points, dists = points[np.isfinite(dists)], dists[np.isfinite(dists)]#
	                                                                     #
	points, dists = points[dists > tol], dists[dists > tol]              #
	                                                                     #
	FPoint, dist = points[np.argmin(dists)], dists[np.argmin(dists)]     #
	                                                                     #
	if FPoint[0] > np.max(y)+dy/2. or FPoint[1] > np.max(x)+dx/2. or FPoint[2] > np.max(z)+dz/2.:
		                                                             #
		exitGrid = True                                              #
		                                                             #
	vVec = np.array([vy[j, i, k], vx[j, i, k], vz[j, i, k]])             #
	                                                                     #
	#  $$   Perform a vector projection of vVec onto unitvector   $$     #
	vmLOS = np.dot(vVec, rayvec)                                         #
	                                                                     #
	vLOS = vmLOS * rayvec                                                #
	                                                                     #
	pnangle = np.round(np.degrees(np.arccos( np.round( np.dot(vLOS, rayvec)/ vmLOS, 2) ) ))
	                                                                     #
	if pnangle == 0.:                                                    #
		                                                             #
		vLOS = vmLOS                                                 #
		                                                             #
	elif pnangle == 180.:                                                #
		                                                             #
		vLOS = -vmLOS                                                #
		                                                             #
	else:                                                                #
		                                                             #
		raise SystemExit("Don't know if vLOS is positive or negative!")
	if GK:                                                               #
		                                                             #
		BVec = np.array([By[j, i, k], Bx[j, i, k], Bz[j, i, k]])     #
		                                                             #
		BLOS = np.dot(vVec, rayvec) * rayvec                         #
		                                                             #
		BmLOS = np.sqrt(np.dot(BLOS, BLOS))                          #
		                                                             #
		pnangle = np.round(np.degrees(np.arccos(np.dot(BLOS, rayvec)/ BmLOS ) ))
		                                                             #
		if pnangle == 0.:                                            #
			                                                     #
			BLOS = BmLOS                                         #
			                                                     #
		elif pnangle == 180.:                                        #
			                                                     #
			BLOS = -BmLOS                                        #
			                                                     #
		else:                                                        #
			                                                     #
			raise SystemExit("Don't know if BLOS is positive or negative!")
			                                                     #
	else:                                                                #
		                                                             #
		BLOS = None                                                  #
	                                                                     #
	return dist, FPoint, j, i, k, exitGrid, vLOS, BLOS                   #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
