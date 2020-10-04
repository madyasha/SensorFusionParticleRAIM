import pandas as pd
import helper as hp
import numpy as np

def camera_prob(nPart,state,tt,X,Y,Z, arr,bias):

#extract the local index from given Xct, Yct and Zct

	#access the index from train_arr
	ind = []
	val = []
	ind.append (arr[tt,0])
	val.append (arr[tt,1])
	ind.append(arr[tt,2])
	val.append(arr[tt,3])
	prob = np.zeros((nPart,2))	
	n = np.random.randint(0,2)
	

	for k in range(2):
		ind_val = int(ind[k])
		xc= (X[ind_val])
		yc= (Y[ind_val])
		zc= (Z[ind_val])
		#print (xc,yc,zc)

		# if tt % 2 == 0 and k == n:
		# 	bias = 100
		# 	xc = xc + bias
		# 	yc = yc + bias
		# 	zc = zc + bias

		#old way
		# if np.random.random()> 0.2:
		# 	if k == n:
		# 		bias = 10
		# 	else:
		# 		bias = -10
		# 	xc = xc + bias
		# 	yc = yc + bias
		# 	zc = zc + bias

		xarr = np.transpose(np.array([xc,yc,zc]))
		coord_array = np.tile(xarr,(nPart,1))
		dist = np.linalg.norm(np.subtract(state,coord_array), ord=2, axis=1)
		prob[:,k] = hp.prob_out_softmax_negative(np.asarray(dist))

	
	return (prob[:,0], prob[:,1])

