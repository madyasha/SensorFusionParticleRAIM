#kl divergence computation
import numpy as np
import camera_prob as cp
import pandas as pd
import cv2, time as t, os, math, operator, re, sys
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import math as m

def baseline_last_img(logwt, state, gpstime, npart, ncamera,tt):
	#print ('computing kld_metric')
	test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'REAR IMAGES/REAR IMAGES')
	#print (test_path)
	num_train = 10
	#test image excel file
	test_excel_file= 'camera_test.xlsx'
	df = pd.read_excel(test_excel_file, usecols="A:C")
	name= df['Name']
	alpha= []
	prob= np.zeros((npart,ncamera))
	#factor to scale probabilities by
	factor= 10 ** 2  #10**14
	logwt= logwt/ factor
	#print ('logwt', logwt)
	
	
	#code for two measurements at every
		#index of the test image in the excel file
	ind= tt* ncamera 
	test_filename = test_path + "/" + name[ind]
	test2_filename = test_path + "/" + name[ind+1]
	cprob,cprob2 = cp.camera_prob(npart,test_filename,test2_filename,state, gpstime, tt,num_train)
	prob[:,0]= np.transpose(cprob)
	prob[:,1]= np.transpose(cprob2)
	for i in range(ncamera):
		clogwt = np.log10(prob[:,i])
		clogwt = clogwt 
		#print ('clogwt', clogwt)
		#print ('clogwt', clogwt)
		numr = np.sum(np.multiply (prob[:,i], logwt - clogwt))
		denr= np.sum (prob[:,i])
		alphai= m.exp(np.divide(numr, denr) - 1)
		alpha.append(alphai)

	#normalize the weights
	alpha= np.divide(np.asarray(alpha), np.sum(np.asarray(alpha)))
	# print ('alpha', alpha)
	qstar= np.matmul(prob, np.transpose(alpha))
	logcwt= (np.log(qstar))*factor
	return logcwt











