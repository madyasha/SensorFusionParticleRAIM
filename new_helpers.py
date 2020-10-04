#kl divergence computation
import numpy as np
import new_camera as cp
import pandas as pd
import cv2, time as t, os, math, operator, re, sys
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import math as m
import helper as hp
from scipy.stats import entropy

def kld_metric(logwt, state, npart, ncamera,tt,Xct,Yct, Zct,train_arr, bias):
	#print ('computing kld_metric')
	prob = np.zeros((npart,2))
	alpha = []
	k = sum(logwt) / len(logwt)
	factor = 10 ** (len (str (int(abs(k // 10)))))
	logwt = logwt/ factor
	logwt = logwt * 10
	num_train = 3
	sind= (tt- num_train)*4
	eind = sind + 4 *num_train
	X = np.zeros((12,1))
	Y = np.zeros((12,1))
	Z = np.zeros((12,1))
	X[:,0] = Xct[sind: eind]
	Y[:,0] = Yct[sind: eind]
	Z[:,0] = Zct[sind: eind]
	sel = np.random.randint(1)
	cprob,cprob2 = cp.camera_prob(npart,state,tt,X,Y,Z,train_arr, bias)
	prob[:,0]= np.transpose(cprob)
	prob[:,1]= np.transpose(cprob2)
	for i in range(ncamera):
		clogwt = (np.log(prob[:,i])) 
		term = KL_help (prob[:,i], logwt, clogwt)
		alphai= m.exp(term - 1)
		#alphai= 10 ** (term - 1)
		alpha.append(alphai)


	#normalize the weights
	alpha= np.divide(np.asarray(alpha), np.sum(np.asarray(alpha)))
	#print (alpha)
	qstar= np.matmul(prob, np.transpose(alpha)) 
	#logcwt= np.log(qstar) 
	logcwt= np.log(qstar) * (factor/(10 ** 2))

	return logcwt

def KL_help (prob, logwt, logcwt):
	return np.sum(np.where(prob != 0, prob * (logwt- logcwt), 0))


def baseline_metric (logwt, state, npart, tt, Xct, Yct, Zct, train_arr, bias):
	k = sum(logwt) / len(logwt)
	#print (logwt)
	factor = 10 ** (len (str (int(abs(k // 10)))))
	num_train = 3
	sind= (tt- num_train)*4
	eind = sind + 4 *num_train
	X = np.zeros((12,1))
	Y = np.zeros((12,1))
	Z = np.zeros((12,1))
	X[:,0] = Xct[sind: eind]
	Y[:,0] = Yct[sind: eind]
	Z[:,0] = Zct[sind: eind]
	prob1,prob2 = cp.camera_prob(npart,state,tt,X,Y,Z,train_arr, bias)
	qstar = prob1 * prob2
	logcwt = (np.log(qstar)) * (factor/10)
	# print (logwt)
	# print (logcwt)
	return logcwt



def baseline_metric_naive (logwt,state, npart, t, X, Y, Z, train_arr, bias):
	k = sum(logwt) / len(logwt)
	factor = 10 ** (len (str (int(abs(k // 10)))))
	ind = ((t-2)* 4) - 3 
	coord_array = np.tile ([X [ind], Y[ind], Z[ind]],(npart, 1))
	if np.random.random()> 0.2:
		bias = 10
		xc = xc + bias
		yc = yc + bias
		zc = zc + bias
	dist = np.linalg.norm(np.subtract(state, coord_array), ord= 2, axis=1)
	prob1 = hp.prob_out_softmax_negative(np.asarray(dist))
	logcwt = np.log(prob1) * (factor/10)
	return logcwt







