#new idea to speed things up
#precompute coord array and store it in an array (not excel file) as well as a flag for threshold if you want to store that time instant
import cv2, time as t, os, math, operator, re, sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import helper as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import camera_prob_new as cp


test_path = 'C:/Users/JNM/Downloads/ParticleRAIM/REAR IMAGES/BLURRED IMAGES'
train_excel_file= 'camera_train.xlsx'
test_excel_file= 'camera_test.xlsx'
df = pd.read_excel(test_excel_file, usecols="A:C")
names_test = df['Name']
test_names = names_test.values.tolist()

df = pd.read_excel(train_excel_file, usecols="A:D")
Xct = df['X']
Yct = df['Y']
Zct = df['Z']
names= df['Name']
names= names.values.tolist()
T = 3440
ncamera = 2
num_train = 3
arr = np.zeros((3440,4))


for tt in range(T):
	if tt> 2:
		print (tt)
		ind= tt* ncamera 
		test_filename = test_path + "/" + test_names[ind]
		test2_filename = test_path + "/" + test_names[ind+1]
		(ind1,val1,ind2,val2) = cp.camera_prob(test_filename,test2_filename, tt,num_train,Xct,Yct,Zct,names)
		#print (ind1,val1,ind2,val2)
		arr[tt,0] = int(ind1)
		arr[tt,1] = int(val1)
		arr[tt,2] = int(ind2)
		arr[tt,3] = int(val2)

np.save('train_data_blur_new.npy',arr)
#np.savetxt('train_data.csv',arr,delimiter= ',')