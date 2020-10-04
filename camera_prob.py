import cv2, time as t, os, math, operator, re, sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import helper as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os


def camera_prob(nPart, test_filename,test2_filename, state,tt,num_train,Xct,Yct,Zct,names) :

#idea is to search in entire training database but truncate based on local time first and everything previously
#return top N matches or probabilities

	N = num_train # Number of sample
	file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'REAR IMAGES/REAR IMAGES')
	#------------------------------------------------------------------------------------------------------------					
	rootDir = file_path		
	global nodeIndex, nodes, tree, imagesInLeaves, avgDepth				# The root directory of the dataset
	nodes = {}								# List of nodes (list of SIFT descriptors)
	nodeIndex = 0							# Index of the last node for which subtree was constructed
	tree = {}								# A dictionary in the format - node: [child1, child2, ..]
	branches = 5							# The branching factor in the vocabulary tree
	leafClusterSize = 20					# Minimum size of the leaf cluster
	imagesInLeaves = {}						# Dictionary in the format - leafID: [img1:freq, img2:freq, ..]
	doc = {}								# 							#		#
	maxDepth = 5
	avgDepth = 0
	model =  MiniBatchKMeans(n_clusters=branches)	# The KMeans Clustering Model
	sift = cv2.ORB_create()
	leafClusterSize = 2*branches
	sind= (tt- num_train)*4
	eind = sind + 4 *num_train
	X = Xct[sind: eind]
	Y = Yct[sind: eind]
	Z = Zct[sind: eind]
	fileList= names[sind:eind]
	dirName = file_path


	# Function to dump all the SIFT descriptors from training data in the feature space
	def dumpFeatures(rootDir):
		features = []
		for fname in fileList:
			#print (dirName + "/" + fname)
			img = cv2.imread(dirName + "/" + fname)
			kp = sift.detect(img, None)
			kp,des = sift.compute(img, kp)
			#print (des)
			#kp, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread(dirName + "/" + fname), cv2.COLOR_BGR2GRAY), None)
			for d in des:
				features.append(d)
			del kp, des
		features = np.array(features)
		return features

	# Function to construct the vocabulary tree
	def constructTree(node, featuresIDs, depth,tree,nodeIndex,avgDepth):
		tree[node] = []
		"""
		tree = {}	
		tree[node] = []
		imagesInLeaves = {}	
		nodeIndex= 0
		avgDepth = 0
		"""
		if len(featuresIDs) >= leafClusterSize and depth < maxDepth :
			# Here we will fetch the cluster from the indices and then use it to fit the kmeans
			# And then just after that we will delete the cluster
			# Using the array of indices instead of cluster themselves will reduce the memory usage by 128 times :)
			model.fit([features[i] for i in featuresIDs])
			childFeatureIDs = [[] for i in range(branches)]
			for i in range(len(featuresIDs)):
				childFeatureIDs[model.labels_[i]].append(featuresIDs[i])
			for i in range(branches):
				nodeIndex = nodeIndex + 1
				nodes[nodeIndex] = model.cluster_centers_[i]
				tree[node].append(nodeIndex)
				constructTree(nodeIndex, childFeatureIDs[i], depth + 1,tree,nodeIndex,avgDepth)
		else:
			imagesInLeaves[node] = {}
			avgDepth = avgDepth + depth

	# Function to lookup a SIFT descriptor in the vocabulary tree, returns a leaf cluster
	def lookup(descriptor, node):
		D = float("inf")
		goto = None
		for child in tree[node]:
			dist = np.linalg.norm([nodes[child] - descriptor])
			if D > dist:
				D = dist
				goto = child
		if tree[goto] == []:
			return goto
		return lookup(descriptor, goto)	

	# Constructs the inverted file frequency index
	def tfidf(filename):
		global imagesInLeaves
		kp, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread(dirName + "/" + fname), cv2.COLOR_BGR2GRAY), None)
		for d in des:
			leafID = lookup(d, 0)
			if filename in imagesInLeaves[leafID]:
				imagesInLeaves[leafID][filename] += 1
			else:
				imagesInLeaves[leafID][filename] = 1
		del kp, des

	# This function returns the weight of a leaf node
	def weight(leafID):
		return math.log1p(N/1.0*len(imagesInLeaves[leafID]))

	# Returns the scores of the images in the dataset and plots them
	def getScores(q):
		scores = {}
		
		for fname in fileList:
			img = dirName + "/" + fname

			scores[img] = 0
			for leafID in imagesInLeaves:
				if leafID in doc[img] and leafID in q:
					scores[img] += math.fabs(q[leafID] - doc[img][leafID])
				elif leafID in q and leafID not in doc[img]:
					scores[img] += math.fabs(q[leafID])
				elif leafID not in q and leafID in doc[img]:
					scores[img] += math.fabs(doc[img][leafID])
				
		index = findBest(scores)
		return index


	def findBest(scores):
		scores_f=[]
		for ind,s in enumerate(scores.items()):
			scores_f.append(s[1])
		#find index of image with max score
		#print ('scores',scores_f)
		index = np.argsort(-1* np.asarray(scores_f))
		return index[0]
		

	def match(filename,X,Y,Z):
		# q is the frequency of this image appearing in each of the leaf nodes
		q = {}
		kp, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), None)
		for d in des:
			leafID = lookup(d, 0)
			if leafID in q:
				q[leafID] += 1
			else:
				q[leafID] = 1
		s = 0.0
		for key in q:
			q[key] = q[key]*weight(key)
			s += q[key]
		for key in q:
			q[key] = q[key]/s
		
		ind_val = getScores(q)
		#print ('ind', ind_val)
		xc= X[ind_val]
		yc= Y[ind_val]
		zc= Z[ind_val]
		coord_array = np.tile([xc,yc,zc],(nPart,1))
		#print (coord_array)
		dist = np.linalg.norm(np.subtract(state,coord_array), ord=2, axis=1)	
		prob= hp.prob_out_softmax_negative(np.asarray(dist))
		return prob

	#------------------------------------------------------------------------------------------------------------
	features = dumpFeatures(rootDir)
	root = features.mean(axis = 0)
	nodes[0] = root
	featuresIDs = [x for x in range(len(features))]
	constructTree(0, featuresIDs, 0,tree,0,0)
	del features

	avgDepth = int(avgDepth/len(imagesInLeaves))

	for fname in fileList:
		filename = dirName + "/" + fname
		tfidf(filename)

	#
	for leafID in imagesInLeaves:
		for img in imagesInLeaves[leafID]:
			if img not in doc:
				doc[img] = {}
			doc[img][leafID] = weight(leafID)*(imagesInLeaves[leafID][img])
	for img in doc:
		s = 0.0
		for leafID in doc[img]:
			s += doc[img][leafID]
		for leafID in doc[img]:
			doc[img][leafID] /= s

    
	prob_val = match(test_filename,X.tolist(),Y.tolist(),Z.tolist())
	#print ('test', test_filename)
	prob_val2= match(test2_filename,X.tolist(),Y.tolist(),Z.tolist())
	#print ('test2', test2_filename)
	return prob_val, prob_val2