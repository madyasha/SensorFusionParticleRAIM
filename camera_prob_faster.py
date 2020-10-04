import cv2, time as t, os, math, operator, re, sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import helper as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import cProfile


def camera_prob(nPart, test_filename,test2_filename, state,tt,num_train,Xct,Yct,Zct,names, train_features, test_feature1, test_feature2,tlen) :

#idea is to search in entire training database but truncate based on local time first and everything previously
#return top N matches or probabilities

	N = num_train # Number of samples to take as training set, for time being make it equal to nparticles but should be more
	file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'REAR IMAGES/REAR IMAGES')
	rootDir = file_path
	dirName = file_path
	#print (dirName)
	#------------------------------------------------------------------------------------------------------------					
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
	leafClusterSize = 2*branches
	sind = (tt - num_train)* 4
	eind = sind + 4 *num_train
	#print ('in cprob')
	#print (sind,eind)
	X = Xct[sind: eind]
	Y = Yct[sind: eind]
	Z = Zct[sind: eind]
	fileList= names[sind: eind]
	#print (len(fileList))

	def dumpFeatures(rootDir, features):
		return features


	# Function to construct the vocabulary tree
	def constructTree(node, featuresIDs, depth,tree,nodeIndex,avgDepth):
		tree[node] = []
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
	def tfidf(filename, des):
		print (filename)
		global imagesInLeaves
		for d in des:
			leafID = lookup(d, 0)
			print (leafID)
			if filename in imagesInLeaves[leafID]:
				imagesInLeaves[leafID][filename] += 1
			else:
				imagesInLeaves[leafID][filename] = 1
		#print (imagesInLeaves[leafID][filename])

		del des


	# This function returns the weight of a leaf node
	def weight(leafID):
		return math.log1p(N/1.0*len(imagesInLeaves[leafID]))

	# Returns the scores of the images in the dataset and plots them
	def getScores(q):
		#print ('flen', len(fileList))
		scores = {}		
		for fname in fileList:
			#msg = dirName %s fname % my_var
			img = dirName + "/" + fname
			#print ('getscores',img)
			scores[img] = 0
			#print (doc)
			#print (doc[img])
			#matrix = [[j for j in range(5)] for i in range(5)] 

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
		

	def match(filename,X,Y,Z, test_feature):
		# q is the frequency of this image appearing in each of the leaf nodes
		q = {}
		des = test_feature

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
	features = dumpFeatures(rootDir,train_features)
	root = features.mean(axis = 0)
	nodes[0] = root
	featuresIDs = [x for x in range(len(features))]
	constructTree(0, featuresIDs, 0,tree,0,0)
	del features
	avgDepth = int(avgDepth/len(imagesInLeaves))

	for i in range(len(fileList)):
		filename = dirName + "/" + fileList[i]
		desc = train_features[int(tlen[i,1]):int(tlen[i,1] + tlen[i,0]) ,:]		#print ('final test')
		tfidf(filename, desc)

	#global imagesInLeaves
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

    
	prob_val = match(fileList,X.tolist(),Y.tolist(),Z.tolist(), test_feature1)
	prob_val2= match(fileList,X.tolist(),Y.tolist(),Z.tolist(), test_feature2)
	#cProfile.run('camera_prob')
	#print (test2_filename)
	return prob_val, prob_val2