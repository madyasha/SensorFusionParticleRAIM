import numpy as np

def prob_out_softmax(x):
	return (np.exp(x)/np.sum(np.exp(x), axis=0))

def prob_out_softmax_negative(x):
	return (np.exp(-x)/np.sum(np.exp(-x), axis=0))

def spherical_softmax(x):
	return (np.square(-x)/np.sum(np.square(-x), axis=0))

def spherical_softmax_stab(x, eps):
	return ((np.square(-x)+ eps)/((eps*(-x).shape[0])+ np.sum(np.square(-x), axis=0)))

def taylor_softmax(x):
	numr= 1 + (-x) + 0.5* np.square(-x)
	denr= x.shape[0]+ np.sum(0.5 * np.square(-x), axis=0) + np.sum(-x,axis=0)
	return (numr/ denr)

def rbf_kernel(x, width):
	xs= np.square(x)
	ws= width**2
	return (np.exp(-(np.true_divide(xs,2*ws)))/np.sum(np.exp(-(np.true_divide(xs,2*ws))), axis=0))

