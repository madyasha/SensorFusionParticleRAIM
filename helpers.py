########################################################################
# Particle Metropolis-Hastings using gradient and Hessian information
# Copyright (c) 2014 Johan Dahlin ( johan.dahlin (at) liu.se )
#
# helpers.py
# Different helper files
#
########################################################################

import numpy as np
import os
import math
from scipy import stats
pi = math.pi

#############################################################################################################################
# Calculate the Integrated Autocorrlation Time (disabled)
#############################################################################################################################   
def IACT(x):
    return 1.0;

#############################################################################################################################
# Calculate the Squared Jump distance
#############################################################################################################################   
def SJD(x):
    tmp = np.diff( x ) ** 2;
    out = np.sum( tmp );

    return out / ( len(x) - 1.0 );

#############################################################################################################################
# Calculate the log-pdf of a univariate Gaussian
#############################################################################################################################   
def loguninormpdf(x,mu,sigma):
    return [-0.5 * np.log( 2.0 * np.pi * sigma[i]**2) - 0.5 * (x-mu[i])**2 * sigma[i]**(-2) for i in range(len(mu))];

#############################################################################################################################
# Calculate the log-pdf of a multivariate Gaussian with largest residual removal with mean vector mu and covariance matrix S
#############################################################################################################################   
def lognormpdf_RAIM(x,mu,S):
    nx = len(S)
    norm_coeff = nx * np.log( 2.0 * np.pi ) + np.linalg.slogdet(S)[1]
    err = x-mu
    merr = np.mean(err, axis=0)
    err[:, merr.argmax()] = 0
    numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
    return -0.5*(norm_coeff+np.diag(numerator))


def lognormpdf(x,mu,S):
    nx = len(S)
    norm_coeff = nx * np.log( 2.0 * np.pi ) + np.linalg.slogdet(S)[1]
    err = x-mu

    numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
    return -0.5*(norm_coeff+np.diag(numerator))
#############################################################################################################################
# Calculate the flatenned log-pdf of a multivariate Gaussian with mean vector mu and covariance matrix S
#############################################################################################################################   
# def logfnormpdf(x,mu,S):
#     nx = len(S)
#     sel = np.tile(np.eye(nx), (int(len(mu)/nx),1))
#     norm_coeff = np.tile(np.sum(sel[0,:])*(np.log( 2.0 * np.pi ) + np.diag(S)),int(len(mu)/nx))
#     err = x-mu
#     err = np.multiply(err,sel)
#     numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
#     return -0.5*(norm_coeff+np.diag(numerator))

#############################################################################################################################
# Calculate the updated weights of log-pdf using RAIM based fault probabilities
#############################################################################################################################   
# def logRAIMwt(w, logwt, nx):
#     sel= np.tile(np.eye(nx), (1, int(len(w)/nx)))
#     pi = sel @ w
#     pi = pi + 1e-10
#     pi /= sum(pi)
#     logpi = np.tile(np.log(pi), int(len(w)/nx))

#     return logwt + logpi

##############################################################################################
# Check if a matrix is positive semi-definite but checking for negative eigenvalues
#############################################################################################################################   
def isPSD(x):
    return np.all(np.linalg.eigvals(x) > 0)

#############################################################################################################################
# Print verbose progress reports from sampler  [TODO] Debug this!
#############################################################################################################################   
# def verboseProgressPrint(kk,par,thp,th,aprob,accept,step,scoreP,score,infom,infomP,v):
#     print("Reminder verboseProgressPrint: not debugged yet...")
    
#     print("===========================================================================================")
#     print("Iteration: " + str(kk) + " of " + str(par.nMCMC) + " complete.");
#     print("===========================================================================================")
#     print("Proposed parameters: " + str(thp) + " and current parameters: " + str(th) + ".");
#     print("Acceptance probability: " + str(aprob) + " with outcome: " + str(accept) + ".");

#     if (v==1) :
#         print("Scaled score vector for proposed: " + str(step**2*0.5*scoreP) + " and curret: " + str(step**2*0.5*score) );

#     if (v==2):
#         print("Scaled score vector for proposed: " + str(step**2*0.5*scoreP/infomP) + " and current: " + str(step**2*0.5*score/infom) );
#         print("Step size squared for proposed: " + str(step**2/infomP) + " and current: " + str(step**2/infom) );

#     print("");


#############################################################################################################################
# Check if dirs for outputs exists, otherwise create them
#############################################################################################################################   
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    
#############################################################################################################################
# Generate points in a circle
#############################################################################################################################   
def PointsInCircum(r, n=100):
    return [(math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r) for x in range(0, n+1)]

#############################################################################################################################
# Integrate over 2D Gaussian
#############################################################################################################################   

def gauss_int(gaus, a, b):
    ngaus = lambda x,y: gaus([x,y])
    return dblquad(ngaus, a[0], b[0], a[1], b[1])

#############################################################################################################################
# Fit 2D Gaussian
#############################################################################################################################   

def _gaussian(x, mu, S):
    err = x-mu
    numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
    h = 1/np.sqrt(2*np.pi*np.linalg.det(S))
    return h*np.exp(-0.5*numerator)

def gaussian(mu_x, mu_y, S_11, S_12, S_22):
    """Returns a gaussian function with the given parameters"""

    S = np.array([
        [S_11**2, S_12],
        [0, S_22**2]
    ])
    mu = np.array(
        [mu_x, mu_y]
    )
    return lambda x: _gaussian(np.array(x), mu, S)

def moments(p, w):
    """Returns (height, mu_x, mu_y, S_11, S_12, S_22)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    
    height = w.max()
    wsum = np.sum(w)
    p_x = p[:,0]
    p_y = p[:,1]
    p_z = p[:,2]
    mu_x = np.average(p_x, weights=w)
    mu_y = np.average(p_y, weights=w)
    mu_z = np.average(p_z, weights=w)
    S_11 = np.sqrt(np.average((p_x-mu_x)**2, weights=w))
    S_12 = np.average((p_x-mu_x)*(p_y-mu_y), weights=w)
    S_13 = np.average((p_x-mu_x)*(p_z-mu_z), weights=w)
    S_22 = np.sqrt(np.average((p_y-mu_y)**2, weights=w))
    S_23 = np.average((p_y-mu_y)*(p_z-mu_z), weights=w)
    S_33 = np.sqrt(np.average((p_z-mu_z)**2, weights=w))

    return mu_x, mu_y, mu_z,S_11, S_12, S_13,S_22, S_23,S_33

def _errorf(par, p, w):
    gaus = gaussian(*par)
    pred = [gaus(it) for it in p]
    return((pred-w)**2)

def fitgaussian(p, w):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(p, w)
    # print(params)
    errorfunction = lambda par: _errorf(par, p, w)
    p, success = optimize.leastsq(errorfunction, params)
    return p

#############################################################################################################################
# KL Divergence KL(Q||P)
#############################################################################################################################   

def comp_KL(w1, p1, w2, p2):
    #think this might be 2D, change to 3D
    mu_x, mu_y, mu_z,S_11, S_12, S_13,S_22, S_23,S_33 = moments(p1, w1)
    
    S_a = np.array([
        [S_11**2, S_12, S_13],
        [S_12, S_22**2, S_23],
        [S_13,S_23,S_33**2]
    ])
    mu_a = np.array(
        [mu_x, mu_y, mu_z]
    )

    mu_x, mu_y, mu_z,S_11, S_12, S_13,S_22, S_23,S_33 = moments(p2, w2)
    
    S_b = np.array([
        [S_11**2, S_12, S_13],
        [S_12, S_22**2, S_23],
        [S_13,S_23,S_33**2]
    ])
    mu_b = np.array(
        [mu_x, mu_y,mu_z]
    )
    err = mu_a - mu_b
    hamming = np.dot(np.dot(err,np.linalg.pinv(S_a)),err.transpose())
    div = np.trace(np.dot(np.linalg.pinv(S_a),S_b))
    ll = np.log(np.linalg.det(S_a)/np.linalg.det(S_b)) - 2
    return 0.5*(div + hamming + ll)

#############################################################################################################################
# Fit 2D KDE
#############################################################################################################################   

def resampleSimple(w, p):
        H = 120
        u = float(np.random.uniform())
        w = np.cumsum(w) / sum(w)
        Ind = np.empty(H, dtype='int')
        j = 0
        for k in range(H):
            uu = (u + k) / H
            while w[j] < uu and j < H-1:
                j += 1
            Ind[k] = j
        part = p[Ind, :]
        return part + np.random.randn(*part.shape)

def fit_kde(w, p):
        p = resampleSimple(w, p)
        p_ls = np.transpose(p)
        try:
            #kern = stats.gaussian_kde(p_ls)
            kern = stats.gaussian_kde(p_ls)
        except:
            kern = None
        return kern

#############################################################################################################################
# Save distribution plots
#############################################################################################################################   

# def saveDebug(kern, kern_aux, sys, samp_k):
#     fig, ax = plt.subplots()
#     for kaux in kern_aux:
#         pts = kaux.resample(200)
#         ax = sns.kdeplot(pts[0, :], pts[1, :], shade=True,
#                          shade_lowest=False, legend=True, cbar=True)
#     pts = kern.resample(200)
#     ax = sns.kdeplot(pts[0, :], pts[1, :], legend=True, cbar=True)
#     # plt.scatter(samp_k[:,0], samp_k[:,1])
#     # plt.legend(labels=['u\'_1', 'u\'_2', 'u\'_3', 'u\'_4', 'u_true'])
#     plt.savefig(os.path.join(sys.dbg_dir, "db.png"))

#############################################################################################################################
# Integrity bound helpers
#############################################################################################################################   
#integrity check this
def compute_eps(lda,kl, delta, N):
    return lda/N*(kl + np.log((N+1)/delta))
  
def inv_bernoulli_div(q, p):
    q = np.array(q)
    p = np.array(p)
    denom = 1./q + 1./(1-q)
    return np.sqrt(2*p/denom)
#############################################################################################################################
# SHUBH CODE
#############################################################################################################################  
def logRAIMwt(raimwt, logwt, nx):
    sel = np.tile(np.eye(nx), (1, int(len(raimwt)/nx)))
    pi = sel @ np.exp(raimwt)
    pi = pi + 1e-10
    pi /= sum(pi)
    #pi = np.ones(len(pi))
    logpi = np.tile(np.log(pi), int(len(raimwt)/nx))
    return logwt + logpi, pi
##############################
def logfnormpdf(x,mu,S):
    nx = len(S)
    sel = np.tile(np.eye(nx), (int(len(mu)/nx),1))
    norm_coeff = np.tile(np.sum(sel[0,:])*(np.log( 2.0 * np.pi ) + np.diag(S)),int(len(mu)/nx))
    err = x-mu
    err = np.multiply(err,sel)
    num0 = np.dot( np.dot(err,np.linalg.inv(S)),err.transpose())
    num0 = num0 + np.log(num0)
    numerator = np.dot( np.dot(err,np.linalg.inv(S)),err.transpose())
    return -0.5*(norm_coeff+np.diag(num0)), -0.5*(norm_coeff+np.diag(numerator))