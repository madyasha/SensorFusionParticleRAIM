########################################################################
# Particle Metropolis-Hastings using gradient and Hessian information
# Copyright (c) 2014 Johan Dahlin ( johan.dahlin (at) liu.se )
#
# smc.py
# Sequential Monte Carlo samplers
#
########################################################################

import logging
import new_helpers as nhp
import baseline_last_img as base
import seaborn as sns
import _thread
import numpy as np
import pandas as pd
from helpers import *
import matplotlib.pyplot as plt
import matplotlib
import itertools
from scipy import stats
from IPython import display
import dask.dataframe as dd
import time
import math
import random
pi = math.pi
sns.set(color_codes=True)

class smcSampler(object):

    #############################################################################################################################
    # Initalisation
    #############################################################################################################################
    score = []
    infom = []
    xhatf = []
    xhatp = []
    xhats = []
    ll = []
    w = []
    a = []
    p = []
    v = []
    infom1 = []
    infom2 = []
    infom3 = []
    integ = []
    klinteg = []
    error = []

    #############################################################################################################################
    # Default settings
    #############################################################################################################################
    
    #maybe change the following part
    nPart = 120 #num prange meas * T, 95
    Po = 0
    xo = np.array([4051761,617191,4870822])
    so = np.array([4051761,617191,4870822])
    resampFactor = 100
    onlydiagInfo = 1
    makeInfoPSD = 0
    #mode = "pf"
    mode= "praim"
    mode2 = 'pcraim'
    #mode2 = 'b1'
    resamplingType = "residual"
    filterType = "bootstrap"
    smootherType = "filtersmoother"
    plot_particles = False
    cmap = matplotlib.cm.get_cmap('viridis')

    #############################################################################################################################
    # Particle filtering: bootstrap particle filter
    #############################################################################################################################
    def bPF(self, data, sys, par, fig):
        ct = 10
        self.xo = sys.true[0]
        self.so = sys.true[0]
        self.res = np.zeros((sys.T, sys.num_gnss))
        a = np.zeros((self.nPart, sys.T))
        s = np.zeros((self.nPart, 3, sys.T))
        p = np.zeros((self.nPart, 3, sys.T))
        p_aux = np.zeros((self.nPart, 3, int(sys.par[5])))
        w = np.zeros((self.nPart, sys.T))
        w_b = np.zeros((self.nPart, int(sys.par[5])))
        xh = np.zeros((sys.T, 3))
        xh_b = np.zeros((sys.T, 3))
        sh = np.zeros((sys.T, 3))
        llp = 0.0
        clist = []
        klist = []
        elist = []
        integ_step = 0
        p[:, :, 0] = self.xo + 1.0*np.random.randn(self.nPart, 3)
        s[:, :, 0] = self.so

        train_excel_file= 'camera_train.xlsx'
        df = pd.read_excel(train_excel_file, usecols="A:D")
        self.Xct = df['X']
        self.Yct = df['Y']
        self.Zct = df['Z']


        u_odo = sys.ubar/ 0.2
        self.train_arr = np.load('train_data_blur.npy')

        for tt in range(0, sys.T):
            # print ('syst', sys.T)
            print ('tt', tt)
            if self.mode == "praim" and tt> 760:
                integ_step = tt % sys.par[7] == 1 and tt > sys.par[7]

            if tt != 0:
                # Resample (if needed by ESS criteria)
                if ((np.sum(w[:, tt-1]**2))**(-1) < (self.nPart * self.resampFactor)):
                    nIdx = self.resampleResidual(w[:, tt-1], par)
                    nIdx = np.transpose(nIdx.astype(int))
                else:
                    nIdx = np.arange(0, self.nPart)

                # Propagate
                s[:, :, tt] = sys.h(
                    p[nIdx, :, tt-1], data.u[tt-1], s[nIdx, :, tt-1], tt-1)
                p[:, :, tt] = sys.f(p[nIdx, :, tt-1], data.u[tt-1], s[:, :, tt], data.y[tt-1], tt-1) + \
                    np.random.randn(self.nPart, 3) @ sys.fn(
                        p[nIdx, :, tt-1], s[:, :, tt], data.y[tt-1], tt-1)    
                


                if integ_step:               
                    for aux_i in range(int(sys.par[5])):
                        #au = u_odo[tt-1,:] + np.random.randn(1)* 10**(1)
                        au =np.ones_like(p[nIdx, :, tt-1])
                        au[:,0:2] = PointsInCircum(1, n= self.nPart -1)
                        p_aux[:, :, aux_i] = sys.f(p[nIdx, :, tt-1], au, s[:, :, tt], data.y[tt-1], tt-1) + np.random.randn(
                            self.nPart, 3) @ sys.fn(p[nIdx, :, tt-1], s[:, :, tt], data.y[tt-1], tt-1)
                a[:, tt] = nIdx

            # Calculate weights
            if self.mode == "praim":
                raimwt, logwt = logfnormpdf(data.y[tt], sys.g(
                    p[:, :, tt], data.u[tt], s[:, :, tt], tt), sys.gn(p[:, :, tt], s[:, :, tt], tt))                        
                #logwt = np.ones_like(logwt)
            wmax = np.max(logwt)
            w[:, tt] = np.exp((logwt - wmax)) 


            # Calculate RAIM weights
            if self.mode == "praim":                
                logwt, pi = logRAIMwt(raimwt, logwt, sys.num_gnss);



                if tt> 2 and tt % 1 == 0:                  
                    if self.mode2 == 'pcraim':
                        logcwt= nhp.kld_metric (logwt, p[:, :, tt],self.nPart, sys.num_cam,tt*sys.skip_num,self.Xct,self.Yct, self.Zct,self.train_arr, sys.par[4])
                        logwt= logwt + logcwt 
                    elif self.mode2 == 'b1':
                        logcwt = nhp.baseline_metric (logwt, p[:, :, tt], self.nPart, tt, self.Xct,self.Yct, self.Zct, self.train_arr, sys.par[4])
                        logwt= logwt + logcwt
                    elif self.mode2 == 'b2':
                        logcwt = nhp.baseline_metric_naive(logwt, p[:, :, tt], self.nPart, tt, self.Xct,self.Yct, self.Zct, self.train_arr,sys.par[4])
                        logwt= logwt + logcwt  
                    
                # print ('g', sys.g(
                #     p[:, :, tt], data.u[tt], s[:, :, tt], tt) )
                # print ('data', data.y[tt])
                res= sys.g(
                    p[:, :, tt], data.u[tt], s[:, :, tt], tt) - data.y[tt]
                self.res[tt, :] =  res[0,:]
                #print (self.res[tt, :])
                #print ('residual', rs.shape)

                wmax    = np.max(logwt);
                w[:,tt] = np.exp(sys.par[6]*(logwt - wmax));
       

            if integ_step:
                for aux_i in range(int(sys.par[5])):
                    raimwt2, logwt2 = logfnormpdf(data.y[tt], sys.g(p_aux[:, :, aux_i], data.u[tt], s[:, :, tt], tt), sys.gn(
                         p_aux[:, :, aux_i], s[:, :, tt], tt))                                     
                    w2max = np.max(logwt2)
                    w_b[:, aux_i] = np.exp(logwt2 - w2max)

                    logwt2, pi2 = logRAIMwt(raimwt2, logwt2, sys.num_gnss) #CHANGE TO NUM GNSS

                    if self.mode2 == 'pcraim':
                        logcwt2 = nhp.kld_metric (logwt2, p[:, :, tt],self.nPart, sys.num_cam,tt*sys.skip_num,self.Xct,self.Yct, self.Zct,self.train_arr, sys.par[4])
                        logwt2 = logwt2 + logcwt2  

                    w2max    = np.max(logwt2);
                    w_b[:,aux_i] = np.exp(sys.par[6]*(logwt2 - w2max));

            # Estimate log-likelihood
            log_term = np.log(np.sum(w[:, tt])) - np.log(self.nPart)            
            llp += wmax + log_term
           
            # Estimate state
            w[:, tt] /= np.sum(w[:, tt])
            xh[tt] = np.average(p[:, :, tt], weights=w[:, tt], axis=0) 
            sh[tt] = np.average(s[:, :, tt], weights=w[:, tt], axis=0)

            #estimate accuracy


            # KDE Integrity
            if integ_step:
                kout, cf, kf, gte = self.kdecomp(
                    sys, w[:, tt], w_b, p_aux, p[:, :, tt], tt, np.ones_like(w[:, tt]), p[:, :, tt], data)
                clist.append(cf)
                klist.append(kf)
                elist.append(gte)

        self.xhatf = xh
        self.shatf = sh
        self.ll = llp
        self.w = w
        self.a = a
        self.p = p
        self.s = s
        self.integ = clist
        self.klinteg = klist
        self.error = elist

    #############################################################################################################################

    def plotTrajectories(self, sys):
        plt.plot(self.p[:, 0, sys.T-1], self.p[:, 1, sys.T-1], 'r.')
        ii = np.argmax(self.w[:, sys.T-1])
        att = ii
        for tt in np.arange(sys.T-2, 0, -1):
            at = self.a[att, tt+1]
            at = at.astype(int)
            plt.plot((self.p[at, 0, tt], self.p[att, 0, tt+1]),
                      (self.p[at, 1, tt], self.p[att, 1, tt+1]), 'k')
            #plt.scatter()
            att = at
            att = att.astype(int)

    #############################################################################################################################
    # Resampling helpers
    #############################################################################################################################
    #check this function
    def resampleResidual(self, w, par):
        num_gnss = 12
        H = self.nPart
        H_n = int(H/num_gnss)
        # divide by num_gnss
        Ind = np.empty(H_n, dtype='int')
        num_copies = (np.floor(H_n*np.asarray(w))).astype(int)
        j = 0
        for k in range(H):
            for _ in range(num_copies[k]):  # make n copies
                Ind[j] = k
                j += 1
        residual = w*H_n - num_copies
        residual /= sum(residual)
        csw = np.cumsum(residual)
        csw[-1] = 1
        Ind[j:H_n] = np.searchsorted(csw, np.random.random(H_n-j))
        
        return np.repeat(Ind, num_gnss) #num gnss

    def kdecomp(self, sys, w, w_b, p_aux, p, tt, w_prev, p_prev, data):
        kern = fit_kde(w, p)
        kern_aux = []
        for i in range(int(sys.par[5])):
            kernt = fit_kde(w_b[:, i], p_aux[:, :, i])
            kern_aux.append(kernt)
        emp_err = 0
        totpdf = 0
        circ = np.array([sys.pl, sys.pl,sys.pl])
        for wt,samp in zip(w,p):
            glist = []
            for kaux in kern_aux:
                if kaux is None:
                    glist.append((0))
                    continue    
                #problem with this line
                pgt = kaux.integrate_box(samp-circ, samp+circ)           
                glist.append((1-pgt))
            emp_err += wt*np.average(glist[:sys.temp_odom])            
            totpdf += wt
            
        emp_err /= totpdf
        print ('emp error', emp_err)

        
        kl_prev = comp_KL(w, p, w_prev, p_prev)        
        z = None
        gt = sys.true[tt, :]
        gt_p = kern.integrate_box(gt-circ, gt +circ)
        print ('1-gtp', 1-gt_p)
        return z, emp_err, kl_prev, 1-gt_p



#############################################################################################################################
# End of file
#############################################################################################################################
