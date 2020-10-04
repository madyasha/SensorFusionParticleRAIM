########################################################################
# Particle Metropolis-Hastings using gradient and Hessian information
# Copyright (c) 2014 Johan Dahlin ( johan.dahlin (at) liu.se )
#
# classes.py
# Encodes the structure of different models
#
########################################################################

import numpy as np
from helpers import *
import pandas as pd
import random

########################################################################
# Class definitions
########################################################################

class stParameters(object):
    
    # Standard settings
    nPars                 = 1;      # No. parameters to infer
    nMCMC                 = 1000;   # No. MCMC iterations
    nBurnIn               = 500;    # No. MCMC iterations as burn-in
    stepSize              = 0.1;    # Proposal step size
    normLimit             = 0.01;   # Limit for computing hitting time
    dataset               = 0.0;    # Which data set to use
    fileprefix            = [];     # Prefix for file names
    nProgressReport       = 100;    # When to print new information
    verboseSampler        = 0;      # Write out verbose information?
    zeroscore             = 0;      # Neglect the score information?
    writeOutPriorWarnings = 0;      # Write out if proposed parameter is
                                    # discarded due to prior

class stSystemLGSS(object):
    
    ########################################################################
    # LGSS model
    ########################################################################
    #
    # x(t+1) = par[0] * x(t) + par[1] * v(t),    v(t) ~ N(0,1)
    # y(t)   =          x(t) + par[2] * e(t),    e(t) ~ N(0,1)
    #
    ########################################################################

    # Settings
    nPar       = 3;                         # Total no. parameters in model
    par        = np.zeros(3);
    T          = 0
    xo = np.array([4051761,617191,4870822])
    so = np.array([4051761,617191,4870822])
    model      = "Linear Gaussian system"
    supportsFA = 1.0;                       # Is it fully adaptable?
    scale      = 1.0;                       # rescale the model with scale
    version    = "standard"                 # parameterisation (standard or tanhexp)
    dh         = 0.0;
    
    
    #remember to change the line below
    def __init__(self, num_gnss):
        #self.createSat(3440,num_gnss) 
        pass

               
#modify from the following
    def createSat(self, T, N): 
        self.sat_pos=[]
        excel_file= 'ranges.xlsx'
        df = pd.read_excel(excel_file, usecols="A:G")
        X= df['X']
        Y= df['Y']
        Z= df['Z']
        for n in range(N):
            #st_l = [np.array([X[n+ t*N], Y[n+ t*N],Z[n+ t*N]])  for t in range(T)]
            st_l = [np.array([X[self.min_num*N + n+ t*self.skip_num*N], Y[self.min_num*N + n+ t*self.skip_num*N],Z[self.min_num*N +n+ t*self.skip_num*N]])  for t in range(T)]
            self.sat_pos.append(st_l)
            #print (st_l)
        self.sat_pos = np.array(self.sat_pos)

        

    # Functions for reparametrisations
    
    def storeParameters(self,newParm,sys,par):
        self.par = np.zeros(self.nPar);

        for kk in range(0,par.nPars):
            self.par[kk] = np.array(newParm[kk], copy=True)

        for kk in range(par.nPars,self.nPar):
            self.par[kk] = sys.par[kk];

    def returnParameters(self,par):
        out = np.zeros(par.nPars);

        for kk in range(0,par.nPars):
            out[kk]  = self.par[kk];

        return(out);

    def transform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.tanh( self.par[0] );
            self.par[1] = np.exp ( self.par[1] ) * self.scale;
        else:
            self.par[1] = self.par[1] * self.scale;

    def invTransform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.arctanh( self.par[0] );
            self.par[1] = np.log    ( self.par[1] / self.scale );
        else:
            self.par[1] = self.par[1] / self.scale;

    # Model functions
    def h(self, xt, st, ut, tt):
        return 0.0;
    
    def f(self, xt, ut, st, yt, tt):
        return self.par[0] * xt + ut*np.ones_like(xt);
    
    def fn(self, xt, st, yt, tt):
        var = np.eye(3)* np.sqrt((self.par[1])) 
        #var[2,2] = 1e-7
        return var
   
       
    def g(self, xt, ut, st, tt):
        #returns predicted pseudo range
        num_gnss= 12
        if len(xt) <4:
            x_t = np.array([xt[0], xt[1], xt[2]])
            out = np.zeros((num_gnss))
            for i in range(num_gnss):
                out[i] = np.linalg.norm(x_t-self.sat_pos[i][tt])
                
        else:
            out = np.zeros((len(xt),len(self.sat_pos)))
            for i in range(num_gnss):
                for j in range(len(xt)):
                    x_t = np.array([xt[j,0], xt[j,1], xt[j,2]])
                    out[j,i] = np.linalg.norm(x_t-self.sat_pos[i][tt])
        #print (tt,'out',out)            
        return out

    def gn(self, xt, st, tt):
        return np.eye(len(self.sat_pos))*self.par[2]            

    # Fully-adapted model functions
    def fa(self, xt, ytt, ut, st, tt):
        delta = self.par[1]**(-2) + self.par[2]**(-2); delta = 1.0 / delta;
        return delta * ( ytt * self.par[2]**(-2) + self.par[1]**(-2) * self.par[0] * xt );

    def fna(self, xt, ytt, st, tt):
        delta = self.par[1]**(-2) + self.par[2]**(-2); delta = 1.0 / delta;
        return np.sqrt(delta);
    
    def ga(self, xt, ut, st, tt):
        return self.par[0] * xt;

    def gna(self, xt, st, tt):
        return np.sqrt( self.par[1]**2 + self.par[2]**2 );   
    
    # Gradients of the distributions of states and measurements
    def Dparm(self, xtt, xt, yt, st, at, par):
        
        nOut = len(xtt);
        gradient = np.zeros(( nOut, par.nPars ));
        Q1 = self.par[1]**(-1);        
        Q2 = self.par[1]**(-2);
        Q3 = self.par[1]**(-3);        
        R1 = self.par[2]**(-1);
        R3 = self.par[2]**(-3);
        px = xtt - self.par[0] * xt;
        py = yt - xt;

        if (self.version == "tanhexp"):
            for v1 in range(0,par.nPars):
                if v1 == 0:
                    gradient[:,v1] = xt * Q2 * px * ( 1.0 - self.par[0]**2 );
                elif v1 == 1:
                    gradient[:,v1] = ( Q2 * px**2 - 1.0 ) * self.scale;
                elif v1 == 2:
                    gradient[:,v1] = R3 * py**2 - R1;
                else:
                    gradient[:,v1] = 0.0;
        else:
            for v1 in range(0,par.nPars):
                if v1 == 0:
                    gradient[:,v1] = xt * Q2 * px;
                elif v1 == 1:
                    gradient[:,v1] = ( Q3 * px**2 - Q1 ) * self.scale;
                elif v1 == 2:
                    gradient[:,v1] = R3 * py**2 - R1;
                else:
                    gradient[:,v1] = 0.0;            
        return(gradient);

    # Hessians of the distributions of states and measurements
    def DDparm(self, xtt, xt, yt, st, at, par):
        
        nOut = len(xtt);
        hessian = np.zeros( (nOut, par.nPars,par.nPars) );
        Q1 = self.par[1]**(-1);
        Q2 = self.par[1]**(-2);
        Q3 = self.par[1]**(-3);
        Q4 = self.par[1]**(-4);
        R2 = self.par[2]**(-2);
        R4 = self.par[2]**(-4);
        px = xtt - self.par[0] * xt;
        py = yt - xt;

        if (self.version == "tanhexp"):
            for v1 in range(0,par.nPars):
                for v2 in range(0,par.nPars):
                    if ( (v1 == 0) & (v2 == 0) ):
                        hessian[:,v1,v2] = - xt**2 * Q2 * ( 1.0 - self.par[0]**2 )**2 - 2.0 * self.par[0] * Q2 * xt * px * ( 1.0 - self.par[0]**2 )
    
                    elif ( (v1 == 1) & (v2 == 1) ):
                        hessian[:,v1,v2] = - 2.0 * Q2 * px**2 * self.scale**2;
    
                    elif ( ( (v1 == 1) & (v2 == 0) ) | ( (v1 == 0) & (v2 == 1) ) ):
                        hessian[:,v1,v2] = - 2.0 * xt * Q2 * px * ( 1.0 - self.par[0] ) * self.scale;
    
                    elif ( (v1 == 2) & (v2 == 2) ):
                        hessian[:,v1,v2] = R2 - 3.0 * R4 * py**2
    
                    else:
                        hessian[:,v1,v2] = 0.0;
                    
        else:
            for v1 in range(0,par.nPars):
                for v2 in range(0,par.nPars):
                    if ( (v1 == 0) & (v2 == 0) ):
                        hessian[:,v1,v2] = - xt**2 * Q2;
    
                    elif ( (v1 == 1) & (v2 == 1) ):
                        hessian[:,v1,v2] = ( Q2 - 3.0 * Q4 * px**2 - Q1 ) * self.scale**2;
    
                    elif ( ( (v1 == 1) & (v2 == 0) ) | ( (v1 == 0) & (v2 == 1) ) ):
                        hessian[:,v1,v2] = - 2.0 * xt * Q3 * px * self.scale;
    
                    elif ( (v1 == 2) & (v2 == 2) ):
                        hessian[:,v1,v2] = R2 - 3.0 * R4 * py**2
    
                    else:
                        hessian[:,v1,v2] = 0.0;            

        return(hessian);

    # Uniform prior
    def priorUniform(self):
        return 1.0;

    # Derivatives of priors
    def prior(self):
        return(0.0);

    def dprior1(self, v1):
        return(0.0);

    def ddprior1(self, v1, v2):
        return(0.0);

    def Jacobian( self ):
        if (self.version == "tanhexp"):
            return np.log( 1.0 - self.par[0]**2 ) + np.log( self.par[1] );
        else:
            return 0.0;

class stSystemHW(object):
    ########################################################################
    # HW model
    ########################################################################
    #
    # x(t+1) = par[0] * x(t) + par[1] * v(t),                v(t) ~ N(0,1)
    # y(t)   =                 par[2] exp(-x(t)/2)* e(t),    e(t) ~ N(0,1)
    #
    ########################################################################

    nPar       = 3;
    par        = np.zeros(3);
    T          = 0
    xo         = 0
    so         = 0;
    model      = "Hull-White Stochastic Volatility model"
    supportsFA = 0.0;
    scale      = 1.0;
    version    = "standard"
    dh         = 0.0;

    def storeParameters(self,newParm,sys,par):
        self.par = np.zeros(self.nPar);

        for kk in range(0,par.nPars):
            self.par[kk] = np.array(newParm[kk], copy=True)

        for kk in range(par.nPars,self.nPar):
            self.par[kk] = sys.par[kk];

    def returnParameters(self,par):
        out = np.zeros(par.nPars);

        for kk in range(0,par.nPars):
            out[kk]  = self.par[kk];

        return(out);

    def transform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.tanh( self.par[0] );
            self.par[1] = np.exp ( self.par[1] ) * self.scale;
        else:
            self.par[1] = self.par[1] * self.scale;

    def invTransform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.arctanh( self.par[0] );
            self.par[1] = np.log    ( self.par[1] / self.scale );
        else:
            self.par[1] = self.par[1] / self.scale;

    def h(self, xt, st, ut, tt):
        return 0.0;
    
    def f(self, xt, ut, st, yt, tt):
        return self.par[0] * xt;
    
    def fn(self, xt, st, yt, tt):
        return self.par[1];        
    
    def g(self, xt, ut, st, tt):
        return 0;

    def gn(self, xt, st, tt):
        return self.par[2] * np.exp( 0.5 * xt);

    def fa(self, xt, ytt, ut, st, tt):
        return 0;

    def fna(self, xt, ytt, st, tt):
        return 0;
    
    def ga(self, xt, ut, st, tt):
        return 0;

    def gna(self, xt, st, tt):
        return 0;
    
    def Dparm(self, xtt, xt, yt, st, at, par):
        
        nOut = len(xtt);
        gradient = np.zeros(( nOut, par.nPars ));
        Q1 = self.par[1]**(-1);        
        Q2 = self.par[1]**(-2);
        Q3 = self.par[1]**(-3);        
        R1 = self.par[2]**(-1);
        R3 = self.par[2]**(-3);
        px = xtt - self.par[0] * xt;
        py = yt;

        if (self.version == "tanhexp"):
            for v1 in range(0,par.nPars):
                if v1 == 0:
                    gradient[:,v1] = xt * Q2 * px * ( 1.0 - self.par[0]**2 );
                elif v1 == 1:
                    gradient[:,v1] = ( Q2 * px**2 - 1.0 ) * self.scale;
                elif v1 == 2:
                    gradient[:,v1] = ( R3 * py**2 - R1 ) * np.exp( 0.5 * xt );
                else:
                    gradient[:,v1] = 0.0;
        else:
            for v1 in range(0,par.nPars):
                if v1 == 0:
                    gradient[:,v1] = xt * Q2 * px;
                elif v1 == 1:
                    gradient[:,v1] = ( Q3 * px**2 - Q1 ) * self.scale;
                elif v1 == 2:
                    gradient[:,v1] = ( R3 * py**2 - R1 ) * np.exp( 0.5 * xt );
                else:
                    gradient[:,v1] = 0.0;            
        return(gradient);


    def DDparm(self, xtt, xt, yt, st, at, par):
        
        nOut = len(xtt);
        hessian = np.zeros( (nOut, par.nPars,par.nPars) );
        Q1 = self.par[1]**(-1);
        Q2 = self.par[1]**(-2);
        Q3 = self.par[1]**(-3);
        Q4 = self.par[1]**(-4);
        R2 = self.par[2]**(-2);
        R4 = self.par[2]**(-4);
        px = xtt - self.par[0] * xt;
        py = yt;

        if (self.version == "tanhexp"):
            for v1 in range(0,par.nPars):
                for v2 in range(0,par.nPars):
                    if ( (v1 == 0) & (v2 == 0) ):
                        hessian[:,v1,v2] = - xt**2 * Q2 * ( 1.0 - self.par[0]**2 )**2 - 2.0 * self.par[0] * Q2 * xt * px * ( 1.0 - self.par[0]**2 )
    
                    elif ( (v1 == 1) & (v2 == 1) ):
                        hessian[:,v1,v2] = - 2.0 * Q2 * px**2 * self.scale**2;
    
                    elif ( ( (v1 == 1) & (v2 == 0) ) | ( (v1 == 0) & (v2 == 1) ) ):
                        hessian[:,v1,v2] = - 2.0 * xt * Q2 * px * ( 1.0 - self.par[0] ) * self.scale;
    
                    elif ( (v1 == 2) & (v2 == 2) ):
                        hessian[:,v1,v2] = ( R2 - 3.0 * R4 * py**2 ) * np.exp( xt )
    
                    else:
                        hessian[:,v1,v2] = 0.0;
                    
        else:
            for v1 in range(0,par.nPars):
                for v2 in range(0,par.nPars):
                    if ( (v1 == 0) & (v2 == 0) ):
                        hessian[:,v1,v2] = - xt**2 * Q2;
    
                    elif ( (v1 == 1) & (v2 == 1) ):
                        hessian[:,v1,v2] = ( Q2 - 3.0 * Q4 * px**2 - Q1 ) * self.scale**2;
    
                    elif ( ( (v1 == 1) & (v2 == 0) ) | ( (v1 == 0) & (v2 == 1) ) ):
                        hessian[:,v1,v2] = - 2.0 * xt * Q3 * px * self.scale;
    
                    elif ( (v1 == 2) & (v2 == 2) ):
                        hessian[:,v1,v2] = ( R2 - 3.0 * R4 * py**2 ) * np.exp( xt )
    
                    else:
                        hessian[:,v1,v2] = 0.0;            

        return(hessian);

    def priorUniform(self):
        return 1.0;

    def prior(self):
        return(0.0);

    def dprior1(self, v1):
        return(0.0);


    def ddprior1(self, v1, v2):
        return(0.0);

    def Jacobian( self ):
        if (self.version == "tanhexp"):
            return np.log( 1.0 - self.par[0]**2 ) + np.log( self.par[1] );
        else:
            return 0.0;

########################################################################
# Generate data
########################################################################
class stData(object):
    x = []
    u = []
    y = []
    e = []
    v = []
    T = []
    model = []

    # the class "constructor"  - It's actually an initializer
    def __init__(self):
        model = "empty"

    def sample(self, sys, u):
        s = np.zeros((sys.T+1,3));
        x = np.zeros((sys.T+1,3));
        y = np.zeros((sys.T, sys.num_gnss));
        real = np.zeros((sys.T, sys.num_gnss));
        v = np.random.randn(sys.T,3);
        e = np.random.randn(sys.T, sys.num_gnss);

        x[0] = sys.xo;
        s[0] = sys.so;       
        sel = [0]*sys.num_gnss;

        #read range measurements from excel file
        excel_file= 'ranges.xlsx'
        df = pd.read_excel(excel_file, usecols="A:M")
        prange= df['Prange']
        res = df['Residuals'].to_numpy()
        pr_ind= np.arange(sys.min_num*sys.num_gnss, sys.T_0* sys.num_gnss, sys.num_gnss*sys.skip_num)


        file= 'ground_truth.xlsx'
        df = pd.read_excel(file, usecols="A:M")

        for tt in range(0, sys.T):
            rs = res[pr_ind[tt]: pr_ind[tt] + sys.num_gnss]
            if np.random.random()>sys.bias_p:
                sel = [0]*sys.num_gnss
                #ind = random.sample(range(0, sys.num_gnss), sys.num_faults)
                # for k in ind:
                #     #sel[k] = sys.par[4][0]
                #     sel[k] = -rs[k]
                for i in range(sys.num_faults):
                    ind = np.random.randint(0,sys.num_gnss)
                    sel[ind] = sys.par[4][0]

            #print ('tt-bias', tt, sel)
            #y[tt] = prange[pr_ind[tt]: pr_ind[tt] + sys.num_gnss] + np.asarray(sel)
            y[tt] = prange[pr_ind[tt]: pr_ind[tt] + sys.num_gnss] + np.asarray(sel) - res[pr_ind[tt]: pr_ind[tt] + sys.num_gnss]
            s[tt+1] = sys.h(x[tt], u[tt], s[tt], tt);
            x[tt+1] = sys.f(x[tt], u[tt], s[tt], y[tt], tt) + np.matmul(sys.fn(x[tt], s[tt], y[tt], tt) , v[tt]);

            # u[:,0] = df['VelX'].to_numpy()[sys.min_num: sys.T_0:sys.skip_num]*sys.skip_num
            # u[:,1] = df['VelY'].to_numpy()[sys.min_num: sys.T_0:sys.skip_num]*sys.skip_num
            #u[:,2] = df['VelZ'].to_numpy()[sys.min_num: sys.T_0:sys.skip_num]*sys.skip_num


        self.x = x[0:sys.T,:]
        self.s = s[0:sys.T,:]
        self.y = y;
        self.u = u + sys.odom_noise*v;
        self.v = v;
        self.e = e;
        self.T = sys.T;
        self.model = sys.model;

#############################################################################################################################
# End of file
#############################################################################################################################  
