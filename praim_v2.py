from classes import *
from helpers import *
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
from smc import smcSampler
from IPython import display
import os,sys
from scipy.signal import savgol_filter
import csv


# clear = lambda: os.system('cls')  # On Windows System
# clear()

data = stData()

op_dir = "RESULTS/SEPT 2/AL_16m_new"

if not os.path.exists(op_dir):
    os.makedirs(op_dir)


num_runs         = 10; #number of runs, delete this line
num_gnss         = 12; #number of pseudorange measurements 
sys              = stSystemLGSS(num_gnss)
par              = stParameters();
sys.version      = "standard"

po1     = 0.90

lda              = 0.8 
wt1              = 1; 
wt2              = 1; 
sys.par          = np.zeros((11,1))
sys.par[0]       = 1; #autoregressive term

sys.par[1]       = 2; #1e-5; #ground truth motion randomness, 1e-5 original
sys.par[2]       = 10; #0.01; #measurement noise variance- 10, maybe make this 25
sys.odom_noise   = 0.5

sys.num_faults   = 2 #3 originally
sys.par[4]       = 100; #bias noise-100, fault added to the pseudo range
filepath         = os.path.join(op_dir,'2_100.xlsx')

sys.par[5]       = 5; #seems to be an imp parameternumber of odometries for empirical risk, 10 originally

sys.par[6]       = 0.01;#logwt gamma scaling


sys.par[7]       = 100; #integrity compute interval 10 originally, this should be a multiple of 4- 1 so like 119 for eg.


sys.bias_p       = 0.2 #bias probability
sys.num_gnss     = num_gnss; #number of pseudorange measurements
sys.num_cam      = 2; #no. of camera measurements btwn 2 pseudorange measurements

sys.T_0          = 3440#set to 400 originally
sys.skip_num     =  1  #10
sys.min_num      =  0 #760
sys.T            = int((sys.T_0 - sys.min_num)/sys.skip_num)#set to 400 originally

sys.pl           = 16; #this is the alert limit, might need to change this to a low no.
sys.temp_odom    = 10;

sys.createSat(sys.T, num_gnss)


file= 'ground_truth.xlsx'
df = pd.read_excel(file, usecols="A:M")

sys.true= np.zeros((sys.T,3))
sys.true[:,0] = df['X'].to_numpy()[sys.min_num:sys.T_0:sys.skip_num]
sys.true[:,1] = df['Y'].to_numpy()[sys.min_num:sys.T_0:sys.skip_num]
sys.true[:,2] = df['Z'].to_numpy()[sys.min_num:sys.T_0:sys.skip_num] 


u = np.zeros((sys.T,3))
u[:,0] = df['Vx'].to_numpy()[sys.min_num: sys.T_0:sys.skip_num]*sys.skip_num
u[:,1] = df['Vy'].to_numpy()[sys.min_num: sys.T_0:sys.skip_num]*sys.skip_num
u[:,2] = df['Vz'].to_numpy()[sys.min_num: sys.T_0:sys.skip_num]*sys.skip_num



data.sample(sys,u)
sys.ubar = u

fig, ax = plt.subplots()

sm = smcSampler()
sm.bPF(data,sys,par,ax)
t = np.linspace(1,sys.T,len(sm.integ))
t2 = np.linspace(1,sys.T)
if sm.mode == "praim":
    emp_risk = wt1 * np.asarray(sm.integ)
    epsilon = compute_eps(lda, sm.klinteg, 0.2, sys.temp_odom)
    div_risk = (wt2) * (np.asarray(inv_bernoulli_div(emp_risk, epsilon)))
    tot_risk = emp_risk 
    tot_risk[tot_risk>1] = 1
    print ('emp risk', emp_risk)
    print ('div',div_risk)
    
    # print (len(tot_risk))
    ref_risk = sm.error
    print('ref_risk', ref_risk)
    

    m_err = np.linalg.norm(np.subtract(sys.true,sm.xhatf), ord=2, axis=1)
    #add in the new stuff-optional to use alright
    print('Mean Error',np.average(m_err))

    fault_err = m_err[t[tot_risk<=1].astype(int) - 1]

    # integ_ok1 = len(tot_risk[tot_risk < po1])
    # integ_ok2 = len(tot_risk[tot_risk < po2])
    # no_fault = len(fault_err[fault_err < sys.pl])
    # with_fault = len(fault_err[fault_err > sys.pl])

    err_int_ok1 = np.average(m_err[t[tot_risk < po1].astype(int) - 1])
    err_int_w1 = np.average(m_err[t[tot_risk > po1].astype(int) - 1])

    fa = 0
    mi = 0

    for i in range(len(fault_err)):
        if fault_err[i] > sys.pl and tot_risk[i] < po1:
            mi += 1
        elif fault_err[i] < sys.pl and tot_risk[i] > po1:
            fa += 1

    df = pd.DataFrame (sm.xhatf)
    df.to_excel(filepath, index=False)


    fail_rat = np.sum(ref_risk>tot_risk)/len(tot_risk)*100
    print (fail_rat)
    bound_gap = np.average(np.abs(tot_risk-ref_risk))
    csv_data =  [sys.num_faults, sys.par[4][0], np.average(m_err)]
    csv_data1 = [fail_rat, np.average(m_err[t[ref_risk>tot_risk].astype(int) - 1]), bound_gap]
    csv_data2 = [fa, mi, err_int_ok1, err_int_w1]
    
    #csv_data2 = [no_fault, with_fault, integ_ok1, err_int_ok1, err_int_w1,integ_ok2, err_int_ok2, err_int_w2]

    with open(r'IM_metrics_16m.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(csv_data)
        writer.writerow(csv_data1)
        writer.writerow(csv_data2)
    
    plt.figure(1)
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("white")
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('0.01')  
    ax.grid(linestyle='-', linewidth='0.1', color='grey')
    #plt.ylim(0,1)
    plt.plot(t,ref_risk, color='c', linewidth=2)
    plt.plot(t,tot_risk, color='k', linewidth=2)
    plt.savefig(os.path.join(op_dir, str(sys.num_faults) + "f" + str(sys.par[4])+"_rbound.png"))

   

    # plt.clf()
    # plt.scatter(sys.true[:,0],sys.true[:,1],color='c')   
    # sm.plotTrajectories(sys)
    # plt.savefig(os.path.join(op_dir, str(sys.num_faults) + "f_praim" + str(sys.par[4]) + str(np.average(m_err)) + "_path.png" ))

