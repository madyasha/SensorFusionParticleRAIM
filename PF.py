import numpy as np
import math
from simulation import dt

class Particle(object):
    '''INITIALIZE VARIABLES'''
    def __init__(self, id, x,y):
        self.x = x
        self.y = y
        self.id = id

class PFilt(object):
    '''INITIALIZE VARIABLES'''
    def __init__(self, s, p, num_s):
        self.particles = self.gen_particles(s[:2], [10,10], num_s, 0)
        self.weights = []
        self.num_particles = num_s
        self.sats = p

        self.t = 0
    
        self.dx = s[2]
        self.dy = s[3]
        self.weights = np.ones((self.num_particles,))
        self.err_fac = 10
        self.pmove = 0.5
        
    '''GENERATE PARTICLES FROM INITAL STATES'''    
    def gen_particles(self, mu, sig, n, s_id): 
        particles = []
        for i in range(n):
            p_x = np.random.normal(loc=mu[0], scale=sig[0])
            p_y = np.random.normal(loc=mu[1], scale=sig[1])
            particles.append(Particle(s_id+i,p_x, p_y))
        return particles

    '''CALCULATE EXPECTED MEASUREMENT FOR A PARTICLE'''
    def pred_meas(self, particle):
        vr = np.array([particle.x, particle.y, 0])
        ranges = []
        
        for sat in self.sats:
            vs = np.array([sat[0],sat[1], sat[2]])
        
            ranges.append(np.linalg.norm(vr - vs))

        return np.array(ranges)

    '''COMPUTE ERROR BETWEEN EXPECTED AND OBTAINED MEASUREMENT FOR A PARTICLE'''
    def comp_pred_err(self, particle, tranges):
        ranges = self.pred_meas(particle)
        # print((tranges-ranges)**2)
        res_err = (tranges-ranges)**2
        m_err = max(res_err)
        # m_err = 0
        return self.err_fac*(np.sum(res_err)-m_err)

    '''MOVE A PARTICLE ACCORDING TO ODOMENTRY AND NOISE'''
    def pred(self, particle):
        particle.x += (self.dx)*dt + np.random.normal(0,self.pmove)
        particle.y += (self.dy)*dt + np.random.normal(0,self.pmove)

    '''CORRECT SPEED FROM ODOMETRY'''
    def upd_spd(self, odom):
        self.dx = odom[0]
        self.dy = odom[1]
    
    '''CHANGE WEIGHT GRADIENT BASED ON PARTICLE DISTRIBUTION'''
    def upd_err_fac(self, pos):
        # print(np.std(pos[:,0]),np.std(pos[:,1]))
        if np.std(pos[:,0]) < 4 and np.std(pos[:,1]) < 4:
            self.err_fac = 200
            self.pmove = 0.05

    '''FULL UPDATE STEP - ODOMETRY UPDATE, RESAMPLE THEN COMPUTE WEIGHTS'''
    def update(self, tranges, odom):
        self.t += 1
        self.upd_spd(odom)
        
        # RESAMPLING
        N_eff_weight = sum(1/self.weights**2)
        N_eff_weight /= self.num_particles
        print(N_eff_weight)
        if N_eff_weight>2e7 and self.t>1:
            self.particles = self.resample()

        errors = []
        for particle in self.particles:
            self.pred(particle)
            error = self.comp_pred_err(particle,tranges)
            errors.append(error)

        errors = np.array(errors)
        # WEIGHT UPDATE
        errors = (errors - np.max(errors))
        wts = self.sigmoid(errors) + 1e-100
        wts /= np.sum(wts) 
        self.weights *= wts
        self.weights /= np.sum(self.weights)
        
    '''HELPER FUCTION FOR SIGMOID'''
    def sigmoid(self, x):
        """Numerically-stable sigmoid function."""
        z = np.exp(-x)
        return z / (1 + z)

    '''RESAMPLE USING CDF OF WEIGHTS - SYSTEMATIC RESAMPLING'''
    def resample(self):
        new_particles = []
        # sample_u = np.random.uniform(0,1/self.num_particles)
        # index = int(sample_u * (self.num_particles - 1))
        # if self.weights == []:
        #     self.weights = [1] * self.num_particles
        #     print(self.weights)
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1
        idx = np.searchsorted(cdf,np.random.random(self.num_particles))
        # i = int(np.random.uniform(0,1) * (self.num_particles - 1))
        # i = np.argmax(self.weights)
        # i=0

        pos_vals = []

        # for j in range(self.num_particles):
        #     beta = sample_u + (j-1)/self.num_particles
        #     while beta > cdf[i]:
        #         # beta -= self.weights[index]
        #         i = (i + 1) % self.num_particles

        #     particle = self.particles[i]
            
        #     new_particles.append(Particle(particle.id, particle.x, particle.y))
        #     pos_vals.append([particle.x, particle.y])
        # self.upd_err_fac(np.array(pos_vals))
        
        for i in idx:
            particle = self.particles[i]
            
            new_particles.append(Particle(particle.id, particle.x+np.random.normal(0,1), particle.y+np.random.normal(0,1)))
            pos_vals.append([particle.x, particle.y])
        self.upd_err_fac(np.array(pos_vals))
        
        self.weights = [1/self.num_particles] * self.num_particles
        return new_particles

    '''PRINT PARTICLE STATES'''
    def pstate(self):
        for particle in self.particles:
            print(particle.x, particle.y)
