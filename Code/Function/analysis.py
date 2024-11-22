import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

import pickle
import scipy.stats as stats
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.linalg import svd, orth
from  sklearn.decomposition import PCA as SKPCA

from . import dynamicalsystem as ds

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
ssm_dir = current_script_path.parents[2] / 'SSM'
sys.path.insert(0, str(ssm_dir))
import ssm as ssm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

sigma = 3  # smoothing amount

def keep_diag(matrix):
    return np.diag(np.diag(matrix))


def calculate_vector_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cos_angle = dot_product / (magnitude1 * magnitude2)

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees   


def angle_between_planes(plane1, plane2):
    def fit_plane(points):
        """Find best-fit plane through points using PCA"""
        # Center the points
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        # Get plane normal using PCA
        pca = SKPCA(n_components=3)
        pca.fit(centered_points)
        
        # The third component (least variance) is the plane normal
        normal = pca.components_[2]
        return center, normal
    
    def get_plane_equation(center, normal):
        """Get normalized plane equation coefficients (ax + by + cz + d = 0)"""
        a, b, c = normal
        d = -np.dot(normal, center)
        norm = np.sqrt(a*a + b*b + c*c)
        return a/norm, b/norm, c/norm, d/norm
    
    center, normal = fit_plane(plane1)
    coefs1 = get_plane_equation(center, normal)
    
    center, normal = fit_plane(plane2)
    coefs2 = get_plane_equation(center, normal)
    
    a1, b1, c1, _ = coefs1
    a2, b2, c2, _ = coefs2
    
    # Angle between plane normals
    cos_theta = abs(a1*a2 + b1*b2 + c1*c2)
    # Clip to handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta) * 180 / np.pi
    
    # Return smaller angle between planes
    return min(angle, 180 - angle)
    

def calculate_principal_angles(A, B):
    QA = orth(A)
    QB = orth(B)

    _, S, _ = svd(QA.T @ QB)
    S = np.clip(S, -1, 1)
    angles = np.arccos(S)
    
    return angles

class PCA:
    def __init__(self, data_stand, multiple_gaps = True):
        self.data = data_stand
        self.score = None 
        self.loading = None 
        self.variance = None

        self.score_per_gap = None
        
        self.Run_PCA_Analysis()
        
        if multiple_gaps: self.Separate_Multiple_Gaps()

        
    def Run_PCA_Analysis(self):
        data = self.data.reshape(self.data.shape[0], -1)
        U, s, Vh = np.linalg.svd(data.T, full_matrices=False)
        
        s_sqr = s ** 2
        self.variance = s_sqr/np.sum(s_sqr) 
        self.score = np.array([U.T[i] * s[i] for i in range(len(s))])
        self.loading = Vh
    
    def Separate_Multiple_Gaps(self):
        valid_PC = min(5, self.score.shape[0])
        self.score_per_gap = self.score[:valid_PC].reshape(valid_PC, self.data.shape[1], self.data.shape[2])
           
class Projection:
    def __init__(self, data, subspace):
        self.data = data 
        self.subspace = subspace
        self.data_projection = keep_diag((subspace @ data.T)/ (subspace @ subspace.T + 1e-18)) @ subspace
        self.data_exclude_projection = self.data - self.projection

class Model:
    def __init__(self, group, train_gap_idx=8):
        self.group = group  
        self.train_gap_idx = train_gap_idx 
        self.gap_dur = round(self.group.gaps[self.train_gap_idx]*1000)
        self.model = self.Get_Model()
        
        self.models, self.best_gap_idx = [], None
        #if cross_validate: self.Cross_Validation()
        
        self.N, self.PCs = None, None
        self.x, self.y, self.z = None, None, None
        self.Input_On, self.Input_Off = None, None
    
    
    def Get_Model(self):
        return ds.DynamicalSystem(self.group, self.train_gap_idx)
    
    def Train(self, cross_validate = False):
        if cross_validate: self.model.Cross_Validate_Optimization()
        else:
            self.model.Optimize_Params()
            
    def Get_Input(self, SoundInput, tau_I, tau_A, delay, invert = False):
        def Get_w(N, tau):
            w = [np.exp(-i/tau) for i in range(N)]
            w = w/np.sum(w)
            return np.array(w)

        def Get_rI(S, tau_I):
            rI = np.zeros(len(S))
            N = round(5*tau_I)
            wI = Get_w(N, tau_I)
            for t in range(0, len(S)):
                if t < N:
                    rI[t] = np.sum(wI[:t][::-1] * S[:t])
                else:
                    rI[t] = np.sum(wI[::-1] * S[t-N:t])
            return rI

        def Get_rA(S, rI, tau_A):
            rA = np.zeros(len(S))
            M = round(5*tau_A)
            wA = Get_w(M, tau_A)
            for t in range(0, len(S)):
                if t < M:
                    rA[t] = 1 / (1 + np.sum(wA[:t][::-1] * rI[:t]))
                else:
                    rA[t] = 1 / (1 + np.sum(wA[::-1] * rI[t-M:t]))
            return rA

        def Get_rIA(S, tau_I, tau_A):
            rI = Get_rI(S, tau_I)
            rA = Get_rA(S, rI, tau_A)
            rIA = np.zeros(len(S))
            for t in range(0, len(S)):
                rIA[t] = rI[t] * rA[t]
            return rI, rA, rIA
        
        S = SoundInput.copy()
        for t in range(1, len(SoundInput)):
            if SoundInput[t-1] == max(SoundInput) and SoundInput[t] == min(SoundInput):
                for i in range(delay):
                    if SoundInput[i] == max(SoundInput): 
                        S[i] = min(SoundInput)
                
        rI, rA, rIA = Get_rIA(S, tau_I, tau_A)
        if not invert:
            rIA -= np.mean(rIA[50:100])
        if invert: 
            rIA *= -1
            rIA -= np.mean(rIA[300:350])
        for i in range(len(rIA)): 
            if rIA[i] < 0: rIA[i] = 0
        for i in range(100+delay+1): rIA[i] = 0
        return rIA
            
    def Run(self, SoundInput = []):
        if len(SoundInput) < 1:
            SoundInput = np.zeros(1000) + self.model.S_off
            for t in range(100, 100 + 250): SoundInput[t] = self.model.S_on 
            for t in range(350+self.model.gap_dur, 350+self.model.gap_dur+100): SoundInput[t] = self.model.S_on
            
        self.OnS = self.Get_Input(SoundInput, self.model.tau_I_on * self.model.tau_I_on_coef, self.model.tau_A_on* self.model.tau_A_on_coef, round(self.model.delay_on * self.model.delay_on_coef), invert = False)
        self.OffS = self.Get_Input(SoundInput, self.model.tau_I_off * self.model.tau_I_off_coef, self.model.tau_A_off * self.model.tau_A_off_coef, round(self.model.delay_off * self.model.delay_off_coef), invert = True)
        self.N = np.zeros((3, len(SoundInput)))
        for t in range(1,len(SoundInput)):
            noise = np.random.normal(0, np.sqrt(0.001), (3, 1))
            self.N[:,[t]] = self.N[:,[t-1]] +(self.model.W@self.N[:,[t-1]] + self.model.OnRe*self.OnS[t-1] + self.model.OffRe*self.OffS[t-1])*self.model.Nt + noise
            
class HMM:
    def __init__(self, N, state = 3):
        self.observation = N
        self.n_state = state
        self.model = None
        self.parameters = None
        self.states = None
        self.TransM = None
        self.loglikelihood = None
        self.kl_divergence = None
        self.ConnecM = None
        
        
    def Fit_Model(self):
        # Fit Models
        fitting_input = self.observation.T
        self.model = ssm.HMM(self.n_state, len(fitting_input[0]), observations="gaussian")
        lls = self.model.fit(fitting_input, method="em", num_iters=50, init_method="kmeans")
        
        self.parameters = self.model.observations.params
        #np.save('../../SocialData/HMMStates/Params_' + self.mouse.type + "_" + self.mouse.mouse + '.npy', self.parameters)
        
        # Sort and Save Transition Matrix
        self.TransM = self.model.transitions.transition_matrix
        #np.save('../../SocialData/HMMStates/TransM_' + self.mouse.type + "_" + self.mouse.mouse + '.npy', self.TransM)
        
        # Infer and Save states
        obs = self.observation.T
        self.loglikelihood = self.model.log_likelihood(obs)
        self.states = self.model.most_likely_states(obs)
        #np.save('../../SocialData/HMMStates/States_' + self.mouse.type + "_" + self.mouse.mouse + ".npy", self.states)