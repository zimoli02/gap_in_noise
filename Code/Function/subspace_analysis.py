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
from sklearn.decomposition import PCA as SKPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


from . import dynamicalsystem as ds
from . import analysis

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
ssm_dir = current_script_path.parents[2] / 'SSM'
sys.path.insert(0, str(ssm_dir))
import ssm as ssm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def Center(data):
    data_centered = []
    for i in range(len(data)):
        data_centered.append(data[i] - np.mean(data[i]))
    return np.array(data_centered)

def Generate_Orthonormal_Matrix(n=10):
    # Generate a random n×n matrix
    A = np.random.randn(n, n)
    # QR decomposition: A = QR where Q is orthogonal
    Q, R = np.linalg.qr(A)
    return Q

def Calculate_Similarity(period1, period2, method = 'Trace', shuffle_method = False):
    if shuffle_method == 'Shuffle_Neuron':
        # Shuffle the index of neurons but keep the overall population activity
        random_idx = np.random.randint(0,len(period1), size = len(period1))
        period1 = period1[random_idx]
        random_idx = np.random.randint(0,len(period2), size = len(period2))
        period2 = period2[random_idx]
        
    if shuffle_method == 'Add_Neuron':
        # Add 20% neurons with gaussian noise
        mean, std = 3, 3 
        n= int(len(period1)*0.2)
        noise_matrix = std * np.random.randn(n, len(period1[0])) + mean
        period1 = np.concatenate([noise_matrix, period1], axis = 0)
        noise_matrix = std * np.random.randn(n, len(period2[0])) + mean
        period2 = np.concatenate([noise_matrix, period2], axis = 0)
        
    if shuffle_method == 'Add_Noise':
        mean, std = 3, 3 
        noise_matrix = std * np.random.randn(len(period1), len(period1[0])) + mean
        period1 = period1 + noise_matrix
        noise_matrix = std * np.random.randn(len(period2), len(period2[0])) + mean
        period2 = period2 + noise_matrix
        
    if method == 'Pairwise':
        # Mean-center the data matrix
        period1 = Center(period1)
        period2 = Center(period2)
        # Use SVD to find the basis of these data matrices
        _, _, V1 = np.linalg.svd(period1.T, full_matrices=False)
        _, _, V2 = np.linalg.svd(period2.T, full_matrices=False)
        
        if shuffle_method == 'Rotate':
            Q = Generate_Orthonormal_Matrix(len(V2))
            V2 = Q @ V2
            
        # Calculate the pairwise alignment
        PC_Alignment = []
        for i in range(min(len(V1), len(V2))):
            dot_product = min(abs(np.dot(V1[i], V2[i])), 1)
            alignment = 1 - np.arccos(dot_product)/(np.pi/2)
            PC_Alignment.append(alignment)
        return PC_Alignment[0]
    
    if method == 'CCA':
        # Mean-center the data matrix
        period1 = Center(period1)
        period2 = Center(period2)
        # Use SVD to find the basis of these data matrices
        _, _, V1 = np.linalg.svd(period1.T, full_matrices=False)
        _, _, V2 = np.linalg.svd(period2.T, full_matrices=False)
        
        if shuffle_method == 'Rotate':
            Q = Generate_Orthonormal_Matrix(len(V2))
            V2 = Q @ V2
            
        # Calculate the canonical correlation
        W = V1 @ V2.T
        _, Sigma, _ = np.linalg.svd(W)
        # Calculate the largest alignment (smallest angle)
        largest_alignment = 1 - np.arccos(min(Sigma[0], 1))/(np.pi/2)
        smallest_alignment = 1 - np.arccos(min(Sigma[-1], 1))/(np.pi/2)

        return largest_alignment
    
    if method == 'RV':
        # RV Coefficient:  see the extent to which the two sets of variables give similar images of the n individuals, 
        # variables: time T; individuals: neuron N
        # T can be different - see the to which the two sets of timeseries give similar images of the N neurons
        period1 = period1.T 
        period2 = period2.T

        # All variables (neural data at each timepoint) have been centred to have means equal to 0
        period1 = Center(period1)
        period2 = Center(period2)

        # C(X) (NxN) as a measure of the relative positions of points in a configuration
        S_X1 = period1.T @ period1
        S_X2 = period2.T @ period2
        #Config1 = S_X1 / np.sqrt(np.trace(S_X1 @ S_X1))
        #Config2 = S_X2 / np.sqrt(np.trace( S_X2 @ S_X2))
        
        if shuffle_method == 'Rotate':
            Q = Generate_Orthonormal_Matrix(len(S_X2))
            S_X2 = Q @ S_X2 @ Q.T
        
        RV = np.trace(S_X1 @ S_X2) / np.sqrt(np.trace(S_X1 @ S_X1) * np.trace(S_X2 @ S_X2))
        
        return RV
    
    if method == 'Trace':
        # Mean-center the data matrix
        period1 = Center(period1)
        period2 = Center(period2)   
        # Calculate covariance matrix
        C1 = period1@(period1.T)/len(period1[0])
        C2 = period2@(period2.T)/len(period2[0])
        
        if shuffle_method == 'Rotate':
            Q = Generate_Orthonormal_Matrix(len(C2))
            C2 = Q @ C2 @ Q.T
        
        # Scale the covariance matrix
        S1 = C1/np.trace(C1)
        S2 = C2/np.trace(C2)
        PR1 = np.trace(S1)**2/(np.trace(S1@S1))
        PR2 = np.trace(S2)**2/(np.trace(S2@S2))
        
        return np.trace(S1@S2) * (np.sqrt(PR1)*np.sqrt(PR2))
    

def Subspace_Comparison_Simulation(n_observation, n_feature):
    def Generate_Projections(n_rank, n_observation):
        orthonormal_matrix = Generate_Orthonormal_Matrix(max(n_rank, n_observation))
        return orthonormal_matrix[:n_observation, :n_rank]
        
    def Generate_Loadings(n_rank, n_feature):
        orthonormal_matrix = Generate_Orthonormal_Matrix(max(n_rank, n_feature))
        return orthonormal_matrix[:n_feature, :n_rank]
        
    def Generate_Sigma(n_rank):
        random_array = np.random.rand(n_rank)
        sigma = random_array[np.argsort(random_array)[::-1]]
        Sigma = np.diag(sigma)
        return Sigma
    
    def Generate_Data(n_observation, n_feature):
        n_rank = min(n_observation, n_feature)
        U_1, U_2 = Generate_Projections(n_rank, n_observation), Generate_Projections(n_rank, n_observation)  
        S = Generate_Sigma(n_rank)
        V_1, V_2 = Generate_Loadings(n_rank, n_feature), Generate_Loadings(n_rank, n_feature)
        return [(U_1 @ S @ V_1.T).T, (U_1 @ S @ V_2.T).T, (U_2 @ S @ V_1.T).T, (U_2 @ S @ V_2.T).T]
    
    def Compare_Subspace_Similarity(data_matrices, method):
        sim = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                sim[i,j] = Calculate_Similarity(data_matrices[i], data_matrices[j], method = method)
        return sim
    
    def Draw_Compare_Subspace_Similarity(Matrices, method):
        Similarity_Index = Compare_Subspace_Similarity(Matrices, method = method)
        if method == 'Pairwise': figtitle = 'Pairwise Cosine Alignment'
        if method == 'CCA': figtitle = 'CCA Coefficient'
        if method == 'RV': figtitle = 'RV Coefficient'
        if method == 'Trace': figtitle = 'Covariance Alignment'
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        heatmap = sns.heatmap(Similarity_Index, ax = axs, cmap = 'YlGnBu', vmin = 0, vmax = 1, square=True, cbar = True)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        cbar.ax.set_yticks([0,0.5,1.0])

        axs.set_aspect('auto')
        for i in range(0, 5, 1):
            axs.axhline(y=i, color='red', linewidth=3)
            axs.axvline(x=i, color='red', linewidth=3)

        labels = ['Matrix1', 'Matrix2', 'Matrix3', 'Matrix4']
        label_positions = [0.5, 1.5, 2.5, 3.5]
        axs.set_xticks(label_positions)
        axs.set_xticklabels(labels, rotation=0, fontsize=28)
        axs.set_yticks(label_positions)
        axs.set_yticklabels(labels, rotation=0, fontsize=28)
        fig.suptitle(figtitle, fontsize = 36)
        plt.tight_layout()
        return fig
            
    Matrices = Generate_Data(n_observation, n_feature)
    
    fig_Pairwise = Draw_Compare_Subspace_Similarity(Matrices, method = 'Pairwise')
    fig_CCA = Draw_Compare_Subspace_Similarity(Matrices, method = 'CCA')
    fig_RV = Draw_Compare_Subspace_Similarity(Matrices, method = 'RV')
    fig_Trace = Draw_Compare_Subspace_Similarity(Matrices, method = 'Trace')
    
    return fig_Pairwise, fig_CCA, fig_RV, fig_Trace
    
    
    
        