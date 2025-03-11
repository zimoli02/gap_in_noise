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

from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

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
    

def Compare_Subspace_Similarity(periods, method, shuffle_method = False):
    n_period = len(periods)
    sim = np.zeros((n_period, n_period))
    for i in range(n_period):
            for j in range(i, n_period):
                sim[i,j] = sim[j, i] = Calculate_Similarity(periods[i], periods[j], method = method,  shuffle_method = shuffle_method)
    return sim

def Comparison_Method_Full_Title(method):
    if method == 'Pairwise': return 'Pairwise Cosine Alignment'
    if method == 'CCA': return 'CCA Coefficient'
    if method == 'RV': return 'RV Coefficient'
    if method == 'Trace': return 'Covariance Alignment'

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
        return [(U_1 @ S @ V_1.T).T, (U_2 @ S @ V_1.T).T, (U_1 @ S @ V_2.T).T, (U_2 @ S @ V_2.T).T]
    
    def Draw_Compare_Subspace_Similarity(Matrices, method):
        Similarity_Index = Compare_Subspace_Similarity(Matrices, method = method)
        figtitle = Comparison_Method_Full_Title(method)
        
        fig, axs = plt.subplots(1, 1, figsize=(12, 10))
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
        axs.set_xticklabels(labels, rotation=0, fontsize=30)
        axs.set_yticks(label_positions)
        axs.set_yticklabels(labels, rotation=0, fontsize=30)
        fig.suptitle(figtitle, fontsize = 40, fontweight='bold')
        plt.tight_layout()
        return fig
            
    Matrices = Generate_Data(n_observation, n_feature)
    
    fig_Pairwise = Draw_Compare_Subspace_Similarity(Matrices, method = 'Pairwise')
    fig_CCA = Draw_Compare_Subspace_Similarity(Matrices, method = 'CCA')
    fig_RV = Draw_Compare_Subspace_Similarity(Matrices, method = 'RV')
    fig_Trace = Draw_Compare_Subspace_Similarity(Matrices, method = 'Trace')
    
    return [fig_Pairwise, fig_CCA, fig_RV, fig_Trace]

def Draw_Standard_Subspace_Location(Group, subspace_name, period_length = 100, offset_delay = 10):
    gap_idx = 9
    gap_dur = int(Group.gaps[gap_idx]*1000)
    if subspace_name == 'On':
        start, end, on_off = 100, 100 + period_length, 1
    if subspace_name == 'Off':
        start, end, on_off = 450 + gap_dur + offset_delay, 450 + gap_dur + period_length + offset_delay, 0
    if subspace_name == 'SustainedNoise':
        start, end, on_off = 350-period_length, 350, 1
    if subspace_name == 'SustainedSilence':
        start, end, on_off = 1000-period_length, 1000, 0
    
    gap_label = Group.gaps_label[gap_idx]
    Sound = gap_label*60 
    Sound += (1-gap_label)*10

    fig, axs = plt.subplots(1,1,figsize = (20, 5))
    axs.plot(np.arange(1000), Sound, color = 'black')
    ymin, ymax = 10, 60
    axs.fill_between(np.arange(len(gap_label)), ymin, ymax, where= gap_label == 1, color='dimgrey', alpha=0.2)
    axs.fill_between(np.arange(len(gap_label))[start:end], ymin, ymax, where= gap_label[start:end] == on_off, color='red', alpha=0.5)
    axs.set_yticks([10,60])
    axs.tick_params(axis='both', labelsize = 28)
    axs.set_xlabel('Time (ms)', fontsize = 32)
    axs.set_ylabel('Sound Level (dB)', fontsize = 32)
    axs.annotate('Noise 1', xy=(180, 62), fontsize=36, color='black')
    axs.annotate('Noise 2', xy=(100 + 250 + 256, 62), fontsize=36, color='black')
    axs.annotate('Gap', xy=(100 + 250 + 100, 12), fontsize=36, color='black')
    axs.set_title('On-Space Location', fontsize = 40, fontweight = 'bold', y = 1.15)

    return fig
    

def Standard_Subspace_Comparison(Group, subspace_name, period_length = 100, offset_delay = 10):
    def Get_Data_Periods(subspace_name):
        periods = []
        for gap_idx in range(10):
            gap_dur = int(Group.gaps[gap_idx]*1000)
            if subspace_name == 'On':
                start, end = 100, 100 + period_length
            if subspace_name == 'Off':
                start, end = 450 + gap_dur + offset_delay, 450 + gap_dur + period_length + offset_delay
            if subspace_name == 'SustainedNoise':
                start, end = 350-period_length, 350
            if subspace_name == 'SustainedSilence':
                start, end = 1000-period_length, 1000
            periods.append(Group.pop_response_stand[:, gap_idx, start:end])
        return periods
    
    def Draw_Compare_Subspace_Similarity_Result(axs, periods, subspace_name, method):
        Similarity_Index = Compare_Subspace_Similarity(periods, method = method)
        
        heatmap = sns.heatmap(Similarity_Index, ax = axs, cmap = 'YlGnBu', vmin = 0, vmax = 1, square=True, cbar = True)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)
        cbar.ax.set_yticks([0,0.5,1.0])

        axs.set_aspect('auto')
        labels = [f'#{i}' for i in range(1, 11, 1)]
        label_positions = [i + 0.5 for i in range(10)]
        axs.set_xticks(label_positions)
        axs.set_xticklabels(labels, rotation=0, fontsize=28)
        axs.set_yticks(label_positions)
        axs.set_yticklabels(labels, rotation=0, fontsize=28)

        return axs
    
    def Draw_Compare_Subspace_Similarity_Result_Test(axs, periods, subspace_name, method):
        def add_significance_bar(ax, x1, x2, y, p):
            h = 0.02
            text_space = 0.03
            if p < 0.001:
                h = 0.015
                #h = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', lw=2)
                ax.text((x1+x2)/2, y+h - text_space, '***', ha='center', va='bottom', size = 32)
            elif p < 0.01:
                #h = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', lw=2)
                ax.text((x1+x2)/2, y+h - text_space, '**', ha='center', va='bottom', size = 32)
            elif p < 0.05:
                #h = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', lw=2)
                ax.text((x1+x2)/2, y+h - text_space, '*', ha='center', va='bottom', size = 32)
        
        Similarity_Indices = []
        for shuffle_method in [False, 'Shuffle_Neuron', 'Add_Neuron', 'Add_Noise', 'Rotate']:
            Similarity_Index = Compare_Subspace_Similarity(periods, method = method, shuffle_method = shuffle_method)
            Similarity_Indices.append(Similarity_Index)
        
        mask = ~np.eye(len(Similarity_Index), dtype=bool)
        Means = [np.mean(Similarity_Index[mask]) for Similarity_Index in Similarity_Indices]
        Stds = [np.std(Similarity_Index[mask]) for Similarity_Index in Similarity_Indices]
        p_values = [stats.ks_2samp(Similarity_Index[mask], Similarity_Indices[0][mask])[1] for Similarity_Index in Similarity_Indices]  
        
        x = np.arange(5)
        colors = ['black', 'red', 'blue', 'green', 'purple']
        axs.bar(x, Means, yerr=np.array(Stds)*3, color=colors, alpha=0.6, capsize=10, width=0.8, error_kw={'capthick': 3, 'elinewidth': 2.5})
        axs.axhline(y=0, linestyle = '--', color = 'black')

        max_y = max(Means) + max(Stds)*3
        for i in range(5):
            add_significance_bar(axs, 0, i, 1 + 0.1*(i-1), p_values[i])

        axs.set_yticks([0, 0.5, 1], labels = [0, 0.5, 1])
        axs.set_xticks([0, 1, 2, 3, 4], labels = ['Orig.', 'Shuffle_Neuron', 'Add_Neuron', 'Add_Noise', 'Rotate'], rotation = 45, ha = 'right')
        axs.tick_params(axis='both', labelsize=28)
        axs.set_ylim(0, 1.33)
        
        return axs
        
    def Draw_Compare_Subspace_Similarity(periods, subspace_name, method):
        fig, axs = plt.subplots(1, 2, figsize = (20, 10), gridspec_kw={'width_ratios': [3,2]})
        axs[0] = Draw_Compare_Subspace_Similarity_Result(axs[0], periods, subspace_name, method)
        axs[1] = Draw_Compare_Subspace_Similarity_Result_Test(axs[1], periods, subspace_name, method)
        fig.suptitle('Sustained Silence Space: Pairwise Cosine Alignment', fontsize = 44, fontweight = 'bold')
        return fig

    periods = Get_Data_Periods(subspace_name)
    fig_Pairwise = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'Pairwise')
    fig_CCA = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'CCA')
    fig_RV = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'RV')
    fig_Trace = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'Trace') 
    return [fig_Pairwise, fig_CCA, fig_RV, fig_Trace]

def Subspace_Similarity_for_All_Gaps(Group, subspace_name, methods, standard_period_length=100, period_length=100, offset_delay = 10):
    def Get_Standard_Period():
        if subspace_name == 'On':
            start, end = 100, 100 + standard_period_length
        if subspace_name == 'Off':
            start, end = 450 + offset_delay, 450 + standard_period_length + offset_delay
        if subspace_name == 'SustainedNoise':
            start, end = 350-standard_period_length, 350
        if subspace_name == 'SustainedSilence':
            start, end = 1000-standard_period_length, 1000
        data = Group.pop_response_stand[:, 0, start:end]
        return data 
    
    def Get_Similarity_Index_per_Gap(gap_idx, standard_period):
        Similarity_Indices = {method:[] for method in methods}
        for t in range(100, 1000):
            period = Group.pop_response_stand[:, gap_idx, t - period_length:t]
            for method in methods:
                Similarity_Index = Calculate_Similarity(standard_period, period, method = method)
                Similarity_Indices[method].append(Similarity_Index)
        return Similarity_Indices
    
    def Draw_Similarity_Index_for_All_Gap(standard_period):
        fig, axs = plt.subplots(10, 1, figsize=(20, 70))  
        check_point = 100
        for gap_idx in range(10):
            Similarity_Indices = Get_Similarity_Index_per_Gap(gap_idx, standard_period)
            gap_dur = round(Group.gaps[gap_idx]*1000)

            for method in methods:
                axs[gap_idx].plot(np.arange(check_point, 1000), Similarity_Indices[method], color = colors[method], linewidth = 6, alpha = 0.9)

            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[gap_idx].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = 'dimgrey', alpha = 0.1)

            axs[gap_idx].set_xticks([], labels = [])
            axs[gap_idx].set_yticks([0,1], labels = ['', ''])
            axs[9].set_xticks([100, 1000], labels = [100, 1000])
            axs[gap_idx].tick_params(axis = 'both', labelsize = 30)
            axs[gap_idx].set_ylabel(f'Gap = {gap_dur} ms', fontsize = 32, fontweight = 'bold')
        axs[9].set_xlabel('Time (ms)', fontsize = 32, fontweight = 'bold')
        fig.suptitle(f'Compare with {subspace_name}-Space', fontsize = 46, fontweight = 'bold', y=0.9)
        
        lines, labels = [], []
        for method in methods:
            line = Line2D([0], [0], color=colors[method], lw=6, alpha = 0.9)
            lines.append(line)
            labels.append(Comparison_Method_Full_Title(method))
        fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.88), ncol=1, fontsize=32)
        return fig
        
    standard_period = Get_Standard_Period()
    colors = {'Pairwise':'#0047AB', 'CCA':'#DC143C', 'RV':'#228B22', 'Trace':'#800080'}
    fig = Draw_Similarity_Index_for_All_Gap(standard_period)
    
    return fig

