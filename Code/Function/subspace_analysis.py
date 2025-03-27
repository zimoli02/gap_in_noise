import os
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
from matplotlib.colors import Normalize
import matplotlib.patches as patches
pal = sns.color_palette('viridis_r', 11)

from . import analysis


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency")

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'eps'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = True

subspacepath = '/Volumes/Research/GapInNoise/Data/Subspace/'

# Basic Functions

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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
        
        k = min(10, len(V1))
        V1 = V1[:k]
        V2 = V2[:k]
        
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

def Comparison_Method_Full_Title(method):
    if method == 'Pairwise': return 'Pairwise Cosine Alignment'
    if method == 'CCA': return 'CCA Coefficient'
    if method == 'RV': return 'RV Coefficient'
    if method == 'Trace': return 'Covariance Alignment'

def sigmoid(x, L, x0, k, c):
    return L / (1 + np.exp(-k * (x - x0))) + c

def inverse_sigmoid(y, L, x0, k, c):
    return x0 - (1 / k) * np.log((L / (y-c)) - 1)

# Non-Specific Plotting

def Draw_Standard_Subspace_Location(Group, subspace_name, period_length = 100, offset_delay = 10):
    gap_idx = 0
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
    axs.fill_between(np.arange(len(gap_label)), ymin, ymax, where= gap_label == 1, color='lightgrey')
    axs.fill_between(np.arange(len(gap_label))[start:end], ymin, ymax, where= gap_label[start:end] == on_off, color='lightcoral')
    axs.set_yticks([10,60])
    axs.tick_params(axis='both', labelsize = 43)
    axs.set_xlabel('Time (ms)', fontsize = 40)
    axs.set_ylabel('Sound Level (dB)', fontsize = 32)
    axs.annotate('Noise 1+2', xy=(190, 62), fontsize=40, color='black')
    #axs.annotate('Noise 1', xy=(180, 62), fontsize=36, color='black')
    #axs.annotate('Noise 2', xy=(100 + 250 + 256, 62), fontsize=36, color='black')
    #axs.annotate('Gap', xy=(100 + 250 + 100, 12), fontsize=36, color='black')
    axs.set_title(f'{subspace_name}-Space Location', fontsize = 54, fontweight = 'bold', y = 1.15)

    return fig

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


# Group-Specific Analysis

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
        fig.suptitle(f'{subspace_name} Space: {Comparison_Method_Full_Title(method)}', fontsize = 44, fontweight = 'bold')
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
    
    def Get_Similarity_Index_for_All_Gap(standard_period):
        label = Group.geno_type + '_' + Group.hearing_type
        file_path = check_path(subspacepath + f'SubspaceEvolution/{subspace_name}/')
        try:
            with open(file_path + f'{label}.pkl', 'rb') as f:
                data = pickle.load(f)
            print('Data Existed!')
        except FileNotFoundError:
            data = {}
            for gap_idx in range(10):
                data[gap_idx] = Get_Similarity_Index_per_Gap(gap_idx, standard_period)
            with open(file_path + f'{label}.pkl', 'wb') as handle:
                pickle.dump(data, handle)
        return data
       
    def Draw_Similarity_Index_for_All_Gap(Similarity_Index_for_All_Gap):
        fig, axs = plt.subplots(10, 1, figsize=(20, 70))  
        check_point = 100
        for gap_idx in range(10):
            Similarity_Indices = Similarity_Index_for_All_Gap[gap_idx]
            gap_dur = round(Group.gaps[gap_idx]*1000)

            for method in methods:
                axs[gap_idx].plot(np.arange(check_point, 1000), Similarity_Indices[method], color = colors[method], linewidth = 7, alpha = 0.9)

            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[gap_idx].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = 'lightgrey')

            axs[gap_idx].set_xticks([], labels = [])
            axs[gap_idx].set_yticks([0,1], labels = [0, 1])
            axs[9].set_xticks([100, 1000], labels = [100, 1000])
            axs[gap_idx].tick_params(axis = 'both', labelsize = 36)
            axs[gap_idx].set_ylabel(f'Gap = {gap_dur} ms', fontsize = 36, fontweight = 'bold')
        axs[9].set_xlabel('Time (ms)', fontsize = 40, fontweight = 'bold')
        fig.suptitle(f'Similarity with {subspace_name}-Space', fontsize = 54, fontweight = 'bold', y=0.9)
        
        lines, labels = [], []
        for method in methods:
            line = Line2D([0], [0], color=colors[method], lw=6, alpha = 0.9)
            lines.append(line)
            labels.append(Comparison_Method_Full_Title(method))
        legend = fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.88), ncol=1, fontsize=32)
        legend.get_frame().set_facecolor('white')  # White background
        legend.get_frame().set_alpha(1.0)         # Fully opaque
        legend.get_frame().set_linewidth(1.5)     # Add border
        legend.get_frame().set_edgecolor('black') # Black border
        
        return fig
    
    def Justify_the_Separation_Level_for_each_Space_each_Method():
        subspacenames = ['On', 'Off', 'SustainedNoise', 'SustainedSilence']
        for i in range(len(subspacenames)):
            subspacename = subspacenames[i]
            if subspacename != subspace_name: continue
            file_path = subspacepath + f'SubspaceEvolution/{subspacename}/'
            with open(file_path + f'{label}.pkl', 'rb') as f:
                data = pickle.load(f)
            print('Data Existed!')
                
            fig, axs = plt.subplots(2, 2, figsize = (20, 20))
            axs = axs.flatten()
            for j in range(len(methods)):
                method = methods[j]
                Similarity = np.array(data[9][method])  
                
                On_sim = Similarity[:standard_period_length]
                SustainedNoise_sim = Similarity[250-standard_period_length:250]
                Off_sim = Similarity[250+offset_delay:250+offset_delay+standard_period_length]
                SustainedSilence_sim = Similarity[-standard_period_length:]
                
                Similarities = np.array([On_sim, Off_sim, SustainedNoise_sim, SustainedSilence_sim])
                Means = np.array([np.mean(sim) for sim in Similarities])
                Stds = np.array([np.std(sim) for sim in Similarities])
                
                p_values = [stats.ttest_ind(Similarities[i], Similarities[k])[1] for k in range(4)] 
                
                axs[j].bar([0,1,2,3], Means, yerr=np.array(Stds)*3, color=colors[method], alpha=0.6, capsize=10, width=0.8, error_kw={'capthick': 3, 'elinewidth': 2.5})
                
                max_y = max(Means) + max(Stds)*3 + 0.05
                for k in range(4):
                    if k == i: continue
                    add_significance_bar(axs[j], i, k, max_y + 0.1*(k-1), p_values[k])

                
                axs[j].set_yticks([0, 1, 2], labels = [0, 1, 2])
                axs[j].set_xticks([0,1,2,3], ['On', 'Off', 'S.N.','S.L.'], rotation = 45, ha = 'center')
                axs[j].tick_params(axis='both', labelsize=28)
                axs[j].set_ylim(-0.5, 2)
                axs[j].set_title(Comparison_Method_Full_Title(method), fontsize = 34)

            fig.suptitle(f'Similarity with {subspacename}-Space\nduring Different Periods', fontsize = 54, fontweight = 'bold') 
        return fig
    
    label = Group.geno_type + '_' + Group.hearing_type
    standard_period = Get_Standard_Period()
    colors = {'Pairwise':'#0047AB', 'CCA':'#DC143C', 'RV':'#228B22', 'Trace':'#800080'}
    Similarity_Index_for_All_Gap = Get_Similarity_Index_for_All_Gap(standard_period)
    fig = Draw_Similarity_Index_for_All_Gap(Similarity_Index_for_All_Gap)
    fig_justification = Justify_the_Separation_Level_for_each_Space_each_Method()
    
    return fig, fig_justification

def Period_Capacity_in_Subspace_Comparison(Group, method, max_on_capacity = 75, max_off_capacity = 100, max_timewindow = 100, offset_delay = 10):
    def Find_Best_Period_Capacity():
        on_capacities = np.arange(2, max_on_capacity+1)
        off_capacities = np.arange(2, max_off_capacity+1)
        timewindows = np.arange(5, max_timewindow+1, 5)
        
        separate_level = np.zeros((len(on_capacities), len(off_capacities), len(timewindows)))

        # Precompute standard periods for all onset and offset capacities
        standard_on_periods = [
            Group.pop_response_stand[:, 0, 100: 100 + onset_capacity] for onset_capacity in on_capacities
        ]
        standard_off_periods = [
            Group.pop_response_stand[:, 0, 450 + offset_delay: 450 + offset_delay + offset_capacity] for offset_capacity in off_capacities
        ]
        
        for T in range(len(timewindows)):
            period_length = timewindows[T]
            on_off_ratio = np.zeros(len(on_capacities))
            off_on_ratio = np.zeros(len(off_capacities))
            
            # Extract all periods at once
            periods = np.array([
                Group.pop_response_stand[:, 9, t - period_length:t] for t in range(100, 1000)
            ])
            
            for i in range(len(on_capacities)):
                standard_on_period = standard_on_periods[i]
                centered_standard_on_period = Center(standard_on_period)
                
                on_similarity = np.array([
                    Calculate_Similarity(period, centered_standard_on_period, method = method) for period in periods
                ])

                # Extract relevant indices and compute ratio
                max_on_sim_onset = np.max(on_similarity[100-100:200-100])
                max_on_sim_offset = np.max(on_similarity[350 + offset_delay -100:450 + offset_delay - 100])
                on_off_ratio[i] = max_on_sim_onset / max_on_sim_offset
                
            for j in range(len(off_capacities)):
                standard_off_period = standard_off_periods[j]
                centered_standard_off_period = Center(standard_off_period)
                
                off_similarity = np.array([
                    Calculate_Similarity(period, centered_standard_off_period, method = method) for period in periods
                ])

                # Extract relevant indices and compute ratio
                max_off_sim_offset = np.max(off_similarity[350 + offset_delay -100:450 + offset_delay - 100])
                max_off_sim_onset = np.max(off_similarity[100-100:200-100])
                off_on_ratio[j] = max_off_sim_offset / max_off_sim_onset

            for i in range(len(on_capacities)):
                for j in range(len(off_capacities)):
                    separate_level[i, j, T] = on_off_ratio[i] * off_on_ratio[j]
        
        return on_capacities, off_capacities, timewindows, separate_level
    
    def Determine_Best_Capacity(on_capacities, off_capacities, timewindows, separate_level):
        maximum_level = np.max(separate_level)
        threshold = maximum_level*0.5

        parameters = []
        for i in range(len(on_capacities)):
            for j in range(len(off_capacities)):
                for k in range(len(timewindows)):
                    if abs(separate_level[i,j,k] - threshold) < 1e-3: parameters.append([on_capacities[i], off_capacities[j], timewindows[k]])
        centroid = np.mean(np.array(parameters), axis = 0)
        best_on_capacity, best_off_capacity, best_timewindow = round(centroid[0]), round(centroid[1]), round(centroid[2])
        return best_on_capacity, best_off_capacity, best_timewindow
    
    def Draw_Compare_Period_Capacity(on_capacities, off_capacities, timewindows, separate_level):
        
        best_on_capacity, best_off_capacity, best_timewindow = Determine_Best_Capacity(on_capacities, off_capacities, timewindows, separate_level)
        for i in range(len(on_capacities)):
            if abs(on_capacities[i] - best_on_capacity) < 0.5: 
                best_on_idx = i
                break
        for i in range(len(off_capacities)):
            if abs(off_capacities[i] - best_off_capacity) < 0.5: 
                best_off_idx = i
                break
        for i in range(len(timewindows)):
            if abs(timewindows[i] - best_timewindow) < 5: 
                best_timewindow_idx = i
                break

        # Create meshgrid and extract coordinates/values for 3D plot
        X, Y, Z = np.meshgrid(np.arange(len(on_capacities)), 
                            np.arange(len(off_capacities)), 
                            np.arange(len(timewindows)), 
                            indexing='ij')

        # Flatten for 3D scatter
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        values = separate_level[x_flat, y_flat, z_flat]

        # Create a common colormap normalization
        norm = Normalize(vmin=0, vmax=1)
        cmap = 'YlGnBu'

        tick_size = 36
        label_size = 40
        title_size = 44
        labelpad = 30

        fig = plt.figure(figsize=(45, 10))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.4)

        # 1. First subplot: 3D scatter
        ax1 = fig.add_subplot(gs[0], projection='3d')
        sizes = 15 + (80 - 15) * (values)
        scatter = ax1.scatter(x_flat, y_flat, z_flat, 
                            c=values,
                            cmap=cmap,
                            s=sizes,
                            alpha=1,
                            edgecolors='none',
                            norm=norm)

        ax1.set_xticks([0, 23, 48, 73])
        ax1.set_xticklabels([on_capacities[0], on_capacities[23], on_capacities[48], on_capacities[73]], fontsize=tick_size)
        ax1.set_yticks([0, 48, 98])
        ax1.set_yticklabels([off_capacities[0], off_capacities[48], off_capacities[98]], fontsize=tick_size)
        ax1.set_zticks([0, 10, 19])
        ax1.set_zticklabels([timewindows[0], timewindows[10], timewindows[19]], fontsize=tick_size)

        ax1.set_xlabel('On-Space (ms)', fontsize=label_size, labelpad=labelpad)
        ax1.set_ylabel('Off-Space (ms)', fontsize=label_size, labelpad=labelpad)
        ax1.set_zlabel('Time Window (ms)', fontsize=label_size, labelpad=labelpad)
        ax1.set_title('3D View', fontsize=title_size, pad=12)
        ax1.view_init(elev=30, azim=-75)

        # 2. Second subplot: 2D heatmap - Mean along Z-axis (Time Window)
        ax2 = fig.add_subplot(gs[1])
        mean_xy = np.mean(separate_level, axis=2)
        heatmap2 = ax2.imshow(mean_xy.T, 
                            aspect='auto', 
                            origin='lower',
                            cmap=cmap,
                            norm=norm)
        ax2.axvline(x = best_on_idx, linestyle = '-', color = 'red', linewidth = 4)
        ax2.axhline(y = best_off_idx, linestyle = '-', color = 'red', linewidth = 4)
        ax2.text(best_on_idx, -2, f"{best_on_capacity}", 
                color='red', fontsize=tick_size, ha='center', va='top')
        ax2.text(-2, best_off_idx, f"{best_off_capacity}", 
                color='red', fontsize=tick_size, ha='right', va='center')

        ax2.set_xticks([0, 73])
        ax2.set_xticklabels([on_capacities[0], on_capacities[73]], fontsize=tick_size)
        ax2.set_yticks([0, 98])
        ax2.set_yticklabels([off_capacities[0], off_capacities[98]], fontsize=tick_size)

        ax2.set_xlabel('On-Space (ms)', fontsize=label_size)
        ax2.set_ylabel('Off-Space (ms)', fontsize=label_size)
        ax2.set_title('Mean across\nTime Windows', fontsize=title_size)

        # 3. Third subplot: 2D heatmap - Mean along Y-axis (Off-Space)
        ax3 = fig.add_subplot(gs[2])
        mean_xz = np.mean(separate_level, axis=1)
        heatmap3 = ax3.imshow(mean_xz.T, 
                            aspect='auto', 
                            origin='lower',
                            cmap=cmap,
                            norm=norm)

        ax3.axvline(x = best_on_idx, linestyle = '-', color = 'red', linewidth = 4)
        ax3.axhline(y = best_timewindow_idx, linestyle = '-', color = 'red', linewidth = 4)
        ax3.text(best_on_idx, -1, f"{best_on_capacity}", 
                color='red', fontsize=tick_size, ha='center', va='top')
        ax3.text(-1, best_timewindow_idx, f"{best_timewindow}", 
                color='red', fontsize=tick_size, ha='right', va='center')

        ax3.set_xticks([0,73])
        ax3.set_xticklabels([on_capacities[0], on_capacities[73]], fontsize=tick_size)
        ax3.set_yticks([0, 19])
        ax3.set_yticklabels([timewindows[0], timewindows[19]], fontsize=tick_size)

        ax3.set_xlabel('On-Space (ms)', fontsize=label_size)
        ax3.set_ylabel('Time Window (ms)', fontsize=label_size)
        ax3.set_title('Mean across\nOff-Space', fontsize=title_size)

        # 4. Fourth subplot: 2D heatmap - Mean along X-axis (On-Space)
        ax4 = fig.add_subplot(gs[3])
        mean_yz = np.mean(separate_level, axis=0)
        heatmap4 = ax4.imshow(mean_yz.T, 
                            aspect='auto', 
                            origin='lower',
                            cmap=cmap,
                            norm = norm)

        ax4.axvline(x = best_off_idx, linestyle = '-', color = 'red', linewidth = 4)
        ax4.axhline(y = best_timewindow_idx, linestyle = '-', color = 'red', linewidth = 4)
        ax4.text(best_off_idx, -1, f"{best_off_capacity}", 
                color='red', fontsize=tick_size, ha='center', va='top')
        ax4.text(-1, best_timewindow_idx, f"{best_timewindow}", 
                color='red', fontsize=tick_size, ha='right', va='center')

        ax4.set_xticks([0, 98])
        ax4.set_xticklabels([off_capacities[0], off_capacities[98]], fontsize=tick_size)
        ax4.set_yticks([0, 19])
        ax4.set_yticklabels([timewindows[0], timewindows[19]], fontsize=tick_size)

        ax4.set_xlabel('Off-Space (ms)', fontsize=label_size)
        ax4.set_ylabel('Time Window (ms)', fontsize=label_size)
        ax4.set_title('Mean across\nOn-Space', fontsize=title_size)

        # Add a single colorbar for all subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_ticks(np.linspace(0, 1, 6))
        cbar.ax.tick_params(labelsize=tick_size)

        fig.suptitle('Separation Level for On/Off-Subspace Similarity', fontsize=54, fontweight='bold', y=1.1)
        plt.subplots_adjust(right=0.9, wspace=0.3)
        
        return fig
    
    def Get_Similarity_Index_per_Gap(gap_idx, standard_period, timewindow):
        Similarity_Index = []
        for t in range(100, 1000):
            period = Group.pop_response_stand[:, gap_idx, t - timewindow:t]
            Similarity_Index.append(Calculate_Similarity(standard_period, period, method = method))
        return Similarity_Index
    
    def Get_Similarity_Index_for_All_Gap_All_Subspace(on_capacity, off_capacity, timewindow):
        file_path = check_path(subspacepath + f'BestSubspaceComparison/{method}/')
        try:
            with open(file_path + f'{label}.pkl', 'rb') as f:
                Similarities = pickle.load(f)
                On_Similarities = Similarities['On']
                Off_Similarities = Similarities['Off']
            print('Data Existed!')
        except FileNotFoundError:
            standard_on_period = Group.pop_response_stand[:, 0, 100:100 + on_capacity]
            standard_off_period = Group.pop_response_stand[:, 0, 450 + offset_delay:450 + offset_delay + off_capacity]
            On_Similarities, Off_Similarities = {}, {}
            for gap_idx in range(10):
                On_Similarity_Index = Get_Similarity_Index_per_Gap(gap_idx, standard_on_period, timewindow)
                Off_Similarity_Index = Get_Similarity_Index_per_Gap(gap_idx, standard_off_period, timewindow)
                On_Similarities[gap_idx] = On_Similarity_Index
                Off_Similarities[gap_idx] = Off_Similarity_Index
            Similarities = {'On':On_Similarities, 'Off': Off_Similarities}
            with open(file_path + f'{label}.pkl', 'wb') as handle:
                pickle.dump(Similarities, handle)

    label = Group.geno_type + '_' + Group.hearing_type
    try:
        with np.load(subspacepath + f'PeriodCapacity/{method}/{label}.npz') as data:
            on_capacities = data['on_capacities']
            off_capacities = data['off_capacities']
            timewindows = data['timewindows']
            separate_level = data['separate_level']
        print('Data Existed!')
    except FileNotFoundError:
        file_path = check_path(subspacepath + f'PeriodCapacity/{method}/')
        on_capacities, off_capacities, timewindows, separate_level = Find_Best_Period_Capacity()
        np.savez(file_path + f'{label}.npz', 
            on_capacities=on_capacities, 
            off_capacities=off_capacities, 
            timewindows=timewindows, 
            separate_level=separate_level)
    
    on_capacity, off_capacity, timewindow = Determine_Best_Capacity(on_capacities, off_capacities, timewindows, separate_level)
    fig_best_capacity = Draw_Compare_Period_Capacity(on_capacities, off_capacities, timewindows, separate_level)
    
    Get_Similarity_Index_for_All_Gap_All_Subspace(on_capacity, off_capacity, timewindow)
    
    return fig_best_capacity

def Best_Subspace_Comparison(Group, method):
    def Draw_Similarity_Index_for_All_Gap_All_Subspace(On_Similarities, Off_Similarities):
        fig, axs = plt.subplots(10, 1, figsize=(16, 35))  
        check_point = 100
        for gap_idx in range(10):
            On_Similarity_Index = On_Similarities[gap_idx]
            Off_Similarity_Index = Off_Similarities[gap_idx]
            
            gap_dur = round(Group.gaps[gap_idx]*1000)
            axs[gap_idx].plot(np.arange(check_point, 1000), On_Similarity_Index, color = 'red', linewidth = 7)
            axs[gap_idx].plot(np.arange(check_point, 1000), Off_Similarity_Index, color = 'blue', linewidth = 7)
            
            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[gap_idx].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = 'lightgrey')

            axs[gap_idx].set_xticks([], labels = [])
            axs[gap_idx].set_yticks([0,1], labels = [0, 1])
            axs[9].set_xticks([100, 1000], labels = [100, 1000])
            axs[gap_idx].tick_params(axis = 'both', labelsize = 36)
            axs[gap_idx].set_ylabel(f'Gap = {gap_dur} ms', fontsize = 40, fontweight = 'bold')
        axs[9].set_xlabel('Time (ms)', fontsize = 36, fontweight = 'bold')
        fig.suptitle(f'Compare with Best Standard Spaces', fontsize = 54, fontweight = 'bold', y=0.9)
        
        on_line = Line2D([0], [0], color='red', lw=6)
        off_line = Line2D([0], [0], color='blue', lw=6)
        lines, labels = [on_line, off_line], ['On-Similarity', 'Off-Similarity']
        fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.88), ncol=1, fontsize=32)
        return fig
    
    def Draw_Find_Period_Capacity_in_Subspace_Comparison(On_Similarities, Off_Similarities):
        gap_idx = 9
        gap_dur = round(Group.gaps[gap_idx]*1000)
        x, y = On_Similarities[gap_idx], Off_Similarities[gap_idx]
        fig, axs = plt.subplots(1, 2, figsize = (80, 10))

        for i in range(2):
            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[i].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = 'gainsboro')
            
        axs[0].plot(np.arange(100, 1000), x, color = 'red', lw=7)
        axs[1].plot(np.arange(100, 1000), y, color = 'blue', lw=7)

        dotsize = 1000
        max_on_in_x = (np.max(x[:100]), np.argsort(x[:100])[::-1][0])
        axs[0].scatter(max_on_in_x[1] + 100, max_on_in_x[0], color = 'darkmagenta', s=dotsize, label = 'Max On-Similarity during Onset')
        max_off_in_x = (np.max(x[250 + 10:250 + 10 + 100]), np.argsort(x[250 + 10:250 + 10 + 100])[::-1][0])
        axs[0].scatter(max_off_in_x[1] + 250 + 10 + 100, max_off_in_x[0], color = 'magenta', s=dotsize, label = 'Max On-Similarity during Offset')

        max_on_in_y = (np.max(y[:100]), np.argsort(y[:100])[::-1][0])
        axs[1].scatter(max_on_in_y[1] + 100, max_on_in_y[0], color = 'darkmagenta', s=dotsize, label = 'Max Off-Similarity during Onset')
        max_off_in_y = (np.max(y[250 + 10:250 + 10 + 100]), np.argsort(y[250 + 10:250 + 10 + 100])[::-1][0])
        axs[1].scatter(max_off_in_y[1] + 250 + 10 + 100, max_off_in_y[0], color = 'magenta', s=dotsize, label = 'Max Off-Similarity during Offset')

        for i in range(2):
            rect = patches.Rectangle((100, 0),  # Bottom-left corner
                                100,     # Width
                                1,     # Height
                                linewidth=10, linestyle = '--', 
                                edgecolor='saddlebrown', facecolor='none')  # Customize appearance
            axs[i].add_patch(rect)
            
            rect = patches.Rectangle((350 + 10, 0),  # Bottom-left corner
                                100,     # Width
                                1,     # Height
                                linewidth=10, linestyle = '--', 
                                edgecolor='saddlebrown', facecolor='none')  # Customize appearance
            axs[i].add_patch(rect)
            
            # Add double-headed arrows for heights with labels
            # Define height points and labels
            heights = []
            if i == 0:
                heights = [
                    (max_on_in_x[1] + 100, 0, max_on_in_x[0]-0.05, "$h_1$"),
                    (max_off_in_x[1] + 250 + 10 + 100, 0, max_off_in_x[0]-0.05, "$h_2$")
                ]
            else:
                heights = [
                    (max_on_in_y[1] + 100, 0, max_on_in_y[0]-0.05, "$h_3$"),
                    (max_off_in_y[1] + 250 + 10 + 100, 0, max_off_in_y[0]-0.05, "$h_4$")
                ]
            
            # Add arrows and labels
            for x_pos, y_min, y_max, label in heights:
                # Add vertical double-headed arrow
                arrow = patches.FancyArrowPatch(
                    (x_pos, y_min), (x_pos, y_max),
                    arrowstyle='<->', 
                    mutation_scale=50,  # Scale the arrow head
                    linewidth=5,
                    color='black'
                )
                axs[i].add_patch(arrow)
                
                # Add text label to the right of the arrow
                axs[i].text(x_pos + 10, y_max/2, label, 
                        fontsize=50, fontweight='bold', 
                        verticalalignment='center')
                
            legend = axs[i].legend(loc = 'upper right', fontsize = 44)
            legend.get_frame().set_facecolor('white')  # White background
            legend.get_frame().set_alpha(1.0)         # Fully opaque
            legend.get_frame().set_linewidth(1.5)     # Add border
            legend.get_frame().set_edgecolor('black') # Black border
            
            axs[i].set_xticks([100, 1000], labels = [100, 1000])
            axs[i].set_yticks([0,1], labels = [0, 1])
            axs[i].tick_params(axis = 'both', labelsize = 48)
            axs[i].set_ylabel(f'Gap = {gap_dur} ms', fontsize = 54, fontweight = 'bold')
            axs[i].set_xlabel('Time (ms)', fontsize = 54, fontweight = 'bold')
        fig.suptitle(f'Compute Separation Level of Subspace Comparison', fontsize = 72, fontweight = 'bold', y=1.1)
        return fig
    
    def Find_Threshold(array):
        diff = np.diff(array)
        max_increase = np.max(diff)
        for i in range(len(diff)):
            if diff[i] > max_increase/5: 
                return i
                
    def Draw_On_Similarity_Summary(Similarity_Indices, plot_length, subspace_name):
        average_similarity_index = Similarity_Indices[1]
        start, end = 0, 0 + plot_length
        average_similarity_index = average_similarity_index[start:end]
        
        fig1, axs = plt.subplots(1, 1, figsize=(10, 10))  
        delays = []
        peak_values = []
        for gap_idx in range(10):
            gap_dur = round(Group.gaps[gap_idx]*1000)
            start, end = 250 + gap_dur, 250 + gap_dur + plot_length
            
            Similarity_Index = Similarity_Indices[gap_idx]
            axs.plot(np.arange(plot_length), Similarity_Index[start:end], color = pal[gap_idx], linewidth = 6)
                
            delays.append(Find_Threshold(Similarity_Index[start:end]))
            peak_values.append(np.max(Similarity_Index[start:end]))
            
        axs.plot(np.arange(plot_length), average_similarity_index, color = 'lightcoral', linewidth = 8, label = f'{subspace_name}-Subspace Evolution') 
        axs.axvline(x = np.mean(delays), color = 'red', linestyle = ':', linewidth = 4, label = f'Delay = {np.mean(delays)}ms')
        axs.legend(loc = 'upper right', fontsize = 32)
        
        axs.set_xlim((0, plot_length))
        axs.set_xticks([0, 50, 100], labels = [0,  50, 100])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_ylabel('Subspace Similarity', fontsize = 40)
        axs.set_xlabel('Noise 2 Onset (ms)', fontsize = 40)
        fig1.suptitle('Post-Gap Onset\nOn-Similarity', fontsize = 54, fontweight = 'bold')
        
        fig2, axs = plt.subplots(1, 1, figsize=(10, 10))  
        for gap_idx in range(1, 10):
            axs.scatter(gap_idx, peak_values[gap_idx], color = pal[gap_idx], s = 400)
        axs.plot(np.arange(1, 10), peak_values[1:], color = 'blue', linewidth = 5)
        #axs.legend(loc = 'upper left', fontsize = 24)
        
        axs.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_xlabel(f'Gap Duration (ms)', fontsize = 40)
        axs.set_ylabel('Similarity Index', fontsize = 40)
        fig2.suptitle(f'Max. {subspace_name} Similarity', fontsize = 54, fontweight = 'bold')
        
        return fig1, fig2
           
    def Draw_Off_Similarity_Summary(Similarity_Indices, plot_length, subspace_name):
        average_similarity_index = Similarity_Indices[1]
        start, end= 350 + 1, 350 + 1 + plot_length
        average_similarity_index = average_similarity_index[start:end]
        
        fig1, axs = plt.subplots(1, 1, figsize=(10, 10))  
        delays = []
        peak_values = []
        for gap_idx in range(10):
            gap_dur = round(Group.gaps[gap_idx]*1000)
            Similarity_Index = Similarity_Indices[gap_idx]
            start, end= 250, 250 + gap_dur + 100
            axs.plot(np.arange(100 + gap_dur), Similarity_Index[start:end], color = pal[gap_idx], linewidth = 6)
            axs.plot([gap_dur, gap_dur], [0, Similarity_Index[250 + gap_dur]], color = 'grey', linestyle = ':', linewidth = 4)
            delays.append(Find_Threshold(Similarity_Index[start:end]))
            peak_values.append(np.max(Similarity_Index[start:end]))
        axs.plot(np.arange(plot_length), average_similarity_index, color = 'lightcoral', linewidth = 8, label = f'{subspace_name}-Subspace Evolution') 
        axs.axvline(x = np.mean(delays), color = 'red', linestyle = ':', linewidth = 4, label = f'Delay = {np.mean(delays)}ms')
        axs.plot([], [], color = 'grey', linestyle = ':', linewidth = 4, label = 'Noise 2 Starts')
        axs.legend(loc = 'upper right', fontsize = 32)
        
        axs.set_xlim((0, plot_length))
        axs.set_xticks([0, 50, 100], labels = [0,  50, 100])
        axs.set_yticks([0,1], labels = [0,1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_ylabel('Subspace Similarity', fontsize = 40)
        axs.set_xlabel('Noise 1 Offset (ms)', fontsize = 40)
        fig1.suptitle('Pre-Gap Offset\nOff-Similarity', fontsize = 54, fontweight = 'bold')
        
        fig2, axs = plt.subplots(1, 1, figsize=(10, 10))  
        for gap_idx in range(1, 10):
            axs.scatter(gap_idx, peak_values[gap_idx], color = pal[gap_idx], s = 400)
        axs.plot(np.arange(1, 10), peak_values[1:], color = 'green', linewidth = 5)
        
        axs.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_xlabel(f'Gap Duration (ms)', fontsize = 40)
        axs.set_ylabel('Similarity Index', fontsize = 40)
        fig2.suptitle(f'Max. {subspace_name} Similarity', fontsize = 54, fontweight = 'bold')
        
        return fig1, fig2
    
    label = Group.geno_type + '_' + Group.hearing_type
    file_path = check_path(subspacepath + f'BestSubspaceComparison/{method}/')
    with open(file_path + f'{label}.pkl', 'rb') as f:
        Similarities = pickle.load(f)
    On_Similarities = Similarities['On']
    Off_Similarities = Similarities['Off']
    
    fig_best_subspace_comparison = Draw_Similarity_Index_for_All_Gap_All_Subspace(On_Similarities, Off_Similarities)
    fig_explain_find_best_subspace = Draw_Find_Period_Capacity_in_Subspace_Comparison(On_Similarities, Off_Similarities)
    fig_on_similarity_evolution, fig_on_similarity_peak = Draw_On_Similarity_Summary(On_Similarities, plot_length = 100, subspace_name = 'On')
    fig_off_similarity_evolution, fig_off_similarity_peak = Draw_Off_Similarity_Summary(Off_Similarities, plot_length = 100, subspace_name = 'Off')
    
    return fig_best_subspace_comparison, fig_explain_find_best_subspace, [fig_on_similarity_evolution, fig_on_similarity_peak], [fig_off_similarity_evolution, fig_off_similarity_peak]


# Summary for All Groups

def Best_Subspace_Comparison_All_Group_Property(Groups, method, optimised_param = True):
    def Draw_On_Similarity_Properties():
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))  
        for i, (label, Group) in enumerate(Groups.items()):
            if optimised_param:
                file_path = check_path(subspacepath + f'BestSubspaceComparison/{method}/')
                with open(file_path + f'{label}.pkl', 'rb') as f:
                    Similarities = pickle.load(f)
                On_Similarities = Similarities['On']
            else:
                file_path = check_path(subspacepath + f'SubspaceEvolution/On/')
                with open(file_path + f'{label}.pkl', 'rb') as f:
                    data = pickle.load(f)
                On_Similarities = np.array([data[i][method] for i in range(10)])   
            
            peak_values = []
            for gap_idx in range(10):
                gap_dur = round(Group.gaps[gap_idx]*1000)
                start, end = 250 + gap_dur, 250 + gap_dur + 100 
                Similarity_Index = On_Similarities[gap_idx]
                peak_values.append(np.max(Similarity_Index[start:end]))
                axs.scatter(gap_idx, peak_values[gap_idx], color = pal[gap_idx], s = 400)
            axs.plot(np.arange(10), peak_values, color = colors[label], linewidth = 5, label = label)
        
        axs.legend(loc = 'upper left', fontsize = 28)
        axs.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_xlabel(f'Gap Duration (ms)', fontsize = 40)
        axs.set_ylabel('Similarity Index', fontsize = 40)
        fig.suptitle(f'Max. On-Similarity', fontsize = 54, fontweight = 'bold')
        return fig
    
    def Draw_Off_Similarity_Properties():

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))  
        fig1, axs1 = plt.subplots(1, 2, figsize=(8, 10))  
        fig2, axs2 = plt.subplots(1, 1, figsize=(4, 10))   
        for i, (label, Group) in enumerate(Groups.items()):
            if optimised_param:
                file_path = check_path(subspacepath + f'BestSubspaceComparison/{method}/')
                with open(file_path + f'{label}.pkl', 'rb') as f:
                    Similarities = pickle.load(f)
                Off_Similarities = Similarities['Off']
            else:
                file_path = check_path(subspacepath + f'SubspaceEvolution/Off/')
                with open(file_path + f'{label}.pkl', 'rb') as f:
                    data = pickle.load(f)
                Off_Similarities = np.array([data[i][method] for i in range(10)])
            
            peak_values = []
            for gap_idx in range(10):
                gap_dur = round(Group.gaps[gap_idx]*1000)
                start, end = 250, 250 + 100 
                Similarity_Index = Off_Similarities[gap_idx]
                peak_values.append(np.max(Similarity_Index[start:end]))
                axs.scatter(gap_idx, peak_values[gap_idx], color = pal[gap_idx], s = 400)
            
            x_data = np.arange(9)
            y_data = peak_values[1:]
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[max(y_data), np.median(x_data), 1, 0])
            L_fit, x0_fit, k_fit, c_fit = popt
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = sigmoid(x_fit, *popt)
            r2 = r2_score(y_data, sigmoid(x_data, *popt))
            print(f'{label}: R-Squared for sigmoidal fit is {r2} when parameter optimisation is {optimised_param}')
            axs.plot(x_fit+1, y_fit, color = colors[label], linewidth = 5, label = label)
            
            y_percents = [0.01, 0.99]
            titles = ['Lower Bound.', 'Upper Bound.']
            for j in range(2):
                y_percent = y_percents[j]
                x_ = (L_fit-c_fit)*y_percent + c_fit
                axs1[j].bar(i, x_, color = colors[label], width=0.8)
            
            y_percent = 0.5
            x_ = inverse_sigmoid((L_fit-c_fit)*y_percent + c_fit, *popt)
            threshold_gap_dur = np.exp2(x_)
            axs2.bar(i, threshold_gap_dur, color = colors[label], width=0.8)
                
        axs.legend(loc = 'upper left', fontsize = 28)
        axs.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_xlabel(f'Gap Duration (ms)', fontsize = 40)
        axs.set_ylabel('Similarity Index', fontsize = 40)
        fig.suptitle(f'Max. Off-Similarity', fontsize = 54, fontweight = 'bold')
        
        for i in range(2):
            axs1[i].set_yticks([0,1], labels = [0, 1])
            axs1[i].set_xticks([0,1,2,3], ['', '', '',''])
            axs1[i].tick_params(axis='both', labelsize=28)
            axs1[i].set_ylim(0, 1)
            axs1[i].set_ylabel('Off-Similarity', fontsize = 28)
            axs1[i].set_title(titles[i], fontsize = 34)
        
        axs2.set_yticks([0, 10, 20], labels = [0, 10, 20])
        axs2.set_xticks([0,1,2,3], ['', '', '',''])
        axs2.tick_params(axis='both', labelsize=28)
        axs2.set_ylim(0, 20)
        axs2.set_ylabel('Gap Duration (ms)', fontsize = 28)
        axs2.set_title('Threshold', fontsize = 34)
        
        return fig, fig1, fig2
    
    colors = {'WT_NonHL': 'red', 'WT_HL':'orange', 'Df1_NonHL':'black', 'Df1_HL':'grey'}
    fig_on =  Draw_On_Similarity_Properties()
    fig_off, fig_off_boundary, fig_off_threshold = Draw_Off_Similarity_Properties()
    
    return fig_on, fig_off, fig_off_boundary, fig_off_threshold