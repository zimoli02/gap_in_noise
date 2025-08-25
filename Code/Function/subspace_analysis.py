import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import pickle

import scipy.stats as stats 
from scipy.stats import sem
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.linalg import svd, orth
from sklearn.decomposition import PCA as SKPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib as mpl
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


################################################## Colors ##################################################
response_colors = {'on': 'olive', 'off': 'dodgerblue', 'both': 'darkorange', 'none':'grey'}
response_psth_colors = {'on': 'darkkhaki', 'off': 'lightskyblue', 'both': 'bisque', 'none':'lightgrey'}
shape_colors = {1: 'pink', 2: 'lightblue', 0:'grey'}
gap_colors = pal
group_colors =  {'WT_NonHL': 'chocolate', 'WT_HL':'orange', 'Df1_NonHL':'black', 'Df1_HL':'grey'}
space_colors = {'on': 'green', 'off':'blue', 'sustainednoise':'olive', 'sustainedsilence': 'grey'}
period_colors = {'Noise1': 'darkgreen', 'Gap': 'darkblue', 'Noise2': 'forestgreen', 'Post-N2': 'royalblue'}
space_colors_per_gap = {'on': sns.color_palette('BuGn', 11), 'off':sns.color_palette('GnBu', 11)}
method_colors = {'Pairwise':'#0047AB', 'CCA':'#DC143C', 'RV':'#228B22', 'Trace':'#800080'}
shade_color = 'gainsboro'

tick_size = 36
legend_size = 24
label_size = 40
sub_title_size = 44
title_size = 48

################################################## Basic Functions ##################################################

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
    if method == 'Pairwise': return 'Pairwise Cosine Similarity'
    if method == 'CCA': return 'Principal Angle'
    if method == 'RV': return 'RV Coefficient'
    if method == 'Trace': return 'Covariance Alignment'
    
def Comparison_Method_Abbr_Title(method):
    if method == 'Pairwise': return 'PairCos'
    if method == 'CCA': return 'PrAngle'
    if method == 'RV': return 'RV'
    if method == 'Trace': return 'CovAlign'
    
def Subspace_Name_Abbr_Title(subspace_name):
    if subspace_name == 'On': return 'On'
    if subspace_name == 'Off': return 'Off'
    if subspace_name == 'SustainedNoise': return 'S. Noise'
    if subspace_name == 'SustainedSilence': return 'S. Silence'

def sigmoid(x, L, x0, k, c):
    return L / (1 + np.exp(-k * (x - x0))) + c

def inverse_sigmoid(y, L, x0, k, c):
    return x0 - (1 / k) * np.log((L / (y-c)) - 1)

def format_number(val):
    if val < 1: return round(val,1)
    else: return int(val)

################################################## Non-Specific Plotting ##################################################

def Draw_Standard_Subspace_Location(Group, subspace_name, period_length = 50, offset_delay = 10):
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

    fig, axs = plt.subplots(1,1,figsize = (10, 3))
    axs.plot(np.arange(1000), Sound, color = 'black')
    ymin, ymax = 10, 60
    axs.fill_between(np.arange(len(gap_label)), ymin, ymax, where= gap_label == 1, color=shade_color)
    axs.fill_between(np.arange(len(gap_label))[start:end], ymin, ymax, where= gap_label[start:end] == on_off, color='lightcoral')
    axs.set_yticks([10,60])
    axs.set_xticks([0, 1000])
    axs.tick_params(axis='both', labelsize = tick_size)
    axs.set_xlabel('Time (ms)', fontsize = label_size)
    axs.set_ylabel('(dB)', fontsize = label_size)
    axs.annotate('Noise 1+2', xy=(110, 62), fontsize=40, color='black')
    fig.suptitle(f'Standard {Subspace_Name_Abbr_Title(subspace_name)}-Space', fontsize = title_size-4, fontweight = 'bold', y = 1.2)

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
        figtitle = Comparison_Method_Abbr_Title(method)
        
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
        axs.set_xticklabels(labels, rotation=0, fontsize=38)
        axs.set_yticks(label_positions)
        axs.set_yticklabels(labels, rotation=0, fontsize=38)
        fig.suptitle(figtitle, fontsize = title_size, fontweight='bold')
        plt.tight_layout()
        return fig
            
    Matrices = Generate_Data(n_observation, n_feature)
    
    fig_Pairwise = Draw_Compare_Subspace_Similarity(Matrices, method = 'Pairwise')
    fig_CCA = Draw_Compare_Subspace_Similarity(Matrices, method = 'CCA')
    fig_RV = Draw_Compare_Subspace_Similarity(Matrices, method = 'RV')
    fig_Trace = Draw_Compare_Subspace_Similarity(Matrices, method = 'Trace')
    
    return [fig_Pairwise, fig_CCA, fig_RV, fig_Trace]


################################################## Group-Specific Analysis ##################################################

def Standard_Subspace_Comparison(Group, subspace_name, period_length = 50, offset_delay = 10):
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
        
        mask = ~np.eye(Similarity_Index.shape[0], dtype=bool)
        off_diagonal_elements = Similarity_Index[mask]
        print(f'Average Similarity Index for {subspace_name} = {np.mean(off_diagonal_elements)} ({method})')

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=24)
        cbar.ax.set_yticks([0,0.5,1.0])

        axs.set_aspect('auto')
        labels = [f'{i}' for i in range(1, 11, 1)]
        label_positions = [i + 0.5 for i in range(10)]
        axs.set_xticks(label_positions)
        axs.set_xticklabels(labels, rotation=0, fontsize=24)
        axs.set_yticks(label_positions)
        axs.set_yticklabels(labels, rotation=0, fontsize=24)

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
        print(f'p values for shuffle neuron = {p_values[1]}')
        print(f'p values for add neuron = {p_values[2]}')
        print(f'p values for add noise = {p_values[3]}')
        print(f'p values for rotate space = {p_values[4]}')
        print('\n')
        
        x = np.arange(5)
        colors = ['black', 'red', 'blue', 'green', 'purple']
        axs.bar(x, Means, yerr=np.array(Stds)*3, color=colors, alpha=0.6, capsize=10, width=0.8, error_kw={'capthick': 3, 'elinewidth': 2.5})
        axs.axhline(y=0, linestyle = '--', color = 'black')

        max_y = max(Means) + max(Stds)*3
        for i in range(5):
            add_significance_bar(axs, 0, i, 1 + 0.1*(i-1), p_values[i])

        axs.set_yticks([0, 0.5, 1], labels = [0, 0.5, 1])
        axs.set_xticks([0, 1, 2, 3, 4], labels = ['Orig.', 'Shuffle_Neuron', 'Add_Neuron', 'Add_Noise', 'Rotate'], rotation = 45, ha = 'right')
        axs.tick_params(axis='both', labelsize=24)
        axs.set_ylim(0, 1.33)
        
        return axs
        
    def Draw_Compare_Subspace_Similarity(periods, subspace_name, method):
        fig, axs = plt.subplots(1, 2, figsize = (10, 5), gridspec_kw={'width_ratios': [3,2]})
        axs[0] = Draw_Compare_Subspace_Similarity_Result(axs[0], periods, subspace_name, method)
        axs[1] = Draw_Compare_Subspace_Similarity_Result_Test(axs[1], periods, subspace_name, method)
        fig.suptitle(f'{Subspace_Name_Abbr_Title(subspace_name)} Space: {Comparison_Method_Abbr_Title(method)}', fontsize = title_size-4, fontweight = 'bold',y = 1.05)
        return fig

    periods = Get_Data_Periods(subspace_name)
    fig_Pairwise = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'Pairwise')
    fig_CCA = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'CCA')
    fig_RV = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'RV')
    fig_Trace = Draw_Compare_Subspace_Similarity(periods, subspace_name, method = 'Trace') 
    return [fig_Pairwise, fig_CCA, fig_RV, fig_Trace]

def Subspace_Similarity_for_All_Gaps(Group, subspace_name, methods, standard_period_length=50, period_length=50, offset_delay = 10):
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
        example_gaps = [0,4,9]
        fig, axs = plt.subplots(3, 1, figsize=(20, 15))  
        for i in range(3):
            gap_idx = example_gaps[i]
            Similarity_Indices = Similarity_Index_for_All_Gap[gap_idx]
            gap_dur = round(Group.gaps[gap_idx]*1000)

            for method in methods:
                axs[i].plot(np.arange(100, 1000), Similarity_Indices[method], color = method_colors[method], linewidth = 7, alpha = 0.9)

            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[i].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = shade_color)
            
            axs[i].set_xticks([], labels = [])
            axs[i].set_yticks([0,1], labels = [0, 1])
            axs[i].tick_params(axis = 'both', labelsize = tick_size)
            axs[i].set_ylabel(f'Gap #{gap_idx+1}', fontsize = label_size)
            
        axs[2].set_xticks([100, 1000], labels = [100, 1000])
        axs[2].set_xlabel('Time (ms)', fontsize = label_size)
        

        fig.suptitle(title, fontsize = title_size, fontweight = 'bold', y=0.95)
        
        lines, labels = [], []
        for method in methods:
            line = Line2D([0], [0], color=method_colors[method], lw=6, alpha = 0.9)
            lines.append(line)
            labels.append(Comparison_Method_Abbr_Title(method))
        legend = axs[0].legend(lines, labels, loc='upper right', ncol=1, fontsize=32, frameon=True)
        legend.get_frame().set_facecolor('white')  # White background
        legend.get_frame().set_alpha(1.0)         # Fully opaque
        legend.get_frame().set_linewidth(1.5)     # Add border
        legend.get_frame().set_edgecolor('black') # Black border
        legend.set_zorder(10)
        return fig
    
    def Justify_the_Separation_Level_for_each_Space_each_Method():
        def Get_Histogram_by_Space_Across_Gap(data, bins):
            data_on, data_off, data_noise, data_silence = data[0], data[1], data[2], data[3]
            
            hist_on, _ = np.histogram(data_on, bins=bins, density=True)
            hist_off, _ = np.histogram(data_off, bins=bins, density=True)
            hist_noise, _ = np.histogram(data_noise, bins=bins, density=True)
            hist_silence, _ = np.histogram(data_silence, bins=bins, density=True)
            
            return hist_on, hist_off, hist_noise, hist_silence
        
        def Get_JS_Matrix_1D_Across_Gap(data, bins, base = 2):
            hist_on, hist_off, hist_noise, hist_silence = Get_Histogram_by_Space_Across_Gap(data, bins)
            epsilon = 1e-15
            
            hists = [hist_on, hist_off, hist_noise, hist_silence]
            JS_matrix = np.zeros((4, 4))
            
            for i in range(4):
                for j in range(4):
                    P = hists[i] + epsilon
                    Q = hists[j] + epsilon
                    P /= np.sum(P)
                    Q /= np.sum(Q)
                    M = 0.5 * (P + Q)
                    JS = 0.5 * entropy(P, M, base=base) + 0.5 * entropy(Q, M, base=base)
                    JS_matrix[i, j] = JS
            
            return JS_matrix
            
        def Draw_JS_Divergence_Matrix(axs, JS_matrix):
            formatted_annotations = [[format_number(val) for val in row] for row in JS_Matrix]
            sns.heatmap(JS_Matrix, ax = axs, cmap = 'YlGnBu', square = True, cbar = False, vmin = 0, vmax = 1, 
                        annot=formatted_annotations, annot_kws={'size': tick_size})
            axs.set_xticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size-8)
            axs.set_yticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size-8)
            return axs
        
        file_path = subspacepath + f'SubspaceEvolution/{subspace_name}/'
        with open(file_path + f'{label}.pkl', 'rb') as f:
            data = pickle.load(f)
        print('Data Existed!')
            
        fig, axs = plt.subplots(1, 4, figsize = (30, 8))
        axs = axs.flatten()
        for j in range(len(methods)):
            method = methods[j]
            
            On_sim, Off_sim, SustainedNoise_sim, SustainedSilence_sim = np.array([]), np.array([]), np.array([]), np.array([])
            for gap_idx in range(10):
                gap_dur = round(Group.gaps[gap_idx]*1000)
                Similarity = np.array(data[gap_idx][method])  
                
            
                On_sim = np.concatenate((On_sim, Similarity[0:100]))
                Off_sim = np.concatenate((Off_sim, Similarity[360 + gap_dur:460 + gap_dur]))
                SustainedNoise_sim = np.concatenate((SustainedNoise_sim, Similarity[150:250]))
                SustainedSilence_sim = np.concatenate((SustainedSilence_sim, Similarity[-100:]))
                
            min_bin, max_bin = 0, 1
            width = 0.01
            bins = np.arange(min_bin, max_bin + width, width)
            JS_Matrix = Get_JS_Matrix_1D_Across_Gap([On_sim, Off_sim, SustainedNoise_sim, SustainedSilence_sim], bins)
            
            axs[j] = Draw_JS_Divergence_Matrix(axs[j], JS_Matrix)
            axs[j].set_title(Comparison_Method_Abbr_Title(method), fontsize = sub_title_size)

        fig.suptitle(f'J-S Div. of ' + title, fontsize = title_size, fontweight = 'bold', y=1.05) 
        return fig
    
    label = Group.geno_type + '_' + Group.hearing_type
    standard_period = Get_Standard_Period()
    
    if subspace_name == 'On': title = '$R_{On}(t)$'
    if subspace_name == 'Off': title = '$R_{Off}(t)$'
    if subspace_name == 'SustainedNoise': title = '$R_{Noise}(t)$'
    if subspace_name == 'SustainedSilence': title = '$R_{Silence}(t)$'
          
    Similarity_Index_for_All_Gap = Get_Similarity_Index_for_All_Gap(standard_period)
    fig = Draw_Similarity_Index_for_All_Gap(Similarity_Index_for_All_Gap)
    fig_justification = Justify_the_Separation_Level_for_each_Space_each_Method()
    
    return fig, fig_justification 

def Compare_Method_Efficiency(Group, methods, space_names):
    def Get_MulD_Data_by_Period(R):
        period_length = 100
        data_on, data_off, data_noise, data_silence = [], [], [], []
        for gap_idx in range(10):
            gap_dur = round(Group.gaps[gap_idx] * 1000)
            
            data_on.append(R[gap_idx][:period_length])
            data_off.append(R[gap_idx][360 + gap_dur:360 + gap_dur + period_length])
            data_noise.append(R[gap_idx][150:150+period_length])
            data_silence.append(R[gap_idx][-period_length:])

        data_on = np.vstack(data_on)
        data_off = np.vstack(data_off)
        data_noise = np.vstack(data_noise)
        data_silence = np.vstack(data_silence)
        return data_on, data_off, data_noise, data_silence
    
    def kl_divergence_gaussian(mu0, cov0, mu1, cov1, base = 2):
        k = len(mu0)
        cov1_inv = np.linalg.inv(cov1)
        trace_term = np.trace(cov1_inv @ cov0)
        mean_diff = mu1 - mu0
        quad_term = mean_diff.T @ cov1_inv @ mean_diff
        log_det_term = np.log(np.linalg.det(cov1) / np.linalg.det(cov0) + 1e-15)
        
        kl = 0.5 * (trace_term + quad_term - k + log_det_term)
        
        '''if base != np.e:
            kl /= np.log(base)'''
            
        return kl

    def Get_JS_Matrix_MulD_Gaussian(data):
        # Get data by period
        data_on, data_off, data_noise, data_silence = Get_MulD_Data_by_Period(data)
        datas = [data_on, data_off, data_noise, data_silence]

        means = [np.mean(d, axis=0) for d in datas]
        covs = [np.cov(d.T) for d in datas]  # shape D x D

        JS_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                mu1, cov1 = means[i], covs[i]
                mu2, cov2 = means[j], covs[j]

                m = np.concatenate([datas[i], datas[j]], axis=0)
                mu_m, cov_m = np.mean(m, axis=0), np.cov(m.T)
                #mu_m = 0.5 * (mu1 + mu2)
                #cov_m = 0.5 * (cov1 + cov2)

                Dkl1 = kl_divergence_gaussian(mu1, cov1, mu_m, cov_m)
                Dkl2 = kl_divergence_gaussian(mu2, cov2, mu_m, cov_m)

                JS_matrix[i, j] = JS_matrix[j, i] = 0.5 * (Dkl1 + Dkl2)

        return JS_matrix
    
    def Draw_JS_Matrix_MulD_Covariance_Summary():
        def Get_Data_for_Each_Space(label, method, subspace_name):
            with open(subspacepath + f'SubspaceEvolution/{subspace_name}/{label}.pkl', 'rb') as f:
                data = pickle.load(f)
            data_per_space = []
            for gap_idx in range(10):
                data_per_space.append(np.array(data[gap_idx][method]))
            data_per_space = np.array(data_per_space)
            return data_per_space

        def Get_Data_for_Each_Method(label, method):
            #subspace_names = ['On', 'Off', 'SustainedNoise', 'SustainedSilence']
            subspace_names = ['On', 'Off']
            data = []
            for i in range(len(subspace_names)):
                data_per_space = Get_Data_for_Each_Space(label, method, subspace_names[i])
                data.append(np.array(data_per_space))
            data_per_method = np.stack(data, axis = 2)  ## (10, 900, 4)
            return data_per_method

        means = []
        fig, axs = plt.subplots(1, 4, figsize = (41.4, 10))
        plt.subplots_adjust(top=0.8)
        for i in range(len(methods)):
            method = methods[i]
            R = Get_Data_for_Each_Method(label, method)
            JS_matrix = Get_JS_Matrix_MulD_Gaussian(R)
            
            '''
            mask = ~np.eye(JS_matrix.shape[0], dtype=bool)
            off_diagonal_elements = JS_matrix[mask]
            means.append(np.mean(off_diagonal_elements))
            '''
            
            on_mean = (JS_matrix[0, 1] +  JS_matrix[0, 2] + JS_matrix[0, 3])/3
            off_mean = (JS_matrix[1, 0] +  JS_matrix[1, 2] + JS_matrix[1, 3])/3
            means.append((on_mean + off_mean)/2)
            
            formatted_annotations = [[format_number(val) for val in row] for row in JS_matrix]
            sns.heatmap(JS_matrix, ax = axs[i], cmap = 'YlGnBu', square = True, cbar = False, vmin = 0, vmax = 5,
                    annot=formatted_annotations, annot_kws={'size': tick_size})
            axs[i].set_xticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
            axs[i].set_yticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
            axs[i].set_title(Comparison_Method_Abbr_Title(method), fontsize = sub_title_size)
        fig.suptitle('J-S Divergence between Multi-Dim. Representations: Covariance-based', fontsize = title_size, fontweight = 'bold')
        return means, fig
    
    def Draw_JS_Matrix_MulD_Projection_Summary():
        def Get_Data_for_Each_Gap(Group, space_data_loading, PC):
            data_per_space = []
            for gap_idx in range(10):
                data_per_space.append((space_data_loading @ Group.pop_response_stand[:, gap_idx, 100:])[PC])
            data_per_space = np.array(data_per_space)
            return data_per_space

        def Get_Data_for_Each_Subspace(Group, space_name, offset_delay = 10, period_length = 100):
            if space_name == 'On':
                space_data = Group.pop_response_stand[:, 0, 100:100 + period_length]
                space_data_pca = analysis.PCA(space_data, multiple_gaps=False)
                space_data_loading = space_data_pca.loading
            elif space_name == 'Off':
                space_data = Group.pop_response_stand[:, 0, 450 + offset_delay:450 + offset_delay + period_length]
                space_data_pca = analysis.PCA(space_data, multiple_gaps=False)
                space_data_loading = space_data_pca.loading
            else:
                space_data_loading = Group.pca.loading
            
            PCs = [0,1]
            data_per_space = []
            for PC in PCs:
                data_per_space.append(Get_Data_for_Each_Gap(Group, space_data_loading, PC))
            data_per_space = np.stack(data_per_space, axis = 2)
            
            return data_per_space
        
        means = []
        fig, axs = plt.subplots(1, 3, figsize = (30.93, 10))
        plt.subplots_adjust(top=0.8)
        for i in range(len(space_names)):
            space_name = space_names[i]
            R = Get_Data_for_Each_Subspace(Group, space_name)
            JS_matrix = Get_JS_Matrix_MulD_Gaussian(R)

            '''
            mask = ~np.eye(JS_matrix.shape[0], dtype=bool)
            off_diagonal_elements = JS_matrix[mask]
            means.append(np.mean(off_diagonal_elements))
            '''
            
            on_mean = (JS_matrix[0, 1] +  JS_matrix[0, 2] + JS_matrix[0, 3])/3
            off_mean = (JS_matrix[1, 0] +  JS_matrix[1, 2] + JS_matrix[1, 3])/3
            means.append((on_mean + off_mean)/2)
            
            formatted_annotations = [[format_number(val) for val in row] for row in JS_matrix]
            sns.heatmap(JS_matrix, ax = axs[i], cmap = 'YlGnBu', square = True, cbar = False, vmin = 0, vmax = 5,
                    annot=formatted_annotations, annot_kws={'size': tick_size})
            axs[i].set_xticks([0.5, 1.5, 2.5, 3.5], ['Onset', 'Offset', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
            axs[i].set_yticks([0.5, 1.5, 2.5, 3.5], ['Onset', 'Offset', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
            axs[i].set_title(f'{space_name}-space', fontsize = sub_title_size)
        fig.suptitle('J-S Divergence between Multi-Dim. Representations: Projection-based', fontsize = title_size, fontweight = 'bold')
        return means, fig
        
    def Draw_Encoding_Method_Comparison():
        methods = ['PairCos', 'PrAngle', 'RV', 'CovAlign']
        space_names = ['Full', 'On', 'Off']
        encoding_methods = []
        fig, axs = plt.subplots(1, 1, figsize = (10, 11.86))
        plt.subplots_adjust(top=0.8)
        for i in range(len(space_names)):
            axs.bar(i, means_proj[i], color = 'grey', width = 0.8)
            encoding_methods.append(space_names[i] + ' Proj.')
        for i in range(len(methods)):
            axs.bar(i + len(space_names), means_cov[i], color = 'black', width = 0.8)
            encoding_methods.append(methods[i])
        axs.set_xticks(np.arange(len(space_names) + len(methods)), encoding_methods, fontsize = tick_size-4, rotation = 45, ha = 'right')
        axs.set_yticks([0,1,2],[0,1,2], fontsize = tick_size)
        axs.set_ylabel('Average Divergence Index', fontsize = label_size)
        axs.set_xlabel('Encoding Methods', fontsize = label_size)
        fig.suptitle('Average Divergence\nfor Multi-Dim. Representations', fontsize = title_size, fontweight = 'bold')
        return fig

    label = Group.geno_type + '_' + Group.hearing_type

    means_cov, fig_covariance_method_summary = Draw_JS_Matrix_MulD_Covariance_Summary()
    means_proj, fig_projection_summary = Draw_JS_Matrix_MulD_Projection_Summary()
    fig_method_comparison = Draw_Encoding_Method_Comparison()
    
    return fig_covariance_method_summary, fig_projection_summary, fig_method_comparison

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
    
def Period_Capacity_in_Subspace_Comparison(Group, method, max_on_capacity = 75, max_off_capacity = 100, max_timewindow = 100, offset_delay = 10):
    def Draw_Find_Period_Capacity_in_Subspace_Comparison(On_Similarities, Off_Similarities):
        gap_idx = 9
        gap_dur = round(Group.gaps[gap_idx]*1000)
        x, y = On_Similarities[gap_idx], Off_Similarities[gap_idx]
        
        fig, axs = plt.subplots(1, 2, figsize = (45, 6))

        for i in range(2):
            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[i].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = shade_color)
            
        axs[0].plot(np.arange(100, 1000), x, color = space_colors['on'], lw=7)
        axs[1].plot(np.arange(100, 1000), y, color = space_colors['off'], lw=7)

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
                        fontsize=36, fontweight='bold', 
                        verticalalignment='center')
                
            legend = axs[i].legend(loc = 'upper right', fontsize = 36)
            legend.get_frame().set_facecolor('white')  # White background
            legend.get_frame().set_alpha(1.0)         # Fully opaque
            legend.get_frame().set_linewidth(1.5)     # Add border
            legend.get_frame().set_edgecolor('black') # Black border
            
            axs[i].set_xticks([100, 1000], labels = [100, 1000])
            axs[i].set_yticks([0,1], labels = [0, 1])
            axs[i].tick_params(axis = 'both', labelsize = 40)
            axs[i].set_ylabel(f'Gap = {gap_dur} ms', fontsize = 44, fontweight = 'bold')
            axs[i].set_xlabel('Time (ms)', fontsize = 44, fontweight = 'bold')
        fig.suptitle(f'Compute Separation Level of Subspace Comparison', fontsize = 54, fontweight = 'bold', y=1.1)
        return fig
    
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
    label = Group.geno_type + '_' + Group.hearing_type
    file_path = check_path(subspacepath + f'BestSubspaceComparison/{method}/')
    with open(file_path + f'{label}.pkl', 'rb') as f:
        Similarities = pickle.load(f)
    On_Similarities = Similarities['On']
    Off_Similarities = Similarities['Off']
    fig_explain_find_best_subspace = Draw_Find_Period_Capacity_in_Subspace_Comparison(On_Similarities, Off_Similarities)
    
    return fig_best_capacity, fig_explain_find_best_subspace

def Subspace_Similarity_for_All_Gaps_Property(Group, method, optimised_param = True):
    def Find_Threshold(array):
        diff = np.diff(array)
        max_increase = np.max(diff)
        for i in range(len(diff)):
            if diff[i] > max_increase/5: 
                return i
    
    def Draw_Representation_for_All_Gap_All_Subspace(On_Similarities, Off_Similarities):
        fig, axs = plt.subplots(10, 1, figsize=(10, 21))  
        check_point = 100
        for gap_idx in range(10):
            On_Similarity_Index = On_Similarities[gap_idx]
            Off_Similarity_Index = Off_Similarities[gap_idx]
            
            gap_dur = round(Group.gaps[gap_idx]*1000)
            axs[gap_idx].plot(np.arange(check_point, 1000), On_Similarity_Index, color = space_colors['on'], linewidth = 7)
            axs[gap_idx].plot(np.arange(check_point, 1000), Off_Similarity_Index, color = space_colors['off'], linewidth = 7)
            
            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[gap_idx].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = shade_color)

            axs[gap_idx].set_xticks([], labels = [])
            axs[gap_idx].set_yticks([0,1], labels = [0, 1])
            axs[9].set_xticks([100, 1000], labels = [100, 1000])
            axs[gap_idx].tick_params(axis = 'both', labelsize = tick_size-4)
            axs[gap_idx].set_ylabel(f'Gap #{gap_idx}', fontsize = tick_size-4)
        axs[9].set_xlabel('Time (ms)', fontsize = label_size)
        fig.suptitle('$R(t)$', fontsize = title_size, fontweight = 'bold', y=0.9)
        
        on_line = Line2D([0], [0], color=space_colors['on'], lw=6)
        off_line = Line2D([0], [0], color=space_colors['off'], lw=6)
        lines, labels = [on_line, off_line], ['On', 'Off']
        fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.88), ncol=1, fontsize=32)
        return fig
        
    def Draw_Representation_in_Period_Across_Gaps(Similarity_Indices, plot_length, subspace_name):
        colors = space_colors_per_gap[subspace_name[0].lower() + subspace_name[1:]]
        
        if subspace_name == 'On':
            representation_title = '$R_{On}(t)$'
        if subspace_name == 'Off':
            representation_title = '$R_{Off}(t)$'

        fig1, axs = plt.subplots(1, 1, figsize=(10, 10))  
        peak_values = []
        for gap_idx in range(10):
            gap_dur = round(Group.gaps[gap_idx]*1000)
            Similarity_Index = Similarity_Indices[gap_idx]
            
            if subspace_name == 'On':
                start, end = 250 + gap_dur, 250 + gap_dur + plot_length
                title = 'End'
            # + gap_dur
            if subspace_name == 'Off':
                start, end = 250, 250 + plot_length
                title = 'Start'
                
            axs.plot(np.arange(plot_length), Similarity_Index[start:end], color = colors[gap_idx], linewidth = 6)
            peak_values.append(np.max(Similarity_Index[start:end]))
        
        axs.set_xlim((0, plot_length+10))
        axs.set_xticks([0, 50, 100], labels = [0, 50, 100])
        axs.set_yticks([0,1], labels = [0,1])
        axs.tick_params(axis = 'both', labelsize = tick_size)
        axs.set_ylabel(representation_title, fontsize = label_size)
        axs.set_xlabel(f'Time Aft. Gap {title} (ms)', fontsize = label_size)
        
        # Create the colorbar
        cmap = mpl.colors.ListedColormap(colors[:10])  # Use only the first 10 colors
        norm = mpl.colors.Normalize(vmin=0, vmax=10)    # We have 10 gaps (0-9)

        cbar_ax = fig1.add_axes([0.85, 0.15, 0.03, 0.7]) 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Add the "Gap #10" and "Gap #1" labels
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([0.5, 9.5])  # Position ticks appropriately for first and last gaps
        cbar.set_ticklabels(['Gap #1', 'Gap #10'])  # Set labels
        cbar.ax.tick_params(labelsize=32)  # Set font size

        fig1.suptitle(f'{subspace_name}set Representation', fontsize = title_size, fontweight = 'bold')
        
        
        fig2, axs = plt.subplots(1, 1, figsize=(10, 10))  
        for gap_idx in range(0, 10):
            axs.scatter(gap_idx, peak_values[gap_idx], color = colors[gap_idx], s = 400)
        axs.plot(np.arange(0, 10), peak_values[0:], color = space_colors[subspace_name[0].lower() + subspace_name[1:]], linewidth = 5)
        
        axs.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = tick_size)
        axs.set_xlabel(f'Gap Duration (ms)', fontsize = label_size)
        axs.set_ylabel('Max. ' + representation_title , fontsize = label_size)
        fig2.suptitle('Max. ' + representation_title, fontsize = title_size, fontweight = 'bold')
        
        return fig1, fig2

    label = Group.geno_type + '_' + Group.hearing_type
    
    if optimised_param:
        file_path = check_path(subspacepath + f'BestSubspaceComparison/{method}/')
        with open(file_path + f'{label}.pkl', 'rb') as f:
            Similarities = pickle.load(f)
        On_Similarities = Similarities['On']
        Off_Similarities = Similarities['Off']
    else:
        file_path = check_path(subspacepath + f'SubspaceEvolution/On/')
        with open(file_path + f'{label}.pkl', 'rb') as f:
            data = pickle.load(f)
        On_Similarities = np.array([data[i][method] for i in range(10)])  
        
        file_path = check_path(subspacepath + f'SubspaceEvolution/Off/')
        with open(file_path + f'{label}.pkl', 'rb') as f:
            data = pickle.load(f)
        Off_Similarities = np.array([data[i][method] for i in range(10)]) 
    
    fig_subspace_comparison = Draw_Representation_for_All_Gap_All_Subspace(On_Similarities, Off_Similarities)
    fig_on_similarity_evolution, fig_on_similarity_peak = Draw_Representation_in_Period_Across_Gaps(On_Similarities, plot_length = 100, subspace_name = 'On')
    fig_off_similarity_evolution, fig_off_similarity_peak = Draw_Representation_in_Period_Across_Gaps(Off_Similarities, plot_length = 100, subspace_name = 'Off')
    
    return fig_subspace_comparison, [fig_on_similarity_evolution, fig_on_similarity_peak], [fig_off_similarity_evolution, fig_off_similarity_peak]


################################################## Summary for All Groups ##################################################

def Subspace_Comparison_All_Group_Property(Groups, method, optimised_param = True):
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
                #axs.scatter(gap_idx, peak_values[gap_idx], color = space_colors_per_gap['on'][gap_idx], s = 400)
            axs.plot(np.arange(10), peak_values, color = group_colors[label], linewidth = 5, label = label)
        
        axs.legend(loc = 'upper left', fontsize = 28)
        axs.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_xlabel(f'Gap Duration (ms)', fontsize = 40)
        axs.set_ylabel('$R_{On}(t)$', fontsize = 40)
        fig.suptitle('Max. $R_{On}(t)$', fontsize = 54, fontweight = 'bold')
        return fig
    
    def Draw_Off_Similarity_Properties():
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig_, axs_ = plt.subplots(1, 1, figsize=(10, 10)) 
         
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
                #axs.scatter(gap_idx, peak_values[gap_idx], color = space_colors_per_gap['off'][gap_idx], s = 400)
                axs_.scatter(gap_idx, peak_values[gap_idx], color = space_colors_per_gap['off'][gap_idx], s = 400)
            axs.plot(np.arange(10), peak_values, color = group_colors[label], linewidth = 5, label = label)
            
            x_data = np.arange(9)
            y_data = peak_values[1:]
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[max(y_data), np.median(x_data), 1, 0])
            L_fit, x0_fit, k_fit, c_fit = popt
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = sigmoid(x_fit, *popt)
            r2 = r2_score(y_data, sigmoid(x_data, *popt))
            print(f'{label}: R-Squared for sigmoidal fit is {r2} when parameter optimisation is {optimised_param}')
            axs_.plot(x_fit+1, y_fit, color = group_colors[label], linewidth = 5, label = label)
                
        axs.legend(loc = 'upper left', fontsize = 28)
        axs.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs.set_yticks([0,1], labels = [0, 1])
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_xlabel(f'Gap Duration (ms)', fontsize = 40)
        axs.set_ylabel('$R_{Off}(t)$', fontsize = 40)
        fig.suptitle('Max. $R_{Off}(t)$', fontsize = 54, fontweight = 'bold')
        
        axs_.legend(loc = 'upper left', fontsize = 28)
        axs_.set_xticks([1, 3, 5, 7, 9], labels = ['2$^0$', '2$^2$', '2$^4$', '2$^6$', '2$^8$'])
        axs_.set_yticks([0,1], labels = [0, 1])
        axs_.tick_params(axis = 'both', labelsize = 36)
        axs_.set_xlabel(f'Gap Duration (ms)', fontsize = 40)
        axs_.set_ylabel('$R_{Off}(t)$', fontsize = 40)
        fig_.suptitle('Max. $R_{Off}(t)$\nSigmoid-Fit', fontsize = 54, fontweight = 'bold', y = 1.0)
        
        return fig, fig_
    
    fig_on =  Draw_On_Similarity_Properties()
    fig_off, fig_off_sigmoid = Draw_Off_Similarity_Properties()
    
    return fig_on, fig_off, fig_off_sigmoid

def Subspace_Comparison_All_Group_Property_Multiple_Timewindows(Groups, method):
    def Get_Data_Points_for_each_Timewindow():
        max_timewindow = 100
        timewindows = np.arange(5, max_timewindow+1, 5)
        lower_boundaries, upper_boundaries, thresholds = {label:[] for label in Groups.keys()}, {label:[] for label in Groups.keys()}, {label:[] for label in Groups.keys()}
        for i, (label, Group) in enumerate(Groups.items()):
            file_path = check_path(subspacepath + f'SubspaceEvolution/Off/{label}/')
            for t in range(len(timewindows)):
                with open(file_path + f'{timewindows[t]}.pkl', 'rb') as f:
                    data = pickle.load(f)
                Off_Similarities = np.array([data[i][method] for i in range(10)])
                
                # fit sigmoid curves for offset evolution, for each timewindow
                peak_values = []
                for gap_idx in range(10):
                    gap_dur = round(Group.gaps[gap_idx]*1000)
                    start, end = 250, 250 + 100 
                    Similarity_Index = Off_Similarities[gap_idx]
                    peak_values.append(np.max(Similarity_Index[start:end]))
                
                x_data = np.arange(9)
                y_data = peak_values[1:]
                try:
                    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[max(y_data), np.median(x_data), 1, 0])
                    L_fit, x0_fit, k_fit, c_fit = popt
                    x_fit = np.linspace(min(x_data), max(x_data), 100)
                    y_fit = sigmoid(x_fit, *popt)
                    r2 = r2_score(y_data, sigmoid(x_data, *popt))
                    if r2 >= 0.7:
                        up = L_fit + c_fit 
                        low = c_fit
                        lower_boundaries[label].append((up-low)*0.01 + low)
                        upper_boundaries[label].append((up-low)*0.99 + low)
                        thresholds[label].append(np.exp2(inverse_sigmoid((up-low)*0.5 + low, *popt)))
                    else:
                        print(f"Warning: Curve fitting poor for group {label} / t = {timewindows[t]}, skipped this one")
                except (RuntimeError, OptimizeWarning):
                    print(f"Warning: Curve fitting failed for group {label} / t = {timewindows[t]}, skipped this one")

            lower_boundaries[label] = np.array(lower_boundaries[label])
            upper_boundaries[label] = np.array(upper_boundaries[label])
            thresholds[label] = np.array(thresholds[label])

        return lower_boundaries, upper_boundaries, thresholds
    
    def Draw_Off_Similarity_Properties_with_Multiple_Timewindows():
        def Draw_Violin_Plot(axs, violin_data):
            labels = ['WT_NonHL', 'WT_HL','Df1_NonHL', 'Df1_HL']
            
            # Create violin plot
            parts = axs.violinplot(violin_data, positions=range(len(labels)),
                                    showmeans=True, showextrema=True, showmedians=False)
                    
            # Customize violin plot colors
            for j, pc in enumerate(parts['bodies']):
                label = labels[j]
                pc.set_facecolor(group_colors[label])
                pc.set_alpha(0.7)
            
            # Customize other violin plot elements
            #parts['cmedians'].set_color('black')  # Median marker color
            parts['cbars'].set_color('black')   # Center bar color
            parts['cmaxes'].set_color('black')  # Max marker color
            parts['cmins'].set_color('black')   # Min marker color
            
            '''# Add true parameter values as red horizontal lines
            for pos, true_val in enumerate(true_values):
                # Draw a horizontal red line at the true parameter value
                # Make it slightly wider than the violin plot for visibility
                line_width = 0.3  # Adjust this value to change the length of the red line
                axs.hlines(true_val, pos-line_width/2, pos+line_width/2, 
                                    colors='red', linewidth=2, label='True Value' if pos == 0 else "")'''
            
            axs.axhline(y=0, color = 'grey', linestyle = '--')
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            
            return axs

        lower_boundaries, upper_boundaries, thresholds = Get_Data_Points_for_each_Timewindow()
        
        titles= ['Lower Bound.', 'Upper Bound.']
        fig1, axs1 = plt.subplots(1, 2, figsize=(20, 10))  
        lower_boundaries_data = [value for _, value in lower_boundaries.items()]
        axs1[0] = Draw_Violin_Plot(axs1[0], lower_boundaries_data)
        upper_boundaries_data = [value for _, value in upper_boundaries.items()]
        axs1[1] = Draw_Violin_Plot(axs1[1], upper_boundaries_data)
        for i in range(2):
            axs1[i].set_yticks([0,1], labels = [0, 1])
            axs1[i].set_xticks([0,1,2,3], ['WT\nNonHL', 'WT\nHL', '$\mathit{Df1}$/+\nNonHL', '$\mathit{Df1}$/+\nHL'])
            axs1[i].tick_params(axis='both', labelsize=28)
            axs1[i].set_ylim(0, 1.05)
            axs1[i].set_ylabel('$R_{Off}(t)$', fontsize = 34)
            axs1[i].set_title(titles[i], fontsize = 44)
        
        fig2, axs2 = plt.subplots(1, 1, figsize=(10, 10)) 
        thresholds_data = [value for _, value in thresholds.items()]
        axs2 = Draw_Violin_Plot(axs2, thresholds_data)
        axs2.set_yticks([0, 16, 32, 48, 64], labels = [0, 16, 32, 48, 64])
        axs2.set_xticks([0,1,2,3], ['WT\nNonHL', 'WT\nHL', '$\mathit{Df1}$/+\nNonHL', '$\mathit{Df1}$/+\nHL'])
        axs2.tick_params(axis='both', labelsize=28)
        axs2.set_ylim(0, 65)
        axs2.set_ylabel('Gap Duration (ms)', fontsize = 34)
        axs2.set_title('Threshold', fontsize = 44)
        
        print('Compare lower boundaries: ')
        print(f'p value: WT NonHl vs HL = {stats.ks_2samp(lower_boundaries_data[0], lower_boundaries_data[1])[1]}')
        print(f'p value: Df1 NonHl vs HL = {stats.ks_2samp(lower_boundaries_data[2], lower_boundaries_data[3])[1]}')
        
        print('Compare upper boundaries: ')
        print(f'p value: WT NonHl vs HL = {stats.ks_2samp(upper_boundaries_data[0], upper_boundaries_data[1])[1]}')
        print(f'p value: Df1 NonHl vs HL = {stats.ks_2samp(upper_boundaries_data[2], upper_boundaries_data[3])[1]}')
        
        print('Compare thresholds: ')
        print(f'p value: WT NonHl vs HL = {stats.ks_2samp(thresholds_data[0], thresholds_data[1])[1]}')
        print(f'p value: Df1 NonHl vs HL = {stats.ks_2samp(thresholds_data[2], thresholds_data[3])[1]}')
        
        print('\n')
        print('Compare lower boundaries: ')
        print(f'p value: NonHl WT vs Df1 = {stats.ks_2samp(lower_boundaries_data[0], lower_boundaries_data[2])[1]}')
        print(f'p value: HL WT vs Df1 = {stats.ks_2samp(lower_boundaries_data[1], lower_boundaries_data[3])[1]}')
        
        print('Compare upper boundaries: ')
        print(f'p value: NonHl WT vs Df1 = {stats.ks_2samp(upper_boundaries_data[0], upper_boundaries_data[2])[1]}')
        print(f'p value: HL WT vs Df1  = {stats.ks_2samp(upper_boundaries_data[1], upper_boundaries_data[3])[1]}')
        
        print('Compare thresholds: ')
        print(f'p value:NonHl WT vs Df1 = {stats.ks_2samp(thresholds_data[0], thresholds_data[2])[1]}')
        print(f'p value: HL WT vs Df1 = {stats.ks_2samp(thresholds_data[1], thresholds_data[3])[1]}')
        
        return fig1, fig2 
    
    fig_off_boundary, fig_off_threshold = Draw_Off_Similarity_Properties_with_Multiple_Timewindows()
    return fig_off_boundary, fig_off_threshold

def Compare_Different_Parameters(Groups, method):
    def Get_Parameters():
        on_params, off_params, timewindow_params = [], [], []
        for i, (label, Group) in enumerate(Groups.items()):
            label = Group.geno_type + '_' + Group.hearing_type
            
            with np.load(subspacepath + f'PeriodCapacity/{method}/{label}.npz') as data:
                on_capacities = data['on_capacities']
                off_capacities = data['off_capacities']
                timewindows = data['timewindows']
                separate_level = data['separate_level']
            print('Data Existed!')
            
            on_capacity, off_capacity, timewindow = Determine_Best_Capacity(on_capacities, off_capacities, timewindows, separate_level)
            on_params.append(on_capacity)
            off_params.append(off_capacity)
            timewindow_params.append(timewindow)
            
        return on_params, off_params, timewindow_params
    
    def Draw_Parameters(on_capacities, off_capacities, timewindows):
        fig, axs = plt.subplots(1, 1, figsize = (12, 10))
        for i, (label, Group) in enumerate(Groups.items()):
            axs.scatter(i, on_capacities[i], color = group_colors[label], s = 400, facecolors='none')
            axs.scatter(i, off_capacities[i], color = group_colors[label], s = 400, facecolors='none')
            axs.scatter(i, timewindows[i], color = group_colors[label], s = 400, facecolors='none')
            
        axs.plot(np.arange(len(Groups)), on_capacities, color = space_colors['on'], linewidth = 7, label = 'On-Space')
        axs.plot(np.arange(len(Groups)), off_capacities, color = space_colors['off'], linewidth = 7, label = 'Off-Space')
        axs.plot(np.arange(len(Groups)), timewindows, color = 'purple', linewidth = 7, label = 'Current-Timewindow')
        
        axs.legend(loc = 'upper right', fontsize = 32)
        axs.set_yticks([0, 50, 100], labels = [0, 50, 100])
        axs.set_xticks([0,1, 2, 3], ['WT\nNonHL', 'WT\nHL', '$\mathit{Df1}$/+\nNonHL', '$\mathit{Df1}$/+\nHL'])
        axs.tick_params(axis='both', labelsize=36)
        axs.set_ylim(0, 102)
        axs.set_ylabel('Time (ms)', fontsize = 40)
        axs.set_title('Optimized Parameters\nfor Subspace Comparison', fontsize = 44, fontweight = 'bold')
        return fig
    
    on_capacities, off_capacities, timewindows = Get_Parameters()
    fig_param = Draw_Parameters(on_capacities, off_capacities, timewindows)
    
    return fig_param

def Compare_Off_Properties_with_Different_Parameters(Groups, method):
    def Draw_Off_Similarity_Properties():
        def Get_Properties(x_data, y_data):
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[max(y_data), np.median(x_data), 1, 0])
            L_fit, x0_fit, k_fit, c_fit = popt
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = sigmoid(x_fit, *popt)
            r2 = r2_score(y_data, sigmoid(x_data, *popt))
            
            lower_boundary = (L_fit-c_fit)*0.01 + c_fit
            upper_boundary = (L_fit-c_fit)*0.99 + c_fit
            threshold = np.exp2(inverse_sigmoid((L_fit-c_fit)*0.5 + c_fit, *popt))
            
            return lower_boundary, upper_boundary, threshold
            
        fig1, axs1 = plt.subplots(1, 2, figsize=(13, 10))  
        fig2, axs2 = plt.subplots(1, 1, figsize=(7, 10))   
        for i, (label, Group) in enumerate(Groups.items()):
            
            file_path = check_path(subspacepath + f'BestSubspaceComparison/{method}/')
            with open(file_path + f'{label}.pkl', 'rb') as f:
                Similarities = pickle.load(f)
            Off_Similarities_Optimised = Similarities['Off']
            
            file_path = check_path(subspacepath + f'SubspaceEvolution/Off/')
            with open(file_path + f'{label}.pkl', 'rb') as f:
                data = pickle.load(f)
            Off_Similarities_shared = np.array([data[i][method] for i in range(10)])
            
            peak_values_optimised, peak_values_shared = [], []
            for gap_idx in range(10):
                start, end = 250, 250 + 100 
                
                Similarity_Index = Off_Similarities_Optimised[gap_idx]
                peak_values_optimised.append(np.max(Similarity_Index[start:end]))
                
                Similarity_Index = Off_Similarities_shared[gap_idx]
                peak_values_shared.append(np.max(Similarity_Index[start:end]))

            lower_boundary1, upper_boundary1, threshold1 = Get_Properties(np.arange(9), peak_values_optimised[1:])
            lower_boundary2, upper_boundary2, threshold2 = Get_Properties(np.arange(9), peak_values_shared[1:])

            titles = ['Lower Bound.', 'Upper Bound.']
            axs1[0].scatter(0, lower_boundary2, color = group_colors[label], s = 400, facecolors='none')
            axs1[0].scatter(1, lower_boundary1, color = group_colors[label], s = 400, facecolors='none')
            axs1[0].plot([0,1], [lower_boundary2, lower_boundary1], color = group_colors[label], linewidth = 7)
            axs1[1].scatter(0, upper_boundary2, color = group_colors[label], s = 400, facecolors='none')
            axs1[1].scatter(1, upper_boundary1, color = group_colors[label], s = 400, facecolors='none')
            axs1[1].plot([0,1], [upper_boundary2, upper_boundary1], color = group_colors[label], linewidth = 7)
            
            axs2.scatter(0, threshold2, color = group_colors[label], s = 400, facecolors='none')
            axs2.scatter(1, threshold1, color = group_colors[label], s = 400, facecolors='none')
            axs2.plot([0,1], [threshold2, threshold1], color = group_colors[label], linewidth = 7)

        for i in range(2):
            axs1[i].set_yticks([0,1], labels = [0, 1])
            axs1[i].set_xticks([0,1], ['Shared', 'Optims.'], rotation = 45, ha = 'right')
            axs1[i].tick_params(axis='both', labelsize=28)
            axs1[i].set_xlim(-0.8, 1.8)
            axs1[i].set_ylim(0, 1)
            axs1[i].set_ylabel('Off-Similarity', fontsize = 28)
            axs1[i].set_title(titles[i], fontsize = 34)
        
        axs2.set_yticks([0, 8, 16, 24], labels = [0, 8, 16, 24])
        axs2.set_xticks([0,1], ['Shared', 'Optims.'], rotation = 45, ha = 'right')
        axs2.tick_params(axis='both', labelsize=28)
        axs2.set_xlim(-0.8, 1.8)
        axs2.set_ylim(0, 28)
        axs2.set_ylabel('Gap Duration (ms)', fontsize = 28)
        axs2.set_title('Threshold', fontsize = 34)
        
        return fig1, fig2
    
    fig1, fig2 = Draw_Off_Similarity_Properties()
    return fig1, fig2