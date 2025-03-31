import os
import numpy as np
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

grouppath = '/Volumes/Research/GapInNoise/Data/Groups/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'

################################################## Basic Functions ##################################################

def Center(data):
    data_centered = []
    for i in range(len(data)):
        data_centered.append(data[i] - np.mean(data[i]))
    return np.array(data_centered)

def Detect_Transient(trasient_psth, upper_thres, lower_thres):
    # For the common case activity increasing with stim
    flag = 0
    for i in range(trasient_psth.shape[0]-1):
        if trasient_psth[i]>=upper_thres:
            for j in range(i+1, trasient_psth.shape[0]-1):
                if trasient_psth[i] <= trasient_psth[j]:
                    return 1

    # For the rare case activity reducing with stim
    if lower_thres>0:
        for i in range(trasient_psth.shape[0]-1):
            if trasient_psth[i] < lower_thres:
                for j in range(i+1, trasient_psth.shape[0]-1):
                    if trasient_psth[i] >= trasient_psth[j]:
                        return 1
    return flag

def Determine_Responsiveness(Group, neuron_data):
    response = np.zeros((2,10))
    for gap_idx in range(10):
        gap_dur = round(Group.gaps[gap_idx]*1000)
        
        on_background = neuron_data[gap_idx, 50:100].reshape(10, -1).sum(axis=1)/5
        on_period = neuron_data[gap_idx, 50:150].reshape(20, -1).sum(axis=1)/5
        mean, std = np.mean(on_background), np.std(on_background)
        flag = Detect_Transient(on_period[10:], mean + 3*std, mean - 3*std)
        response[0, gap_idx]  = flag

        off_background = neuron_data[gap_idx, 400+gap_dur:450+gap_dur].reshape(10, -1).sum(axis=1)/5
        off_period = neuron_data[gap_idx, 400+gap_dur:560+gap_dur].reshape(32, -1).sum(axis=1)/5
        mean, std = np.mean(off_background), np.std(off_background)
        flag = Detect_Transient(off_period[12:], mean + 3*std, mean - 3*std)
        response[1, gap_idx] = flag
    
    return np.mean(response[0]), np.mean(response[1])

def Determine_Type(Group, neuron_data):
    on, off = Determine_Responsiveness(Group, neuron_data)

    if on >= 0.7 and off < 0.7: unit_type = 'on'
    elif on < 0.7 and off >= 0.7: unit_type = 'off'
    elif on >= 0.7 and off>= 0.7: unit_type = 'both'
    else: unit_type = 'none'
    
    return unit_type

def Determine_Type_All(Group):
    unit_type = []
    matrix = Group.pop_response_stand
    for unit_idx in range(len(matrix)):
        neuron_data = matrix[unit_idx]
        single_unit_type = Determine_Type(Group, neuron_data)
        unit_type.append(single_unit_type)
    return np.array(unit_type)

################################################## Colors ##################################################
response_colors = {'on': 'darkgoldenrod', 'off': 'olivedrab', 'both': 'darkcyan', 'none':'slategray'}
shape_colors = {1: 'pink', 2: 'lightblue', 0:'grey'}
gap_colors = pal
group_colors =  {'WT_NonHL': 'chocolate', 'WT_HL':'orange', 'Df1_NonHL':'black', 'Df1_HL':'grey'}
space_colors = {'on': 'green', 'off':'blue'}
period_colors = {'Noise1': 'darkgreen', 'Gap': 'darkblue', 'Noise2': 'forestgreen', 'Post-N2': 'royalblue'}
space_colors_per_gap = {'on': sns.color_palette('BuGn', 11), 'off':sns.color_palette('GnBu', 11)}
method_colors = {'Pairwise':'#0047AB', 'CCA':'#DC143C', 'RV':'#228B22', 'Trace':'#800080'}
shade_color = 'gainsboro'

################################################## Non-Specific Plotting ##################################################

def Draw_Example_Unit():
    def Draw_Unit(neuron_data, unit_type):
        fig, axs = plt.subplots(1,2,figsize=(10, 10), sharey=True)
        
        gap_idx = 9
        gap_dur = round(Group.gaps[gap_idx]*1000)

        on_period = neuron_data[gap_idx, 50:150].reshape(20, -1).sum(axis=1)/5
        off_period = neuron_data[gap_idx, 400+gap_dur:510+gap_dur].reshape(22, -1).sum(axis=1)/5

        axs[0].bar(np.arange(20), on_period, width=1.0, color = 'grey', alpha = 1)
        axs[1].bar(np.arange(22), off_period, width=1.0, color = 'grey', alpha = 1)
        
        on_background = neuron_data[gap_idx, 50:100].reshape(10, -1).sum(axis=1)/5
        mean, std = np.mean(on_background), np.std(on_background)
        axs[0].plot([0, 20], [mean, mean], color = 'red', linestyle ='-', linewidth = 4)
        axs[0].plot([0, 20], [mean+3*std, mean+3*std], color = 'red', linestyle =':', linewidth = 4)
        axs[0].plot([0, 20], [mean-3*std, mean-3*std], color = 'red', linestyle =':', linewidth = 4)
        
        off_background = neuron_data[gap_idx, 400+gap_dur:450+gap_dur].reshape(10, -1).sum(axis=1)/5
        mean, std = np.mean(off_background), np.std(off_background)
        axs[1].plot([0, 22], [mean, mean], color = 'blue', linestyle ='-', linewidth = 4)
        axs[1].plot([0, 22], [mean+3*std, mean+3*std], color = 'blue', linestyle =':', linewidth = 4)
        axs[1].plot([0, 22], [mean-3*std, mean-3*std], color = 'blue', linestyle =':', linewidth = 4)
        
        for i in range(2):
            axs[i].set_xticks([0, 10, 20], labels=[-50, 0, 50], fontsize = 28)
            axs[i].tick_params(axis ='both', labelsize = 28)

        axs[1].set_yticks([])
        axs[0].set_yticks([0, 100], labels = [0,100], fontsize = 28)
        axs[1].spines['left'].set_visible(False)
        axs[0].set_xlabel('Onset', fontsize = 32)
        axs[1].set_xlabel('Offset', fontsize = 32)
        text = unit_type[0].upper() + unit_type[1:]
        fig.suptitle(text + '-Responsive', fontsize = 40)
        return fig 

    with open(grouppath +  'WT_NonHL.pickle', 'rb') as file:
        Group = pickle.load(file)
        
    types = ['on', 'off', 'both', 'none']
    idx = [1, 0, 2, 46]
    figs = []
    unit_type = Determine_Type_All(Group)
    for i in range(4):
        type_index = unit_type == types[i]
        fig = Draw_Unit(Group.pop_response_stand[type_index][idx[i]], unit_type=types[i])
        figs.append(fig)
    
    return figs



################################################## Group-Specific Analysis ##################################################



################################################## Summary for All Groups ##################################################


def Draw_Unit_Response_Type_All_Group(Groups):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    types = ['on', 'both', 'off', 'none']
    for i, (label, Group) in enumerate(Groups.items()):
        unit_type = Determine_Type_All(Group)
        percs = np.array([len(unit_type[unit_type==types[j]])/len(unit_type) for j in range(len(types))])

        for j in range(len(types)):
            type = types[j]
            color = response_colors[type]
            if j == 0: bottom = 0
            elif j == 1: bottom = percs[0]
            else: bottom=percs[0:j].sum()
            axs.bar(i, percs[j], bottom=bottom, 
                    facecolor=color, edgecolor=color, alpha=1)

    axs.set_ylim([0,1])
    axs.set_ylabel('Percentage (%)',fontsize=32)
    axs.set_yticks([0,1],[0, 100], fontsize=28)
    axs.set_xticks([0,1,2,3],['WT\nNonHL', 'WT\nHL', '$\mathit{Df1}$/+\nNonHL', '$\mathit{Df1}$/+\nHL'], fontsize=28)
    fig.suptitle('Response Type Summary', fontsize = 36)
    
    return fig

def Draw_Unit_Spike_Type_All_Group(Groups):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    types = [1, 2, 0]
    for i, (label, Group) in enumerate(Groups.items()):
        spike_type_label = Group.unit_id[:,2]
        percs = np.array([len(spike_type_label[spike_type_label==types[j]])/len(spike_type_label) for j in range(len(types))])

        for j in range(len(types)):
            type = types[j]
            color = shape_colors[type]
            if j == 0: bottom = 0
            elif j == 1: bottom = percs[0]
            else: bottom=percs[0:j].sum()
            axs.bar(i, percs[j], bottom=bottom,
                    facecolor=color, edgecolor=color, alpha=1)

    axs.set_ylim([0,1])
    axs.set_ylabel('Percentage (%)',fontsize=32)
    axs.set_yticks([0,1],[0, 100], fontsize=28)
    axs.set_xticks([0,1,2,3],['WT\nNonHL', 'WT\nHL', '$\mathit{Df1}$/+\nNonHL', '$\mathit{Df1}$/+\nHL'], fontsize=28)
    fig.suptitle('Spike Type Summary', fontsize = 36)
    
    return fig

def Draw_Responsiveness_Comparison(percent = 0.5):
    def Get_Partial_Pop_Response(start, end):
        data_part = np.zeros((2, 10, 1000))
        for geno_type in ['WT', 'Df1']:
            for hearing_type in ['NonHL', 'HL']:
                with open(grouppath +  f'{geno_type}_{hearing_type}.pickle', 'rb') as file:
                    Group = pickle.load(file)
                for i in range(len(Group.recording_names)):
                    Exp_name = Group.recording_names[i]
                    with open(recordingpath + Exp_name + '.pickle', 'rb') as file:
                        recording = pickle.load(file)
                    
                    meta_psth = np.zeros((2, 10, 1000))
                    meta_psth = np.concatenate((meta_psth,
                                                recording.response['sig_psth'][:,:,start:end,:].mean(axis=2)),
                                                axis=0)
                    pop_response_per_recording = meta_psth[2:]
                    
                    data_part = np.concatenate((data_part,
                                                pop_response_per_recording),
                                                axis=0)
        return  np.array(data_part[2:])
    
    def Draw(response_type, ratio_first, ratio_last):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.scatter(ratio_first, ratio_last, alpha=0.1)
        axs.plot(np.arange(0, 1, 0.0001), np.arange(0, 1, 0.0001), linestyle=':', color='black', label='y = x', linewidth=2)

        slope, intercept, r_value, p_value, std_err = stats.linregress(ratio_first, ratio_last)
        x_fit = np.linspace(-0.1, 1.1, 1000)
        y_fit = slope * x_fit + intercept
        axs.plot(x_fit, y_fit, color='red', linewidth=2, 
                label=f'Fitted line: y = {slope:.3f}x + {intercept:.3f} (R² = {r_value**2:.3f})')

        axs.set_xlim((-0.1, 1.1))
        axs.set_ylim((-0.1, 1.1))
        axs.legend(loc = 'upper left', fontsize = 24)
        axs.set_xlabel(f'{response_type}-Response Ratio (First)', fontsize = 32)
        axs.set_ylabel(f'{response_type}-Response Ratio (Last)', fontsize = 32)
        axs.set_title(f'Comparison of {response_type}-Response', fontsize = 36)
        axs.tick_params(axis ='both', labelsize = 28)
        return fig 

    num_of_trials = int(45*percent)
    data_first = Get_Partial_Pop_Response(start = 0, end = num_of_trials)
    data_last = Get_Partial_Pop_Response(start = 45-num_of_trials, end = 45)
    
    on_response_first, on_response_last = [], []
    off_response_first, off_response_last = [], []

    with open(grouppath +  'WT_NonHL.pickle', 'rb') as file:
        Group = pickle.load(file)
    for n in range(len(data_first)):
        neuron_first = data_first[n]
        neuron_last = data_last[n]
        
        on, off = Determine_Responsiveness(Group, neuron_first)
        on_response_first.append(on)
        off_response_first.append(off)
        
        on, off = Determine_Responsiveness(Group, neuron_last)
        on_response_last.append(on)
        off_response_last.append(off)
        
    fig_on = Draw('On', on_response_first, on_response_last)
    fig_off = Draw('Off', off_response_first, off_response_last)
    
    return fig_on, fig_off