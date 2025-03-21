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

projectionpath = '/Volumes/Research/GapInNoise/Data/Projection/'

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

def Flip(PC, start = 100, end = 125):
    max_idx = np.argsort(abs(PC)[start:end])[::-1]
    if PC[start:end][max_idx[0]] < 0: return True 
    else: return False

def Low_Dim_Activity(Group):
    def Draw_Variance():
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))   
        axs.plot(np.arange(100), Group.pca.variance[:100], color = 'black')
        
        axs.set_xlabel('#Dimension', fontsize = 40)
        axs.set_ylabel('Variance Explained', fontsize = 40)
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_title('Variance Explained by Each PC', fontsize = 54, fontweight = 'bold')
        return fig 
    
    def Draw_Projection(PC):
        fig, axs = plt.subplots(ncols=len(PC), sharex=True, figsize=(40, 10))
        for j in range(len(PC)):
            scores = Group.pca.score_per_gap[PC[j]]
            for i in range(10):
                score_per_gap = scores[i]
                score_per_gap = (score_per_gap-np.mean(score_per_gap[:100]))/np.max(abs(score_per_gap))
                if Flip(score_per_gap): score_per_gap = score_per_gap * (-1)

                lower_bound = i*2-1
                upper_bound = i*2+1
                axs[j].fill_between([0, 0.25], lower_bound, upper_bound, 
                                    facecolor='gainsboro')
                axs[j].fill_between([0.25+Group.gaps[i], 0.35+Group.gaps[i]], lower_bound, upper_bound, 
                                    facecolor='gainsboro')
                axs[j].plot(np.arange(-0.1, 0.9, 0.001), score_per_gap + i*2, 
                            color=pal[i])

            axs[j].set_title('PC '+str(PC[j]+1), fontsize=24)
            axs[j].set_xticks([0, 0.2, 0.4, 0.6, 0.8], labels=[0, 200, 400, 600, 800], fontsize=32)
            axs[j].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['Gap#' + str(g+1) for g in range(10)], fontsize=32)
            axs[j].set_ylim((-1, 19))
            axs[j].set_xlabel('Time (ms)', fontsize=44)
        fig.suptitle('Projections to Each PC', fontweight = 'bold', fontsize = 54)
        return fig
    
    fig_variance = Draw_Variance()
    fig_projection = Draw_Projection(PC = [0,1,2,3])
    
    return fig_variance, fig_projection