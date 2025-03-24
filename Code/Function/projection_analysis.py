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
        rank = min(100, len(Group.pca.variance))
        axs.plot(np.arange(rank), Group.pca.variance[:rank], color = 'black')
        
        axs.set_xlabel('#Dimension', fontsize = 40)
        axs.set_ylabel('Variance Explained', fontsize = 40)
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_title('Variance Explained by Each PC', fontsize = 54, fontweight = 'bold')
        return fig 
    
    def Draw_Projection(PC):
        fig, axs = plt.subplots(ncols=len(PC), sharex=True, figsize=(20, 20))
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
                            color=pal[i], linewidth = 5)

            axs[j].set_title('PC '+str(PC[j]+1), fontsize=48)
            axs[j].set_xticks([0, 0.5, 1.0], labels=[0, 500, 1000], fontsize=32)
            axs[j].set_ylim((-1, 19))
            axs[j].set_xlabel('Time (ms)', fontsize=44)
        axs[0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['Gap#' + str(g+1) for g in range(10)], fontsize=32)
        fig.suptitle('Projections to Each PC', fontweight = 'bold', fontsize = 54, y=0.97)
        return fig
    
    fig_variance = Draw_Variance()
    fig_projection = Draw_Projection(PC = [0,1,2,3])
    
    return fig_variance, fig_projection

def Low_Dim_Activity_Manifold(Group, short_gap = 5, long_gap = 9):
    def two_dim(gap_idx):
        gap_dur = round(Group.gaps[gap_idx]*1000+350)
        linecolors = ['grey', 'green', 'black', 'blue', 'grey']
        linestyle = ['-', '--', '-', '--', ':']
        labels = ['Pre-N1', 'Noise1','Gap', 'Noise2', 'Post-N2']
        starts = [0, 100, 350, gap_dur, gap_dur + 100]
        ends = [100, 350, gap_dur, gap_dur+100, 1000]
        
        PC = [0,1]
        projection = Group.pca.loading @ Center(Group.pop_response_stand[:, gap_idx, :])

        sigma = 2
        x = gaussian_filter1d(projection[PC[0]], sigma=sigma)
        y = gaussian_filter1d(projection[PC[1]], sigma=sigma)
        
        
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))    

        for k in range(1,5):
            axs.plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], 
                                            ls=linestyle[k], c=linecolors[k], linewidth = 1)
            axs.plot([], [], ls=linestyle[k], c=linecolors[k], linewidth = 5, label = labels[k])
            axs.scatter(x[starts[k]:ends[k]], y[starts[k]:ends[k]], 
                                            c=linecolors[k], s = 30)

        axs.scatter(x[350], y[350], label = 'Gap Start', color='red', s=200, alpha=1)
        axs.scatter(x[gap_dur], y[gap_dur], label = 'Gap End', color='magenta', s=200, alpha=1)

        axs.legend(fontsize = 20)
        
        minx, maxx = np.min(Group.pca.score[0]), np.max(Group.pca.score[0])
        miny, maxy = np.min(Group.pca.score[1]), np.max(Group.pca.score[1])
        axs.set_xlim((minx, maxx))
        axs.set_ylim((miny, maxy))
        axs.tick_params(axis = 'both', labelsize = 20)
        axs.set_xlabel('PC1', fontsize = 24)
        axs.set_ylabel('PC2', fontsize = 24)
        axs.set_title(f'Gap = {gap_dur-350} ms', fontsize = 40)
        return fig 
    
    def three_dim(gap_idx):
        def style_3d_ax(ax):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
            
        tick_size = 36
        label_size = 40
        title_size = 44
        labelpad = 30

        fig, axs = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': '3d'})    

        PC = [0,1,2]
        projection = Group.pca.loading @ Center(Group.pop_response_stand[:, gap_idx, :])

        sigma = 2
        x = gaussian_filter1d(projection[PC[0]], sigma=sigma)
        y = gaussian_filter1d(projection[PC[1]], sigma=sigma)
        z = gaussian_filter1d(projection[PC[2]], sigma=sigma)

        gap_dur = round(Group.gaps[gap_idx]*1000+350)
        linecolors = ['grey', 'green', 'black', 'blue', 'black']
        linestyle = ['-', '--', '-', '--', ':']
        labels = ['Pre-N1', 'Noise1','Gap', 'Noise2', 'Post-N2']
        starts = [0, 100, 350, gap_dur, gap_dur + 100]
        ends = [100, 350, gap_dur, gap_dur+100, 1000]

        for k in range(1,4):
            axs.plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                            ls=linestyle[k], c=linecolors[k], linewidth = 1)
            axs.plot([], [], ls=linestyle[k], c=linecolors[k], linewidth = 5, label = labels[k])
            axs.scatter(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                            c=linecolors[k], s = 30)
        axs.scatter(x[350], y[350], z[350], label = 'Gap Start', color='red', s=200, alpha=1)
        axs.scatter(x[gap_dur], y[gap_dur], z[gap_dur], label = 'Gap End', color='magenta', s=200, alpha=1)
        style_3d_ax(axs)
        axs.set_title(f'{gap_dur-350} ms', fontsize = title_size, pad = 12)
        axs.legend(fontsize = 24)
        axs.set_xlabel('PC1', fontsize=label_size, labelpad=labelpad)
        axs.set_ylabel('PC2', fontsize=label_size, labelpad=labelpad)
        axs.set_zlabel('PC3', fontsize=label_size, labelpad=labelpad)
        
        minx, maxx = np.min(Group.pca.score[0]), np.max(Group.pca.score[0])
        miny, maxy = np.min(Group.pca.score[1]), np.max(Group.pca.score[1])
        minz, maxz = np.min(Group.pca.score[2]), np.max(Group.pca.score[2])
        axs.set_xlim((minx, maxx))
        axs.set_ylim((miny, maxy))
        axs.set_zlim((minz, maxz))
        
        axs.view_init(elev=5, azim=85)  # Set a fixed viewing angle
        
        return fig

    fig_short_gap_2D = two_dim(gap_idx = short_gap)
    fig_long_gap_2D = two_dim(gap_idx = long_gap)

    fig_short_gap_3D = three_dim(gap_idx = short_gap)
    fig_long_gap_3D = three_dim(gap_idx = long_gap)
    
    return [fig_short_gap_2D, fig_long_gap_2D], [fig_short_gap_3D, fig_long_gap_3D]
    