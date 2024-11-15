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

from . import dynamicalsystem as ds

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
    def __init__(self, group, gap_idx, input = 'simple'):
        self.group = group  
        self.gap_idx = gap_idx 
        self.gap_dur = round(self.group.gaps[self.gap_idx]*1000)
        
        self.input = input
        self.model = self.Get_Model()
        self.N, self.PCs = None, None
        self.x, self.y, self.z = None, None, None
        self.Input_On, self.Input_Off = None, None
        
    def Get_Model(self):
        #if self.input == 'exponential': return DynamicalSystem_Simple(self.group, self.gap_idx)
        if self.input == 'complex': return ds.DynamicalSystem_Complex(self.group, self.gap_idx)
        
    def Cross_Validation(self):
        average_mse = np.zeros(10)
        self.models = []
        for gap_idx in np.arange(10):
            model = self.model
            model.gap_idx = gap_idx
            model.Set_Gap_Dependent_Params()
            model.opti_start, model.opti_end = 50, 350+round(self.group.gaps[gap_idx]*1000) + 200 
            if gap_idx != self.gap_idx: model.Optimize_Params()
            self.models.append(model)
            mse_per_model = []
            for i in range(10):
                if i == gap_idx: continue 
                model.gap_idx = i
                model.Set_Gap_Dependent_Params()
                model.Run()
                mse_per_model.append(np.mean((model.N - model.PCs)**2))
            average_mse[gap_idx] = np.mean(mse_per_model)
            print('Average MSE for model inferred from gap ' + str(gap_idx) + ": " + str(average_mse[gap_idx]) + '\n')
        best_idx = np.argsort(average_mse)[0]
        best_model = self.models[best_idx]
        self.model, self.gap_idx = best_model, best_idx
        
    def Draw(self):
        def Draw_Trace_2d():
            colors_unit = ['black', 'red', 'blue']
            colors_PC = ['dimgray', 'salmon', 'cornflowerblue']
            labels = ['Unit 1', 'Unit 2', 'Unit 3']
            fig1, axs = plt.subplots(4, 1, sharex=True, figsize=(17, 8))
            axs[0].plot(self.model.times, self.model.N[0], color=colors_unit[0])
            axs[0].plot(self.model.times, self.model.PCs[0], color=colors_PC[0], alpha = 0.5)
            axs[1].plot(self.model.times, self.model.N[1], color=colors_unit[1])
            axs[1].plot(self.model.times, self.model.PCs[1], color=colors_PC[1], alpha = 0.5)
            axs[2].plot(self.model.times, self.model.N[2], color=colors_unit[2])
            axs[2].plot(self.model.times, self.model.PCs[2], color=colors_PC[2], alpha = 0.5)
            axs[3].plot(self.model.times, self.model.OnS, color='darkgreen')
            axs[3].plot(self.model.times, self.model.OffS, color='limegreen')
            #axs[3].legend(loc = 'upper right', fontsize = 20)
            
            for j in range(4):
                axs[j].axhline(y=0, color = 'grey', linestyle = ':')
                axs[j].axvline(x=100, color = 'grey', linestyle = ':')
                axs[j].axvline(x=350, color = 'grey', linestyle = ':')
                axs[j].axvline(x=350+self.model.gap_dur, color = 'grey', linestyle = ':')
                axs[j].axvline(x=450+self.model.gap_dur, color = 'grey', linestyle = ':')
                axs[j].set_xticks([0, 200, 400, 600, 800], labels=[0, 200, 400, 600, 800], fontsize=16)
                if j <3:
                    axs[j].set_ylabel(labels[j], fontsize=24)
                    #axs[j].set_yticks([-1,0,1], labels=[-1,0,1], fontsize=16)
                    #axs[j].set_ylim((-1, 1))
            axs[3].set_xlabel('Time (ms)', fontsize=20)
            axs[0].set_title('Gap Duration: ' + str(self.model.gap_dur) + 'ms', fontsize = 24)
            plt.tight_layout()
            return fig1

        def Draw_Trace_3d():
            def style_3d_ax(ax):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.xaxis.pane.fill = True
                ax.yaxis.pane.fill = True
                ax.zaxis.pane.fill = True
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                ax.set_xlabel('Unit 1', fontsize = 14)
                ax.set_ylabel('Unit 2', fontsize = 14)
                ax.set_zlabel('Unit 3', fontsize = 14)
                
            
            fig2, axs = plt.subplots(1, 2, figsize=(24, 8), subplot_kw={'projection': '3d'})    
            x = gaussian_filter1d(self.model.N[0], sigma=sigma-1)
            y = gaussian_filter1d(self.model.N[1], sigma=sigma-1)
            z = gaussian_filter1d(self.model.N[2], sigma=sigma-1)
            
            PC1 = gaussian_filter1d(self.model.PCs[0], sigma=sigma)
            PC2 = gaussian_filter1d(self.model.PCs[1], sigma=sigma)
            PC3 = gaussian_filter1d(self.model.PCs[2], sigma=sigma)

            gap_dur = self.model.gap_dur + 350
            linecolors = ['grey', 'darkgreen', 'black', 'limegreen', 'black']
            linestyle = ['--', '--', '-', '--', ':']
            labels = ['pre-N1', 'Noise1', 'gap', 'Noise2', 'post-N2']
            starts = [0, 100, 350, gap_dur, gap_dur + 100]
            ends = [100, 350, gap_dur, gap_dur+100, 1000]

            for k in range(5):
                axs[0].plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                            ls=linestyle[k], c=linecolors[k], linewidth = 3, label = labels[k-1])
                axs[1].plot(PC1[starts[k]:ends[k]], PC2[starts[k]:ends[k]], PC3[starts[k]:ends[k]], 
                                            ls=linestyle[k], c=linecolors[k], linewidth = 3, label = labels[k-1])

            for i in range(2):
                axs[i].legend(loc = 'upper center', fontsize = 14)
                style_3d_ax(axs[i])
                axs[i].set_xlim((min(min(PC1), min(x)), max(max(PC1), max(x))))
                axs[i].set_ylim((min(min(PC2), min(y)), max(max(PC2), max(y))))
                axs[i].set_zlim((min(min(PC3), min(z)), max(max(PC3), max(z))))
            axs[0].set_title('Predicted Units', fontsize = 24)
            axs[1].set_title('Original PCs', fontsize = 24)
            plt.tight_layout()    
            return fig2
        
        def Draw_Parameters():
            def custom_fmt(val, i, j):
                if j in [3, 5, 7] and val == 0:
                    return ''
                return f'{val:.2f}'
            
            fig3, axs = plt.subplots(1, 1, figsize=(24, 8))            
                
            Empty_column = np.array([[0],[0],[0]]) 
            Params = np.concatenate((self.model.W, Empty_column, self.model.OnRe, Empty_column, self.model.OffRe, Empty_column, self.model.Nt), axis = 1)
            # Create the heatmap with custom formatting
            annot_matrix = [[custom_fmt(val, i, j) for j, val in enumerate(row)] 
                            for i, row in enumerate(Params)]
            # Create the heatmap with larger annotation font size
            sns.heatmap(Params, ax=axs, cmap='RdBu', vmin=-1, vmax=1, cbar=False, 
                        annot=annot_matrix, fmt='', annot_kws={'size': 40})  # Increased annotation font size
            axs.set_yticklabels(['X', 'Y', 'Z'], rotation=0, fontsize=30)  # Added fontsize parameter
            xticks = axs.get_xticks()
            xtick_labels = ['' for _ in range(len(xticks))]  # Initialize empty labels
            xtick_labels[1] = 'Connection'  # x=1
            xtick_labels[4] = 'OnRe'       # x=4
            xtick_labels[6] = 'OffRe'      # x=6
            xtick_labels[8] = 'TimeScale'  # x=8
            axs.set_xticklabels(xtick_labels, fontsize=30)  # Added fontsize parameter
            
            return fig3
        
        def Draw_Loss_with_Iter():
            fig4, axs = plt.subplots(1,2,figsize = (14,6))
            axs[0].plot(self.model.full_loss[:,0], self.model.full_loss[:,1])
            axs[1].plot(self.model.opti_loss[:,0], self.model.opti_loss[:,1])
            axs[0].set_xlabel('Iter', fontsize = 16)
            axs[1].set_xlabel('Iter', fontsize = 16)
            axs[0].set_ylabel('Loss', fontsize = 16)
            axs[1].set_ylabel('Min_Loss', fontsize = 16)
            plt.tight_layout()
            return fig4
        
        def Draw_Gap_Duration_Recognition():
            def calculate_distance(X, start, end):
                origin = np.mean(X[:,50:100], axis = 1)
                distance = []
                for t in range(start, end): 
                    distance.append(np.sqrt(np.sum((origin - X[:, t]) ** 2)))
                return np.array(distance)

            model = self.model
            fig5, axs = plt.subplots(2, 5, figsize = (30, 12))
            axs = axs.flatten()
            for i in range(10):
                gap_dur = round(self.group.gaps[i]*1000)
                gap_start, gap_end = 350, 350 + gap_dur 
                pre_gap_start, pre_gap_end = 300, 351
                post_gap_start, post_gap_end = 350 + gap_dur, 400 + gap_dur
                if i == 0: 
                    gap_start, gap_end = 350 + gap_dur + 100, 1000
                    pre_gap_start, pre_gap_end = 400, 451
                    post_gap_start, post_gap_end = 900, 1000
                
                model.gap_idx = i
                model.Set_Gap_Dependent_Params()
                model.Run()
                
                baseline = np.mean(calculate_distance(model.PCs, start = 50, end = 100))
                
                pre_distance = calculate_distance(model.N, start = pre_gap_start, end = pre_gap_end)
                axs[i].scatter(np.arange(-len(pre_distance),0,1), pre_distance, color = 'darkgreen')   
                distance = calculate_distance(model.N, start = gap_start, end = gap_end)
                axs[i].scatter(np.arange(len(distance)), distance, color = 'darkblue')
                post_distance = calculate_distance(model.N, start = post_gap_start, end = post_gap_end)
                axs[i].scatter(len(distance) + np.arange(len(post_distance)), post_distance, color = 'brown')
                
                pre_distance_true = calculate_distance(model.PCs, start = pre_gap_start, end = pre_gap_end)
                axs[i].scatter(np.arange(-len(pre_distance_true),0,1), pre_distance_true, color = 'green', alpha = 0.3)   
                distance_true = calculate_distance(model.PCs, start = gap_start, end = gap_end)
                axs[i].scatter(np.arange(len(distance_true)), distance_true, color = 'blue', alpha = 0.3)
                post_distance_true = calculate_distance(model.PCs, start = post_gap_start, end = post_gap_end)
                axs[i].scatter(len(distance_true) + np.arange(len(post_distance_true)), post_distance_true, color = 'orange', alpha = 0.3)

                time = np.argsort(distance)[::-1][0]
                axs[i].axvline(x=time, linestyle = '--', color = 'red', label = 'Peak Time: ' + str(time) + 'ms')
                
                axs[i].legend(loc = 'upper right', fontsize = 20)
                axs[i].axhline(y=baseline, linestyle = '--', color = 'grey')
                        
                if i > 0: 
                    axs[i].set_xlim((-50, 300))
                    axs[i].set_xticks(np.arange(-50, 305, 50), ['-50', '0', '50', '100', '150', '200', '250','300'], fontsize = 20)
                else: 
                    axs[i].set_xlim((-50, 550))
                    axs[i].set_xticks(np.arange(0, 505, 100), ['0', '100', '200', '300', '400', '500'], fontsize = 20)
                axs[i].set_ylim((0, 3))
                axs[i].set_yticks(np.arange(0, 3, 0.5), ['0', '0.5','1.0','1.5','2.0','2.5'], fontsize = 20)
                axs[i].set_xlabel('Time Since Gap Start (ms)', fontsize = 20)
                axs[i].set_ylabel('Distance to Noise On-Set', fontsize = 20)
                axs[i].set_title('Gap = ' + str(gap_dur) + 'ms', fontsize = 24)
            axs[0].set_xlabel('Time Since Post-Noise Off-Set (ms)', fontsize = 20)
            axs[0].set_title('Silence = 550 ms', fontsize = 24)
            plt.tight_layout()
            return fig5
        
        fig1 = Draw_Trace_2d() 
        fig2 = Draw_Trace_3d()
        fig3 = Draw_Parameters() 
        fig4 = Draw_Loss_with_Iter()
        fig5 = Draw_Gap_Duration_Recognition()
        
        return fig1, fig2, fig3, fig4, fig5
    
        