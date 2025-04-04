import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import entropy

from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
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

################################################## Colors and Font ##################################################
response_colors = {'on': 'darkorange', 'off': 'olive', 'both': 'dodgerblue', 'none':'grey'}
response_psth_colors = {'on': 'bisque', 'off': 'darkkhaki', 'both': 'lightskyblue', 'none':'lightgrey'}
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

def Flip(PC, start = 100, end = 125):
    max_idx = np.argsort(abs(PC)[start:end])[::-1]
    if PC[start:end][max_idx[0]] < 0: return True 
    else: return False
    
################################################## Non-Specific Analysis ##################################################  

def Get_Data_by_Space(data, gap_idx, gaps):
    gap_dur = round(gaps[gap_idx]*1000)
    data_on = data[100:200]
    data_off = data[460 + gap_dur:560 + gap_dur]
    data_noise = data[250:350]
    data_silence = data[900:1000]
    
    return data_on, data_off, data_noise, data_silence

def Get_Histogram_by_Space(data, bins, gap_idx, gaps):
    data_on, data_off, data_noise, data_silence = Get_Data_by_Space(data, gap_idx, gaps)
    
    hist_on, _ = np.histogram(data_on, bins=bins, density=True)
    hist_off, _ = np.histogram(data_off, bins=bins, density=True)
    hist_noise, _ = np.histogram(data_noise, bins=bins, density=True)
    hist_silence, _ = np.histogram(data_silence, bins=bins, density=True)
    
    return hist_on, hist_off, hist_noise, hist_silence

def format_number(val):
        if val < 1: return round(val,1)
        else: return int(val)

def Get_KL_Matrix_1D(data, bins, gap_idx, gaps):
    hist_on, hist_off, hist_noise, hist_silence = Get_Histogram_by_Space(data, bins, gap_idx, gaps)
    epsilon = 1e-15
    hists = [hist_on, hist_off, hist_noise, hist_silence]
    KL_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            KL_matrix[i, j] = entropy(hists[i] + epsilon, hists[j] + epsilon)
    return KL_matrix

################################################## Group-Specific Analysis ##################################################

def Low_Dim_Activity(Group):
    def Draw_Variance():
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))   
        rank = min(100, len(Group.pca.variance))
        axs.plot(np.arange(rank), Group.pca.variance[:rank], color = 'black')
        
        axs.set_xlabel('#Dimension', fontsize = label_size)
        axs.set_ylabel('Variance Explained', fontsize = label_size)
        axs.tick_params(axis = 'both', labelsize = tick_size)
        axs.set_title('Variance Explained by Each PC', fontsize = title_size, fontweight = 'bold')
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
                                    facecolor=shade_color)
                axs[j].fill_between([0.25+Group.gaps[i], 0.35+Group.gaps[i]], lower_bound, upper_bound, 
                                    facecolor=shade_color)
                axs[j].plot(np.arange(-0.1, 0.9, 0.001), score_per_gap + i*2, 
                            color=gap_colors[i], linewidth = 5)

            axs[j].set_title('PC '+str(PC[j]+1), fontsize=sub_title_size)
            axs[j].set_xticks([0, 0.5, 1.0], labels=[0, 500, 1000], fontsize=tick_size)
            axs[j].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['' for g in range(10)], fontsize=tick_size)
            axs[j].set_ylim((-1, 19))
            axs[j].set_xlabel('Time (ms)', fontsize=label_size)
        axs[0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['Gap#' + str(g+1) for g in range(10)], fontsize=tick_size)
        fig.suptitle('Projections to Each PC', fontweight = 'bold', fontsize = title_size)
        return fig
    
    fig_variance = Draw_Variance()
    fig_projection = Draw_Projection(PC = [0,1,2,3])
    
    return fig_variance, fig_projection

def Low_Dim_Activity_Manifold(Group, short_gap = 5, long_gap = 9):
    def two_dim(gap_idx):
        gap_dur = round(Group.gaps[gap_idx]*1000+350)
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
            line_color = period_colors[labels[k]]
            axs.plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], 
                                            ls=linestyle[k], c=line_color, linewidth = 1)
            axs.plot([], [], ls=linestyle[k], c=line_color, linewidth = 5, label = labels[k])
            axs.scatter(x[starts[k]:ends[k]], y[starts[k]:ends[k]], 
                                            c=line_color, s = 30)

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
        linestyle = ['-', '--', '-', '--', ':']
        labels = ['Pre-N1', 'Noise1','Gap', 'Noise2', 'Post-N2']
        starts = [0, 100, 350, gap_dur, gap_dur + 100]
        ends = [100, 350, gap_dur, gap_dur+100, 1000]

        for k in range(1,4):
            line_color = period_colors[labels[k]]
            axs.plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                            ls=linestyle[k], c=line_color, linewidth = 1)
            axs.plot([], [], ls=linestyle[k], c=line_color, linewidth = 5, label = labels[k])
            axs.scatter(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                            c=line_color, s = 30)
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
    
def Low_Dim_Activity_by_Space(Group, short_gap = 3, long_gap = 9):
    def Draw_Histogram_by_Space(axs, data, bins, width, gap_idx, gaps):
        min_bin, max_bin = int(np.min(bins)), int(np.max(bins))
        
        hist_on, hist_off, hist_noise, hist_silence = Get_Histogram_by_Space(data, bins, gap_idx, gaps)
        
        axs.bar(np.arange(min_bin, max_bin, width), hist_on, color = space_colors['on'], width = width, alpha = 0.5, label = 'Onset')
        axs.bar(np.arange(min_bin, max_bin, width), hist_off, color = space_colors['off'], width = width, alpha = 0.5, label = 'Offset')
        axs.bar(np.arange(min_bin, max_bin, width), hist_noise, color = space_colors['sustainednoise'], width = width, alpha = 0.5, label = 'Sustained Noise')
        axs.bar(np.arange(min_bin, max_bin, width), hist_silence, color = space_colors['sustainedsilence'], width = width, alpha = 0.5, label = 'Sustained Silence')
        
        return axs
    
    def Draw_Histogram():
        fig, axs = plt.subplots(4, 2, figsize = (20, 20), sharex=False)

        width = 10
        for i in range(len(gap_indices)):
            gap_idx = gap_indices[i]
            for j in range(4):
                R = Group.pca.score_per_gap[j][gap_idx]
                min_bin, max_bin = round(np.min(R)-1), round(np.max(R) + 1)
                bins = np.arange(min_bin, max_bin + width, width)
                axs[j, i] = Draw_Histogram_by_Space(axs[j, i], R, bins, width, gap_idx, gaps)
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                axs[j, i].set_ylabel(f'PC{j+1}', fontsize = label_size)
        gap_dur1, gap_dur2 = round(gaps[gap_indices[0]]*1000), round(gaps[gap_indices[1]]*1000)        
        axs[0,0].set_title(f'Gap = {gap_dur1}ms', fontsize = sub_title_size)
        axs[0,1].set_title(f'Gap = {gap_dur2}ms', fontsize = sub_title_size)
        axs[0,0].legend(loc = 'upper left', fontsize = legend_size)
        axs[0,1].legend(loc = 'upper left', fontsize = legend_size)
        axs[3, 0].set_xlabel('Projection to PC', fontsize = label_size)
        axs[3, 1].set_xlabel('Projection to PC', fontsize = label_size)
        fig.suptitle('Response Distribution in Different Periods', fontsize = title_size, fontweight = 'bold')
        return fig
    
    gaps = Group.gaps
    gap_indices = [short_gap, long_gap]
    fig = Draw_Histogram()
    
    return fig
    
def Low_Dim_Activity_Divergence_by_Space(Group, short_gap = 3, long_gap = 9):
    def Draw_KL_Matrices():
        fig, axs = plt.subplots(1, 4, figsize = (40, 10))

        width = 10
        tick_size = 32
        for i in range(len(gap_indices)):
            gap_idx = gap_indices[i]
            
            # PC1
            R = Group.pca.score_per_gap[0][gap_idx]
            min_bin, max_bin = round(np.min(R)-1), round(np.max(R) + 1)
            bins = np.arange(min_bin, max_bin + width, width)
            KL_matrix = Get_KL_Matrix_1D(R, bins, gap_idx, gaps)
            formatted_annotations = [[format_number(val) for val in row] for row in KL_matrix]
            sns.heatmap(KL_matrix, ax = axs[i*2], cmap = 'YlGnBu', square = True, cbar = False, vmin = 0, vmax = 30, 
                        annot=formatted_annotations, annot_kws={'size': tick_size})
            
            # PC2
            R = Group.pca.score_per_gap[1][gap_idx]
            min_bin, max_bin = round(np.min(R)-1), round(np.max(R) + 1)
            bins = np.arange(min_bin, max_bin + width, width)
            KL_matrix = Get_KL_Matrix_1D(R, bins, gap_idx, gaps)
            formatted_annotations = [[format_number(val) for val in row] for row in KL_matrix]
            sns.heatmap(KL_matrix, ax = axs[i*2+1], cmap = 'YlGnBu', square = True, cbar = False, vmin = 0, vmax = 30,
                        annot=formatted_annotations, annot_kws={'size': tick_size})

        for i in range(4):
            axs[i].set_xticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
            axs[i].set_yticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
        gap_dur1, gap_dur2 = round(gaps[gap_indices[0]]*1000), round(gaps[gap_indices[1]]*1000)        
        axs[0].set_title(f'Gap = {gap_dur1}ms\nProjection to PC1', fontsize = sub_title_size)
        axs[1].set_title(f'Gap = {gap_dur1}ms\nProjection to PC2', fontsize = sub_title_size)
        axs[2].set_title(f'Gap = {gap_dur2}ms\nProjection to PC1', fontsize = sub_title_size)
        axs[3].set_title(f'Gap = {gap_dur2}ms\nProjection to PC2', fontsize = sub_title_size)
        fig.suptitle('K-L Divergence between Response Projection in Different Periods', fontsize = title_size, fontweight = 'bold', y=1.05)
        return fig
        
    gaps = Group.gaps
    gap_indices = [short_gap, long_gap]
    fig_KL = Draw_KL_Matrices()
    return fig_KL
    
def Low_Dim_Activity_in_Different_Space(Group, short_gap = 3, long_gap = 9, space_name = 'On', period_length = 100, offset_delay = 10):
    def Draw_Projection():
        fig, axs = plt.subplots(ncols=len(PC), sharex=True, figsize=(20, 20))
        for j in range(len(PC)):
            for i in range(10):
                score_per_gap = (space_data_loading @ Group.pop_response_stand[:, i, :])[PC[j]]
                score_per_gap = (score_per_gap-np.mean(score_per_gap[:100]))/np.max(abs(score_per_gap))
                if Flip(score_per_gap): score_per_gap = score_per_gap * (-1)

                lower_bound = i*2-1
                upper_bound = i*2+1
                axs[j].fill_between([0, 0.25], lower_bound, upper_bound, 
                                    facecolor=shade_color)
                axs[j].fill_between([0.25+Group.gaps[i], 0.35+Group.gaps[i]], lower_bound, upper_bound, 
                                    facecolor=shade_color)
                axs[j].plot(np.arange(-0.1, 0.9, 0.001), score_per_gap + i*2, 
                            color=gap_colors[i], linewidth = 5)

            axs[j].set_title('PC '+str(PC[j]+1), fontsize=sub_title_size)
            axs[j].set_xticks([0, 0.5, 1.0], labels=[0, 500, 1000], fontsize=tick_size)
            axs[j].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['' for g in range(10)], fontsize=tick_size)
            axs[j].set_ylim((-1, 19))
            axs[j].set_xlabel('Time (ms)', fontsize=label_size)
        axs[0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['Gap#' + str(g+1) for g in range(10)], fontsize=tick_size)
        fig.suptitle(f'Projections to {space_name}set-Space PC', fontweight = 'bold', fontsize = title_size)
        return fig
    
    def Draw_Divergence():
        fig, axs = plt.subplots(2, 2, figsize = (20, 20))
        axs = axs.flatten()
        for i in range(len(gap_indices)):
            gap_idx = gap_indices[i]
            
            # PC1
            R = (space_data_loading @ Group.pop_response_stand[:, gap_idx, :])[0]
            min_bin, max_bin = round(np.min(R)-1), round(np.max(R) + 1)
            bins = np.arange(min_bin, max_bin + width, width)
            KL_matrix = Get_KL_Matrix_1D(R, bins, gap_idx, gaps)
            formatted_annotations = [[format_number(val) for val in row] for row in KL_matrix]
            sns.heatmap(KL_matrix, ax = axs[i*2], cmap = 'YlGnBu', square = True, cbar = False, vmin = 0, vmax = 30, 
                        annot=formatted_annotations, annot_kws={'size': tick_size})
            
            # PC2
            R = (space_data_loading @ Group.pop_response_stand[:, gap_idx, :])[1]
            min_bin, max_bin = round(np.min(R)-1), round(np.max(R) + 1)
            bins = np.arange(min_bin, max_bin + width, width)
            KL_matrix = Get_KL_Matrix_1D(R, bins, gap_idx, gaps)
            formatted_annotations = [[format_number(val) for val in row] for row in KL_matrix]
            sns.heatmap(KL_matrix, ax = axs[i*2+1], cmap = 'YlGnBu', square = True, cbar = False, vmin = 0, vmax = 30,
                        annot=formatted_annotations, annot_kws={'size': tick_size})

        for i in range(4):
            axs[i].set_xticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
            axs[i].set_yticks([0.5, 1.5, 2.5, 3.5], ['On', 'Off', 'S.Noi.', 'S.Sil.'], fontsize = tick_size)
        gap_dur1, gap_dur2 = round(gaps[gap_indices[0]]*1000), round(gaps[gap_indices[1]]*1000)        
        axs[0].set_title(f'Gap = {gap_dur1}ms, Project to PC1', fontsize = sub_title_size)
        axs[1].set_title(f'Gap = {gap_dur1}ms, Project to PC2', fontsize = sub_title_size)
        axs[2].set_title(f'Gap = {gap_dur2}ms, Project to PC1', fontsize = sub_title_size)
        axs[3].set_title(f'Gap = {gap_dur2}ms, Project to PC2', fontsize = sub_title_size)
        fig.suptitle('K-L Divergence between Response Projection\nin Different Periods', fontsize = title_size, fontweight = 'bold', y=1)
        return fig

    if space_name == 'On':
        space_data = Group.pop_response_stand[:, 0, 100:100 + period_length]
    else:
        space_data = Group.pop_response_stand[:, 0, 450 + offset_delay:450 + offset_delay + period_length]
    space_data_pca = analysis.PCA(space_data, multiple_gaps=False)
    space_data_loading = space_data_pca.loading
    
    PC = [0,1,2,3]
    gaps = Group.gaps
    gap_indices = [short_gap, long_gap]
    width = 10
    
    fig_projection = Draw_Projection()
    fig_KL = Draw_Divergence()
    
    return fig_projection, fig_KL
    
def Binary_Classifier(Group, subspace):
    def Get_Model(X, Y, kernel = False, kernel_type = '', alpha = 0):
        if not kernel:
            model = LogisticRegression()
            model.fit(X, Y)
        else:
            model = KernelRidge(kernel=kernel_type, alpha=alpha)
            model.fit(X, Y)

        return model

    def Get_Prediction(gap_idx, model, dim, kernel = False):
        PCs = Group.pca.score[dim, 1000*gap_idx:1000*(gap_idx+1)]
        X = PCs.T  
        if not kernel: 
            s = model.predict_proba(X)
        else:
            s = model.predict(X)
        return s

    def Get_Cross_Validated_Model(subspace, dim, kernel = False, kernel_type = '', alpha = 0):
        LogLoss = np.zeros(10)
        models = []
        for gap_idx in range(10):
            X_test = (subspace @ Group.pop_response_stand[:, gap_idx, :])[dim].T
            y_test = 1-  np.array(Group.gaps_label[gap_idx])
            X_train = np.zeros((len(Group.pop_response_stand), 2))
            y_train = []
            for j in range(10):
                if j == gap_idx: continue 
                X_train = np.concatenate([X_train, subspace @ Group.pop_response_stand[:, j, :]], axis=1)
                y_train += list(Group.gaps_label[j])
            X_train = X_train[dim, 2:].T
            y_train = 1- np.array(y_train)
            model = Get_Model(X_train, y_train, kernel = kernel, kernel_type = kernel_type, alpha = alpha)
            models.append(model)
            if not kernel:
                y_pred = model.predict_proba(X_test)[:, 1] 
                LogLoss[gap_idx] = -1/1000*np.sum(y_test * np.log(y_pred) + (1-y_test)*np.log(1-y_pred))
            else:
                y_pred = model.predict(X_test)
                LogLoss[gap_idx] = np.mean((y_pred - y_test) ** 2)
        min_idx = np.argsort(LogLoss)[0]
        return LogLoss[min_idx], models[min_idx]

    def Find_Efficient_Dim(subspace, kernel = False, kernel_type = '', alpha = 0):
        log_losses, models = [], []
        for i in range(1,min(20, len(Group.pop_response_stand))):
            dim = np.arange(0, i)
            log_loss, model = Get_Cross_Validated_Model(subspace, dim, kernel = kernel, kernel_type = kernel_type, alpha = alpha)
            log_losses.append(log_loss)
            models.append(model)
        return log_losses, models

    def Get_Efficient_Dim():
        LogLosses, binary_models = Find_Efficient_Dim(subspace, kernel = False)
        min_loss_idx = np.argsort(LogLosses)[0]
        binary_model = binary_models[min_loss_idx]
        dim_num = min_loss_idx + 1
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))   
        axs.plot(np.arange(len(LogLosses)), LogLosses, color = 'black')
        axs.scatter(2, LogLosses[2], color = 'red', s = 400)
        axs.set_xlabel('#Dimension', fontsize = 40)
        axs.set_ylabel('log Loss', fontsize = 40)
        axs.tick_params(axis = 'both', labelsize = 36)
        axs.set_title('Prediction Loss for Model', fontsize = 54, fontweight = 'bold')
        
        return fig
    
    def Build_Model():
        dim_num = 3
        dim = np.arange(0, dim_num)
        LogLoss, binary_model = Get_Cross_Validated_Model(subspace, dim = dim, kernel = False)
        
        fig, axs = plt.subplots(10, 1, figsize=(20, 70))  
        for gap_idx in range(10):
            gap_dur = round(Group.gaps[gap_idx]*1000)
            
            s = Get_Prediction(gap_idx, binary_model, dim = dim)
            prob_on = s[:, 0]
            prob_off = s[:, 1]
            
            axs[gap_idx].plot(prob_off, color = gap_colors[gap_idx], linewidth = 7)
            
            ymin, ymax = 0, 1
            mask = Group.gaps_label[gap_idx] == 1
            axs[gap_idx].fill_between(np.arange(len(Group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = shade_color)

            axs[gap_idx].set_xticks([], labels = [])
            axs[gap_idx].set_yticks([0,1], labels = [0, 1])
            axs[9].set_xticks([100, 1000], labels = [100, 1000])
            axs[gap_idx].tick_params(axis = 'both', labelsize = 36)
            axs[gap_idx].set_ylabel(f'Gap = {gap_dur} ms', fontsize = 36, fontweight = 'bold')
        axs[9].set_xlabel('Time (ms)', fontsize = 40, fontweight = 'bold')
        fig.suptitle(f'Predict Sound Off using {dim_num} Dimensions', fontsize = 54, fontweight = 'bold', y=0.9)
        
        return fig
    
    fig_Efficient_Dim  = Get_Efficient_Dim()
    fig_Model_Prediction = Build_Model()
    
    return fig_Efficient_Dim, fig_Model_Prediction
    