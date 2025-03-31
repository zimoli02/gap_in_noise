import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge

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

################################################## Colors ##################################################
response_colors = {'on': 'darkgoldenrod', 'off': 'olivedrab', 'both': 'darkcyan', 'none':'slategray'}
shape_colors = {1: 'pink', 2: 'lightblue', 0:'grey'}
gap_colors = pal
group_colors =  {'WT_NonHL': 'chocolate', 'WT_HL':'orange', 'Df1_NonHL':'black', 'Df1_HL':'grey'}
space_colors = {'on': 'green', 'off':'blue'}
period_colors = {'Pre-N1': 'cornflowerblue', 'Noise1': 'darkgreen', 'Gap': 'darkblue', 'Noise2': 'forestgreen', 'Post-N2': 'royalblue'}
space_colors_per_gap = {'on': sns.color_palette('BuGn', 11), 'off':sns.color_palette('GnBu', 11)}
method_colors = {'Pairwise':'#0047AB', 'CCA':'#DC143C', 'RV':'#228B22', 'Trace':'#800080'}
shade_color = 'gainsboro'

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
                                    facecolor=shade_color)
                axs[j].fill_between([0.25+Group.gaps[i], 0.35+Group.gaps[i]], lower_bound, upper_bound, 
                                    facecolor=shade_color)
                axs[j].plot(np.arange(-0.1, 0.9, 0.001), score_per_gap + i*2, 
                            color=gap_colors[i], linewidth = 5)

            axs[j].set_title('PC '+str(PC[j]+1), fontsize=48)
            axs[j].set_xticks([0, 0.5, 1.0], labels=[0, 500, 1000], fontsize=32)
            axs[j].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['' for g in range(10)], fontsize=32)
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
    