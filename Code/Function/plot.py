import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from . import analysis 

fs = 10
custom_params = {
    "font.size": fs,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
}
sns.set_theme(style="ticks", rc=custom_params)

basepath = '/Volumes/Zimo/Auditory/Data/'
pal = sns.color_palette('viridis_r', 11)

sigma = 3  # smoothing amount

def Flip(PC, start = 100, end = 125):
    max_idx = np.argsort(abs(PC)[start:end])[::-1]
    if PC[start:end][max_idx[0]] < 0: return True 
    else: return False
    
def Get_Plane_Colors(x, y, z):
    x_border, y_border, z_border = min(x), max(y), min(z)
    colors = []
    for i in range(len(x)):
        minn = min(abs(x[i]-x_border), abs(y[i]-y_border), abs(z[i]-z_border))
        if abs(x[i]-x_border) == minn: colors.append('pink')
        if abs(y[i]-y_border) == minn: colors.append('lightblue')
        if abs(z[i]-z_border) == minn: colors.append('yellow')
    return colors

def style_3d_ax(ax, PC):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xlabel(f'PC {PC[0]+1}', fontsize = 14)
    ax.set_ylabel(f'PC {PC[1]+1}', fontsize = 14)
    ax.set_zlabel(f'PC {PC[2]+1}', fontsize = 14)
    
    
class NeuralData:
    def __init__(self, group):
        self.group = group
        self.gaps = group.gaps

    def Plot_Heatmap(self, gap_idx, sort = ['PC', 0]):
        if sort[0] == 'PC': sort_idx = np.argsort(self.group.pca.loading[sort[1]])[::-1]
        if sort[0] == 'offset': sort_idx = np.argsort(self.group.unit_offset)[::-1]
        if sort[0] == 'onset': sort_idx = np.argsort(self.group.unit_offset)[::-1]

        sound_cond = self.group.gaps_label[gap_idx]

        fig, axs = plt.subplots(2, 1, figsize=(30, 6), gridspec_kw={'height_ratios': [30, 1]})
        # At noise duration #i, all trials averaged
        sns.heatmap(self.group.pop_response_stand[:, gap_idx, :][sort_idx], ax=axs[0], 
                    vmin = -0.5, vmax = 0.5, cmap='RdBu', cbar=False)
        sns.heatmap([sound_cond], ax=axs[1], 
                    vmin=0, vmax=1, cmap='Blues', cbar=False)

        for i in range(2):
            axs[i].set_aspect('auto')
            axs[i].set_xticks([])
            axs[i].set_xticklabels([], rotation=0)
            axs[i].set_ylabel('')
            axs[i].set_yticks([])

        plt.tight_layout()
        return fig

class Latent:
    def __init__(self, group):
        self.group = group
        self.gaps = group.gaps
        self.pca = group.pca

        self.score, self.loading, self.variance = None, None, None
        self.Get_PCA()

        self.score_per_gap = group.pca.score_per_gap

        self.euclidean_distance = None
        self.step_distance = None
        
        self.angle_noise, self.angle_gap, self.angle_fix_gap = None, None, None
        
        self.first_step_pre_gap = []
        self.first_step_post_gap = []
        self.first_step_gap = []
    
    def Get_PCA(self):
        self.score = self.pca.score
        self.loading = self.pca.loading
        self.variance = self.pca.variance

    def Plot_Projection(self, PC):
        fig, axs = plt.subplots(ncols=len(PC), sharex=True, figsize=(len(PC)*8-3, 8))
        for j in range(len(PC)):
            scores = self.score_per_gap[PC[j]]
            for i in range(10):
                score_per_gap = scores[i]
                score_per_gap = (score_per_gap-np.mean(score_per_gap[:100]))/np.max(abs(score_per_gap))
                if Flip(score_per_gap): score_per_gap = score_per_gap * (-1)

                lower_bound = i*2-1
                upper_bound = i*2+1
                axs[j].fill_between([0, 0.25], lower_bound, upper_bound, 
                                    facecolor='tab:grey', alpha=0.2)
                axs[j].fill_between([0.25+self.gaps[i], 0.35+self.gaps[i]], lower_bound, upper_bound, 
                                    facecolor='tab:grey', alpha=0.2)
                axs[j].plot(np.arange(-0.1, 0.9, 0.001), score_per_gap + i*2, 
                            color=pal[i])

            axs[j].set_title('PC '+str(PC[j]+1), fontsize=24)
            axs[j].set_xticks([0, 0.2, 0.4, 0.6, 0.8], labels=[0, 200, 400, 600, 800], fontsize=16)
            axs[j].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], labels=['Gap#' + str(g+1) for g in range(10)], fontsize=16)
            axs[j].set_ylim((-1, 19))
            axs[j].set_xlabel('Time (ms)', fontsize=20)
            axs[0].set_ylabel('PC Response (a.u.)', fontsize=20)
        plt.tight_layout()
        return fig

    def Plot_Components_Correlation(self):
        fig, axs = plt.subplots(1, 2, figsize = (11, 5))
        corre_on = [abs(np.corrcoef(self.group.pca.loading[i], self.group.unit_onset)[0,1]) for i in range(10)]
        corre_off = [abs(np.corrcoef(self.group.pca.loading[i], self.group.unit_offset)[0,1]) for i in range(10)]
        axs[0].plot(np.arange(1,11,1), corre_on)
        axs[1].plot(np.arange(1,11,1), corre_off)
        for i in range(2):
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_yticks([0,0.5, 1])
            axs[i].set_xticks([2,4,6,8,10])
            axs[i].set_xlabel('PC#', fontsize = 20)
            axs[i].set_ylabel('Abs. Corre. Coef.', fontsize = 20)
        axs[0].set_title('On-Response', fontsize = 24)
        axs[1].set_title('Off-Response', fontsize = 24)
        plt.tight_layout()
        return fig

    def Plot_Trajectory_3d(self, PC = [0,1,2]):
        def Plot_Step_Distance(ax, x, y, z, colors, start, end, linecolor):
            x_, y_, z_ = x[start:end], y[start:end], z[start:end]
            dx, dy, dz = x_[1:]-x_[:-1], y_[1:]-y_[:-1], z_[1:]-z_[:-1]
            dis = np.sqrt(dx**2 + dy**2 + dz**2)
            ax.plot([round(j*5) for j in range(len(dis))], dis, color = linecolor)
            ax.scatter([round(j*5) for j in range(len(dis))], dis, color = colors[start+1:end])
            ax.set_xlabel('Time Since Start (ms)', fontsize = 20)
    
        fig2, axs_ = plt.subplots(1, 3, figsize=(18,6)) 
        fig1, axs = plt.subplots(2, 5, figsize=(30, 12), subplot_kw={'projection': '3d'})    
        axs = axs.flatten()
        for i in range(len(self.gaps)):
            x = gaussian_filter1d(self.score_per_gap[PC[0], i], sigma=sigma)
            y = gaussian_filter1d(self.score_per_gap[PC[1], i], sigma=sigma)
            z = gaussian_filter1d(self.score_per_gap[PC[2], i], sigma=sigma)
            
            if Flip(x): x *= -1
            if Flip(y): y *= -1
            if Flip(z): z *= -1
        
            plane_colors = Get_Plane_Colors(x, y, z)
            
            gap_dur = round(self.gaps[i]*1000+350)
            linecolors = ['grey', 'darkgreen', 'black', 'limegreen', 'black']
            linestyle = ['--', '--', '-', '--', ':']
            labels = ['pre-N1', 'Noise1', 'gap', 'Noise2', 'post-N2']
            starts = [0, 100, 350, gap_dur, gap_dur + 100]
            ends = [100, 350, gap_dur, gap_dur+100, 1000]
            
            for k in range(5):
                axs[i].plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                                ls=linestyle[k], c=linecolors[k], linewidth = 3, label = labels[k])
                axs[i].scatter(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                                c=plane_colors[starts[k]:ends[k]], s = 30, alpha = 0.5)
            axs[i].set_title(f'{round(self.gaps[i]*1000)} ms', fontsize = 16)
            axs[i].legend(loc = 'upper center', fontsize = 12)

            style_3d_ax(axs[i], PC)
            axs[i].set_xlim((min(x), max(x)))
            axs[i].set_ylim((min(y), max(y)))
            axs[i].set_zlim((min(z), max(z)))

            Plot_Step_Distance(axs_[0], x, y, z, plane_colors, 100, 350, linecolor = pal[i])
            Plot_Step_Distance(axs_[1], x, y, z, plane_colors, 350, gap_dur, linecolor = pal[i])
            Plot_Step_Distance(axs_[2], x, y, z, plane_colors, gap_dur, gap_dur+100, linecolor = pal[i])

            axs_[0].set_xlim((0, 255))
            axs_[1].set_xlim((0, 260))
            axs_[2].set_xlim((0, 105))
            
            axs_[0].set_ylabel('Step Distance', fontsize=20)
            axs_[0].set_title('Noise1', fontsize=24)
            axs_[1].set_title('Gap', fontsize=24)
            axs_[2].set_title('Noise2', fontsize=24)
            
        plt.tight_layout()     

        return fig1, fig2

    def Plot_Trajectory_3d_by_Event(self, PC):
        fig, axs = plt.subplots(8, 5, figsize=(30, 36), subplot_kw={'projection': '3d'}, layout='constrained')

        for gap_idx, gap_type in enumerate(self.gaps):
            # for every trial type, select the part of the component
            x = gaussian_filter1d(self.score_per_gap[PC[0], gap_idx], sigma=sigma)
            y = gaussian_filter1d(self.score_per_gap[PC[1], gap_idx], sigma=sigma)
            z = gaussian_filter1d(self.score_per_gap[PC[2], gap_idx], sigma=sigma)
            
            if Flip(x): x *= -1
            if Flip(y): y *= -1
            if Flip(z): z *= -1
            
            plane_colors = Get_Plane_Colors(x, y, z)
            gap_dur = round(self.gaps[gap_idx]*1000+350)
            linecolors = ['grey', 'darkgreen', 'black', 'limegreen', 'black']
            linemasks = ['white','white','white','white','white']
            linestyle = ['--', '--', '-', '--', ':']
            labels = ['pre-N1', 'Noise1', 'gap', 'Noise2', 'post-N2']
            starts = [0, 100, 350, gap_dur, gap_dur + 100]
            ends = [100, 350, gap_dur, gap_dur+100, 1000]
            
            rol_idx = int(gap_idx / 5) * 4
            col_idx = gap_idx - 5 * int(gap_idx / 5)
            for j in range(4):
                color = linemasks[:j+1] + [linecolors[j+1]] + linemasks[j+2:]
                for k in range(5):
                    axs[rol_idx+j][col_idx].plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                                    ls=linestyle[k], c=color[k], linewidth = 3, label = labels[k])
                axs[rol_idx+j][col_idx].legend(fontsize = 16)
                axs[rol_idx+j][col_idx].scatter(x[starts[j+1]:ends[j+1]], y[starts[j+1]:ends[j+1]], z[starts[j+1]:ends[j+1]], 
                                                c=plane_colors[starts[j+1]:ends[j+1]], s = 30, alpha = 0.5) # pre-gap
                style_3d_ax(axs[rol_idx+j][col_idx], PC)
            axs[rol_idx][col_idx].set_title(f'{round(gap_type*1000)} ms', fontsize = 20)
        return fig
    
    def Plot_Trajectory_3d_Event(self, PC):
        def Plot_Trajectory():
            fig1, axs = plt.subplots(2, 5, figsize=(30, 12), subplot_kw={'projection': '3d'})    
            for j in range(1,6):
                step = 50
                axs[0][j-1].plot(x[100:100+j*step], y[100:100+j*step], z[100:100+j*step], 
                                ls='-', linewidth = 3, c='darkgreen', label = 'Noise1+2')
                axs[0][j-1].scatter(x[100:100+j*step], y[100:100+j*step], z[100:100+j*step], 
                                    c = plane_colors[100:100+j*step], s = 30, alpha = 0.5)
                axs[0][j-1].set_title(f'{round(j*step)} ms', fontsize = 20)
                
                step = 100
                axs[1][j-1].plot(x[100:100+350], y[100:100+350], z[100:100+350], 
                                ls='-', linewidth = 3, c='darkgreen', alpha = 0.4)
                axs[1][j-1].plot(x[gap_dur+100:gap_dur+100+j*step], y[gap_dur+100:gap_dur+100+j*step], z[gap_dur+100:gap_dur+100+j*step], 
                                ls='-', linewidth = 3, c='black', label = 'Post-Noise2')
                axs[1][j-1].scatter(x[gap_dur+100:gap_dur+100+j*step], y[gap_dur+100:gap_dur+100+j*step], z[gap_dur+100:gap_dur+100+j*step],
                                    c = plane_colors[gap_dur+100:gap_dur+100+j*step], s = 30, alpha = 0.5)
                axs[1][j-1].set_title(f'{round(j*step)} ms', fontsize = 20)
                
                for i in range(2):
                    style_3d_ax(axs[i][j-1], PC)
                    axs[i][j-1].set_xlim((min(x), max(x)))
                    axs[i][j-1].set_ylim((min(y), max(y)))
                    axs[i][j-1].set_zlim((min(z), max(z)))
                    axs[i][j-1].legend(fontsize = 20)
            plt.tight_layout()
            return fig1 
        
        def Plot_Trajectory_Step_Distance():
            fig2, axs = plt.subplots(2, 1, figsize=(6,12)) 
            #noise
            x_, y_, z_ = x[100:450], y[100:450], z[100:450]
            dx, dy, dz = x_[1:]-x_[:-1], y_[1:]-y_[:-1], z_[1:]-z_[:-1]
            dis = np.sqrt(dx**2 + dy**2 + dz**2)
            axs[0].plot([round(j) for j in range(len(dis))], dis, color = 'darkgreen')
            axs[0].scatter([round(j) for j in range(len(dis))], dis, color = plane_colors[101:450])
            axs[0].set_xlim((0, 350))
            axs[0].set_xlabel('Time Since Start (ms)', fontsize = 20)
            
            #post-noise silence
            x_, y_, z_ = x[450:], y[450:], z[450:]
            dx, dy, dz = x_[1:]-x_[:-1], y_[1:]-y_[:-1], z_[1:]-z_[:-1]
            dis = np.sqrt(dx**2 + dy**2 + dz**2)
            axs[1].plot([round(j) for j in range(len(dis))], dis, color = 'black')
            axs[1].scatter([round(j) for j in range(len(dis))], dis, color = plane_colors[451:451 + len(dis)])
            axs[1].set_xlim((0, 550))
            axs[1].set_xlabel('Time Since Start (ms)', fontsize = 20)
            
            axs[0].set_ylabel('Step Distance', fontsize=20)
            axs[1].set_ylabel('Step Distance', fontsize=20)
            axs[0].set_title('Noise1+2', fontsize=24)
            axs[1].set_title('Post-Noise2 Silence', fontsize=24)
            plt.tight_layout()
            return fig2
        
        def Plot_Plot_Trajectory_Euclidean_Distance():
            fig3, axs = plt.subplots(2, 1, figsize=(6,12)) 
            #noise
            x_, y_, z_ = x[100:450], y[100:450], z[100:450]
            dx, dy, dz = x_-x_[0], y_-y_[0], z_-z_[0]
            dis = np.sqrt(dx**2 + dy**2 + dz**2)
            axs[0].plot([round(j) for j in range(len(dis))], dis, color = 'darkgreen')
            axs[0].scatter([round(j) for j in range(len(dis))], dis, color = plane_colors[100:450])
            axs[0].set_xlim((0, 350))
            axs[0].set_xlabel('Time Since Start (ms)', fontsize = 20)
            
            #post-noise silence
            x_, y_, z_ = x[450:], y[450:], z[450:]
            dx, dy, dz = x_-x_[0], y_-y_[0], z_-z_[0]
            dis = np.sqrt(dx**2 + dy**2 + dz**2)
            axs[1].plot([round(j) for j in range(len(dis))], dis, color = 'black')
            axs[1].scatter([round(j) for j in range(len(dis))], dis, color = plane_colors[450:450 + len(dis)])
            axs[1].set_xlim((0, 550))
            axs[1].set_xlabel('Time Since Start (ms)', fontsize = 20)
            
            axs[0].set_ylabel('Euclidean Distance', fontsize=20)
            axs[1].set_ylabel('Euclidean Distance', fontsize=20)
            axs[0].set_title('Noise1+2', fontsize=24)
            axs[1].set_title('Post-Noise2 Silence', fontsize=24)
            plt.tight_layout()
            return fig3
            
        gap_idx = 0
        gap_dur = round(self.gaps[gap_idx]*1000+350)
        
        x = gaussian_filter1d(self.score_per_gap[PC[0], gap_idx], sigma=sigma)
        y = gaussian_filter1d(self.score_per_gap[PC[1], gap_idx], sigma=sigma)
        z = gaussian_filter1d(self.score_per_gap[PC[2], gap_idx], sigma=sigma)
        
        if Flip(x): x *= -1
        if Flip(y): y *= -1
        if Flip(z): z *= -1
        
        plane_colors = Get_Plane_Colors(x, y, z)
        
        fig1 = Plot_Trajectory()
        fig2 = Plot_Trajectory_Step_Distance()
        fig3 = Plot_Plot_Trajectory_Euclidean_Distance()

        return fig1, fig2, fig3
    
    def Calculate_First_Step_per_Gap(self, PC):
        def Calculate_Travel_Distance(x, y, z, start, end):
            x_, y_, z_ = x[start:end], y[start:end], z[start:end]
            dx, dy, dz = x_[1:]-x_[:-1], y_[1:]-y_[:-1], z_[1:]-z_[:-1]
            dis = np.sqrt(dx**2 + dy**2 + dz**2)
            if len(dis) < 3: return np.nan
            return np.sum(dis[:20])
 
        for i in range(len(self.gaps)):
            x = gaussian_filter1d(self.score_per_gap[PC[0], i], sigma=sigma)
            y = gaussian_filter1d(self.score_per_gap[PC[1], i], sigma=sigma)
            z = gaussian_filter1d(self.score_per_gap[PC[2], i], sigma=sigma)

            gap_dur = round(self.gaps[i]*1000+350)
            
            first_step = Calculate_Travel_Distance(x, y, z, 100, 350)
            self.first_step_pre_gap.append(first_step)
            first_step = Calculate_Travel_Distance(x, y, z, 350, gap_dur)
            self.first_step_gap.append(first_step)
            first_step = Calculate_Travel_Distance(x, y, z, gap_dur, gap_dur+100)
            self.first_step_post_gap.append(first_step)

    def Calculate_Distance(self, PC=[0, 1, 2]):
        euclidian_distance = np.zeros((10, 1000))
        for i in range(10):
            basepoint = self.score_per_gap[PC][:, i, 0:100].mean(axis=(1))
            for j in range(1000):
                euclidian_distance[i, j] = np.linalg.norm(self.score_per_gap[PC][:, i, j] - basepoint)
        self.euclidean_distance = euclidian_distance

        step_distance = np.zeros((10, 1000))
        for i in range(10):
            for j in range(999):
                step_distance[i, j+1] = np.linalg.norm(self.score_per_gap[PC][:, i, j] - self.score_per_gap[PC][:, i, j+1])
        self.step_distance = step_distance

    def Plot_Distance(self, PC):
        def plot_distance(ax, distance):
            pal = sns.color_palette('viridis_r', 11)
            for i in range(10):
                lower_bound = (i-0.2)*0.5*np.max(abs(distance[-1]))
                upper_bound = (i+0.6)*0.5*np.max(abs(distance[-1]))
                ax.fill_between([0, 0.25], lower_bound,
                                upper_bound, facecolor='tab:grey', alpha=0.2)
                ax.fill_between([0.25+self.gaps[i], 0.35+self.gaps[i]],
                                lower_bound, upper_bound, facecolor='tab:grey', alpha=0.2)
                ax.plot(np.arange(-0.1, 0.9, 0.001), distance[i]+lower_bound, color=pal[i])

            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], labels=[
                          0, 200, 400, 600, 800], fontsize=16)
            ax.set_xlabel('Time (ms)', fontsize=20)

        self.Calculate_Distance(PC)

        fig, axs = plt.subplots(ncols=2, sharex=True,figsize=(10, 6), layout='constrained')
        plot_distance(axs[0], self.euclidean_distance)
        plot_distance(axs[1], self.step_distance)
        axs[0].set_ylabel('Distance (a.u.)', fontsize=20)
        axs[0].set_title('Euclidean Distance', fontsize=24)
        axs[1].set_title('Step Distance', fontsize=24)
        return fig

    def Calculate_Angle(self, PC):
        angle_noise, angle_gap, angle_fix_gap = [], [], []
        for gap_idx in range(3, 10):
            # apply some smoothing to the trajectories
            x = gaussian_filter1d(self.score_per_gap[PC[0], gap_idx], sigma=sigma)
            y = gaussian_filter1d(self.score_per_gap[PC[1], gap_idx], sigma=sigma)
            z = gaussian_filter1d(self.score_per_gap[PC[2], gap_idx], sigma=sigma)

            gap_dur = round(self.gaps[gap_idx]*1000+350)

            N1 = np.array([x[100:200], y[100:200], z[100:200]]).T 
            gap = np.array([x[350:min(450, gap_dur)], y[350:min(450, gap_dur)], z[350:min(450, gap_dur)]]).T 
            N2 = np.array([x[gap_dur:gap_dur+100], y[gap_dur:gap_dur+100], z[gap_dur:gap_dur+100]]).T
            post_N2 = np.array([x[gap_dur+100:], y[gap_dur+100:], z[gap_dur+100:]]).T 

            angle_noise.append(analysis.angle_between_planes(N1, N2))
            angle_gap.append(analysis.angle_between_planes(N1, gap))
            angle_fix_gap.append(analysis.angle_between_planes(N1, post_N2))

        self.angle_noise = angle_noise
        self.angle_gap = angle_gap
        self.angle_fix_gap = angle_fix_gap

    def Plot_Angle(self, PC):
        self.Calculate_Angle(PC)
        fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, layout='constrained')
        axs.plot(np.arange(7), self.angle_gap,
                 color='grey', ls='-', label='Gap')
        axs.plot(np.arange(7), self.angle_fix_gap,
                 color='black', ls=':', label='Post-N2 Silence')
        axs.plot(np.arange(7), self.angle_noise,
                 color='orange', ls='--', label='Noise2')

        axs.legend(loc='lower left', fontsize=14)
        axs.set_xlabel('Gap (ms)', fontsize=20)
        axs.set_xticks([0, 1, 2, 3, 4, 5, 6], labels=[
                       4, '', 16, '', 64, '', 256], fontsize=16)
        axs.set_ylabel('Angle Degree', fontsize=20)
        axs.set_ylim((0,90))
        return fig

    def Plot_Step_Degree(self, PC):
        gap_idx, gap_type = 9, 0.256
        angle_pre_gap, angle_gap, angle_post_gap, angle_fix_gap = [], [], [],[]
        # apply some smoothing to the trajectories
        x = gaussian_filter1d(self.score_per_gap[PC[0], gap_idx], sigma=sigma)
        y = gaussian_filter1d(self.score_per_gap[PC[1], gap_idx], sigma=sigma)
        z = gaussian_filter1d(self.score_per_gap[PC[2], gap_idx], sigma=sigma)

        gap_dur = round(gap_type*1000+350)

        # for noise:
        start_point = np.array([x[100], y[100], z[100]])
        end_point = np.array([x[350], y[350], z[350]])
        mid_point = (start_point+end_point)/2
        
        for i in range(len(x[100:350])):
            point = np.array([x[100+i], y[100+i], z[100+i]])
            angle_pre_gap.append(analysis.calculate_vector_angle(start_point-end_point, point-mid_point))
        
        # for gap:
        start_point = np.array([x[350], y[350], z[350]])
        end_point = np.array([x[gap_dur], y[gap_dur], z[gap_dur]])
        mid_point = (start_point+end_point)/2
        
        for i in range(len(x[350:gap_dur])):
            point = np.array([x[350+i], y[350+i], z[350+i]])
            angle_gap.append(analysis.calculate_vector_angle(start_point-end_point, point-mid_point))
            
        # for post-gap:
        start_point = np.array([x[gap_dur], y[gap_dur], z[gap_dur]])
        end_point = np.array([x[gap_dur+100], y[gap_dur+100], z[gap_dur+100]])
        mid_point = (start_point+end_point)/2
        
        for i in range(len(x[gap_dur:gap_dur+100])):
            point = np.array([x[gap_dur+i], y[gap_dur+i], z[gap_dur+i]])
            angle_post_gap.append(analysis.calculate_vector_angle(start_point-end_point, point-mid_point))
        
        # for post-noise silence
        start_point = np.array([x[gap_dur+100], y[gap_dur+100], z[gap_dur+100]])
        end_point = np.array([x[-1], y[-1], z[-1]])
        mid_point = (start_point+end_point)/2
        
        for i in range(len(x[gap_dur+100:])):
            point = np.array([x[gap_dur+100+i], y[gap_dur+100+i], z[gap_dur+100+i]])
            angle_fix_gap.append(analysis.calculate_vector_angle(start_point-end_point, point-mid_point))

        fig, axs = plt.subplots(1, 1, figsize = (5, 5))
        axs.plot(np.arange(len(angle_pre_gap)), angle_pre_gap, label = 'Noise1', color = 'green')
        axs.plot(np.arange(len(angle_gap)), angle_gap, label = 'Gap', color = 'black')
        axs.plot(np.arange(len(angle_post_gap)), angle_post_gap, label = 'Noise2', color = 'lightgreen')
        axs.plot(np.arange(len(angle_fix_gap)), angle_fix_gap, label = 'Post-Noise2', color = 'grey')
        axs.axhline(y=180, color = 'red', ls = ':')
        axs.legend(loc = 'lower right', fontsize = 14)
        axs.set_xticks([0, 50,100,150,200,250], [0, 50,100,150,200,250], fontsize = 14)
        axs.set_yticks([0,45,90,135,180], [0,45,90,135,180], fontsize = 14)
        #axs.set_ylabel('Degree', fontsize = 16)
        axs.set_ylabel('Distance to Start', fontsize = 16)
        axs.set_xlabel('Time (ms)', fontsize = 16)
        plt.title(self.group.label, fontsize = 20)
        plt.tight_layout()
        return fig

    def Plot_Principal_Angle(self, dim = 5):
        def Plot_Multi_Gaps():
            fig1, axs = plt.subplots(1, 3, figsize=(17, 5))
            count = 0
            for count, gap_type in enumerate([2,3,7]):
                gap_dur = round(self.gaps[gap_type]*1000+350)
                
                pre_noise = self.group.pop_response_stand[:, gap_type,100:350]
                gap =  self.group.pop_response_stand[:,gap_type, 350:gap_dur]
                post_noise = self.group.pop_response_stand[:, gap_type, gap_dur:gap_dur+100]
                silence = self.group.pop_response_stand[:, gap_type, gap_dur+100:]
                periods = [pre_noise, post_noise, gap, silence]
                
                sim = np.zeros((4,4))
                for i in range(4):
                    for j in range(i,4):
                        period1, period2 = periods[i], periods[j]
                        period1_pca = analysis.PCA(period1, multiple_gaps = False)
                        period2_pca = analysis.PCA(period2, multiple_gaps = False)
                        angles = analysis.calculate_principal_angles(period1_pca.loading[:dim].T, period2_pca.loading[:dim].T)
                        sim[i,j] = 1 - np.mean(angles) / (np.pi/2)
                        sim[j,i] = sim[i,j]
                sns.heatmap(sim, ax = axs[count], cmap = 'binary', vmax = 1, square=True,cbar = True)
                
                axs[count].set_aspect('auto')
                axs[count].set_xticklabels(['N1', 'N2', 'Gap', 'Post-N2'], rotation = 0, fontsize = 14)
                axs[count].set_yticklabels(['N1', 'N2', 'Gap', 'Post-N2'], fontsize = 14)
                axs[count].set_title('Gap = ' + str(round(self.gaps[gap_type]*1000)) +'ms', fontsize = 16)
            plt.tight_layout()
            return fig1
        
        def Plot_Single_Neuron_Effect():
            
            N1 = self.group.pop_response_stand[:, 0, 100:200]
            gap = self.group.pop_response_stand[:,0, 460:560]
            
            sim = np.zeros(len(gap))
            for i in range(len(gap)):
                N1_, gap_ = np.delete(N1, i, axis=0), np.delete(gap, i, axis=0)
                period1_pca = analysis.PCA(N1_, multiple_gaps = False)
                period2_pca = analysis.PCA(gap_, multiple_gaps = False)
                angles = analysis.calculate_principal_angles(period1_pca.loading[:dim].T, period2_pca.loading[:dim].T)
                sim[i] = 1 - np.mean(angles) / (np.pi/2)
                
            period1_pca = analysis.PCA(N1, multiple_gaps = False)
            period2_pca = analysis.PCA(gap, multiple_gaps = False)
            angles = analysis.calculate_principal_angles(period1_pca.loading[:dim].T, period2_pca.loading[:dim].T)
            sim_orig = 1 - np.mean(angles) / (np.pi/2)
            
            sim_min = np.min(sim)
            sim_max = np.max(sim)
            bins = np.linspace(sim_min, sim_max, 41)
            
            fig2, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
            axs[0].hist(sim[self.group.unit_id[:,2] == 1], bins = bins, color = 'blue', alpha = 0.9, density = False, label = 'Exci.')
            axs[1].hist(sim[self.group.unit_id[:,2] == 2], bins = bins, color = 'red', alpha = 0.9, density = False, label = 'Inhi.')
            for i in range(2):
                axs[i].axvline(x = sim_orig, color = 'black', linestyle = '--', linewidth = 3)
                axs[i].legend(fontsize = 20)
                axs[i].set_ylabel('Count', fontsize = 24)
                axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axs[i].tick_params(axis='both', labelsize=20)
            axs[1].set_xlabel('Subspace Similarity', fontsize = 24)
            plt.tight_layout()

            fig3, axs = plt.subplots(4, 1, figsize=(6, 20), sharex=True)
            axs[0].hist(sim[self.group.unit_type == 'on'], bins = bins, color = 'blue', alpha = 0.9, density = False, label = 'On')
            axs[1].hist(sim[self.group.unit_type == 'off'], bins = bins, color = 'red', alpha = 0.9, density = False, label = 'Off')
            axs[2].hist(sim[self.group.unit_type == 'both'], bins = bins, color = 'green', alpha = 0.9, density = False, label = 'Both')
            axs[3].hist(sim[self.group.unit_type == 'none'], bins = bins, color = 'brown', alpha = 0.9, density = False, label = 'None')
            for i in range(4):
                axs[i].axvline(x = sim_orig, color = 'black', linestyle = '--', linewidth = 3)
                axs[i].legend(fontsize = 20)
                axs[i].set_ylabel('Count', fontsize = 24)
                axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axs[i].tick_params(axis='both', labelsize=20)
            axs[3].set_xlabel('Subspace Similarity', fontsize = 24)
            plt.tight_layout()
            
            return fig2, fig3
            
        
        fig1 = Plot_Multi_Gaps()
        fig2, fig3 = Plot_Single_Neuron_Effect()
        
        return fig1, fig2, fig3

    def Plot_OnOff_Period(self):
        def Project_to_Vector(matrix, vector):
            matrix_projection = []
            for i in range(matrix.shape[0]):
                y = matrix[i]
                x = vector 
                proj = (y @ x.T) / (np.linalg.norm(x) ** 2) * x
                matrix_projection.append(proj)
            return np.array(matrix_projection).T

        def Align_PC_Sign(PC):
            for i in range(len(PC)):
                if Flip(PC[i][100:350], 0, 25): PC[i] *= -1     
            return PC
        
        def Draw_PC_Variance():
            fig1, axs = plt.subplots(1,1,figsize = (5,5))
            axs.plot(onset_pca.variance[0:10], label = 'on-resp', color = 'darkblue')
            axs.plot(offset_pca.variance[0:10], label = 'off-resp', color = 'lightblue')
            axs.plot(onset_exclude_noise_pca.variance[0:10], label = 'on-resp excluding noise', color = 'darkgreen')
            axs.plot(offset_exclude_noise_pca.variance[0:10], label = 'off-resp excluding noise', color = 'lightgreen')
            axs.legend(fontsize = 16)
            plt.tight_layout()
            return fig1 
        
        def Draw_PC():
            fig2, axs = plt.subplots(3, 2, figsize = (16, 18))
            for i in range(3):
                if Flip(onset_pca.score[i], 0, 25): onset_pca.score[i] *= -1
                if Flip(offset_pca.score[i], 25, 50): offset_pca.score[i] *= -1
                if Flip(onset_exclude_noise_pca.score[i], 0, 25): onset_exclude_noise_pca.score[i] *= -1
                if Flip(offset_exclude_noise_pca.score[i], 25, 50): offset_exclude_noise_pca.score[i] *= -1
                
                axs[i,0].plot(onset_pca.score[i], color = 'black', label = 'Original Subspace')
                axs[i,1].plot(offset_pca.score[i], color = 'black', label = 'Original Subspace')
                axs[i,0].plot(onset_exclude_noise_pca.score[i], color = 'red', label = 'Subspace Exclude Sust. Noise Resp.')
                axs[i,1].plot(offset_exclude_noise_pca.score[i], color = 'red', label = 'Subspace Exclude Sust. Noise Resp.')
                
                axs[i,0].set_ylabel('PC #'+str(i+1), fontsize = 20)
                for j in range(2):
                    axs[i,j].axhline(y = 0, linestyle = ':', color = 'grey')
                    axs[i,j].legend(fontsize = 16)
                axs[i,0].set_xticks([0,50, 100],['0','50','100'])
                axs[i,1].set_xticks([0,50, 100],['0','50','100'])
            axs[0,0].set_title('On-Response Projection', fontsize = 22)
            axs[0,1].set_title('Off-Response Projection', fontsize = 22)
            axs[2,0].set_xlabel('Time (ms)', fontsize = 20)
            axs[2,1].set_xlabel('Time (ms)', fontsize = 20)
            plt.tight_layout()
            return fig2
        
        def Draw_Trajectory_3d():
            fig3, axs = plt.subplots(1, 2, figsize = (16, 6), subplot_kw={'projection': '3d'})
            x_lim, y_lim, z_lim = [],[],[]
            PC = [0,1,2]

            # Onset
            x = gaussian_filter1d(onset_pca.score[PC[0]], sigma=sigma)
            y = gaussian_filter1d(onset_pca.score[PC[1]], sigma=sigma)
            z = gaussian_filter1d(onset_pca.score[PC[2]], sigma=sigma)
            axs[0].plot(x, y, z, label = 'Original Subspace', color = 'black')
            axs[0].scatter(x[0], y[0], z[0], color = 'darkblue', s = 30)
            x_lim.append([min(x), max(x)])
            y_lim.append([min(y), max(y)])
            z_lim.append([min(z), max(z)])

            x = gaussian_filter1d(onset_exclude_noise_pca.score[PC[0]], sigma=sigma)
            y = gaussian_filter1d(onset_exclude_noise_pca.score[PC[1]], sigma=sigma)
            z = gaussian_filter1d(onset_exclude_noise_pca.score[PC[2]], sigma=sigma)
            axs[0].plot(x, y, z, label = 'Subspace Exclude Noise Resp', color = 'red')
            axs[0].scatter(x[0], y[0], z[0], color = 'darkblue', s = 30, label = 'Start')
            axs[0].scatter(0,0,0, color = 'green', s = 30, label = 'Origin')
            x_lim.append([min(x), max(x)])
            y_lim.append([min(y), max(y)])
            z_lim.append([min(z), max(z)])


            # Offset
            x = gaussian_filter1d(offset_pca.score[PC[0]], sigma=sigma)
            y = gaussian_filter1d(offset_pca.score[PC[1]], sigma=sigma)
            z = gaussian_filter1d(offset_pca.score[PC[2]], sigma=sigma)
            axs[1].plot(x, y, z, label = 'Original Subspace', color = 'black')
            axs[1].scatter(x[0], y[0], z[0], color = 'darkblue', s = 30)
            x_lim.append([min(x), max(x)])
            y_lim.append([min(y), max(y)])
            z_lim.append([min(z), max(z)])

            x = gaussian_filter1d(offset_exclude_noise_pca.score[PC[0]], sigma=sigma)
            y = gaussian_filter1d(offset_exclude_noise_pca.score[PC[1]], sigma=sigma)
            z = gaussian_filter1d(offset_exclude_noise_pca.score[PC[2]], sigma=sigma)
            axs[1].plot(x, y, z, label = 'Subspace Exclude Noise Resp', color = 'red')
            axs[1].scatter(x[0], y[0], z[0], color = 'darkblue', s = 30, label = 'Start')
            axs[1].scatter(0,0,0, color = 'green', s = 30, label = 'Origin')
            x_lim.append([min(x), max(x)])
            y_lim.append([min(y), max(y)])
            z_lim.append([min(z), max(z)])

            axs[0].set_title('On-Response Projection', fontsize = 20)
            axs[1].set_title('Off-Response Projection', fontsize = 20)

            x_lim, y_lim, z_lim = np.array(x_lim), np.array(y_lim), np.array(z_lim)
            xlim = (min(x_lim[:,0]),max(x_lim[:,1]))
            ylim = (min(y_lim[:,0]),max(y_lim[:,1]))
            zlim = (min(z_lim[:,0]),max(z_lim[:,1]))
            for i in range(2):
                axs[i].legend(loc = 'upper left', fontsize = 16)
                axs[i].set_xlim(xlim)
                axs[i].set_ylim(ylim)
                axs[i].set_zlim(zlim)
                style_3d_ax(axs[i], PC)
            plt.tight_layout()
            return fig3
        
        def Draw_Data_in_Period_Heatmap():
            fig4, axs = plt.subplots(2, 3, figsize=(5, 24))
            axs = axs.flatten()
            
            onset_sortidx = np.argsort(onset_exclude_noise_pca.loading[0])[::-1]
            sns.heatmap(onset[onset_sortidx], ax = axs[0], cmap = 'RdBu', cbar = False)  
            sns.heatmap(onset_projection[onset_sortidx], ax = axs[1], cmap = 'RdBu', cbar = False) 
            sns.heatmap(onset_exclude_noise[onset_sortidx], ax = axs[2], cmap = 'RdBu', cbar = False) 
            
            offset_sortidx = np.argsort(offset_exclude_noise_pca.loading[0])[::-1]
            sns.heatmap(offset[offset_sortidx], ax = axs[3], cmap = 'RdBu', cbar = False)  
            sns.heatmap(offset_projection[offset_sortidx], ax = axs[4], cmap = 'RdBu', cbar = False)  
            sns.heatmap(offset_exclude_noise[offset_sortidx], ax = axs[5], cmap = 'RdBu', cbar = False) 
            for i in range(6):
                axs[i].set_aspect('auto')
                axs[i].set_xticks([])
                axs[i].set_xticklabels([], rotation = 0)
                axs[i].set_ylabel('')
                axs[i].set_yticks([])
            axs[0].set_ylabel('On-Resp', fontsize = 16)
            axs[3].set_ylabel('Off-Resp', fontsize = 16)
            axs[0].set_title('Original', fontsize = 16)
            axs[1].set_title('Projection', fontsize = 16)
            axs[2].set_title('Proj - Out', fontsize = 16)
            plt.tight_layout()
            return fig4 
        
        def Draw_Projection_to_Subsplace_Heatmap():
            data = self.group.pop_response_stand[:,gap_idx,:]
            fig5, axs = plt.subplots(6, 1, figsize=(24, 18), gridspec_kw={'height_ratios': [30, 30, 30, 30, 30, 1]})
            PCs = 20
            sns.heatmap(Align_PC_Sign(onset_exclude_noise_pca.loading @ data)[:PCs], ax = axs[0], cmap = 'RdBu', cbar = False)  
            axs[0].set_ylabel('On-Resp w/o Noise', fontsize = 20)
            
            sns.heatmap(Align_PC_Sign(offset_exclude_noise_pca.loading @ data)[:PCs], ax = axs[1], cmap = 'RdBu', cbar = False)  
            axs[1].set_ylabel('Off-Resp w/o Noise', fontsize = 20)

            sns.heatmap(Align_PC_Sign(gap_pca.loading @ data)[:PCs], ax = axs[2], cmap = 'RdBu', cbar = False)  
            axs[2].set_ylabel('Gap', fontsize = 20)

            sns.heatmap(Align_PC_Sign(gap_no_onset_pca.loading @ data)[:PCs], ax = axs[3], cmap = 'RdBu', cbar = False)  
            axs[3].set_ylabel('Gap w/o On-Resp', fontsize = 20)

            sns.heatmap(Align_PC_Sign(gap_no_offset_pca.loading @ data)[:PCs], ax = axs[4], cmap = 'RdBu', cbar = False)  
            axs[4].set_ylabel('Gap w/o Off-Resp', fontsize = 20)

            sns.heatmap([self.group.gaps_label[gap_idx]], ax=axs[5], cmap='Blues', vmin=0, vmax=1, cbar=False)
            for i in range(6):
                axs[i].set_aspect('auto')
                axs[i].set_xticks([])
                axs[i].set_xticklabels([], rotation = 0)
                axs[i].set_yticks([])
            plt.tight_layout()
            return fig5
        
        gap_idx = 9
        gap_dur = round(self.group.gaps[gap_idx]*1000+350)

        onset = self.group.pop_response_stand[:,gap_idx,100:200] # first 100 ms of noise1, with delay
        offset = self.group.pop_response_stand[:,gap_idx,100+gap_dur:100+gap_dur+100] # first 100 ms of post-N2 silence, with delay
        noise = self.group.pop_response_stand[:,gap_idx,250:350] # last 100 ms of noise1
        gap = self.group.pop_response_stand[:,gap_idx,350:gap_dur]
        
        gap_no_onset = gap -  np.mean(onset, axis = 1)[:, np.newaxis]
        gap_no_offset = gap -  np.mean(offset, axis = 1)[:, np.newaxis]
        
        onset_projection = Project_to_Vector(onset.T, np.mean(noise, axis = 1))
        onset_exclude_noise = onset - onset_projection

        offset_projection = Project_to_Vector(offset.T, np.mean(noise, axis = 1))
        offset_exclude_noise = offset - offset_projection
        
        onset_pca = analysis.PCA(onset, multiple_gaps = False)
        offset_pca = analysis.PCA(offset, multiple_gaps = False)
        onset_exclude_noise_pca = analysis.PCA(onset_exclude_noise, multiple_gaps = False)
        offset_exclude_noise_pca = analysis.PCA(offset_exclude_noise, multiple_gaps = False)
        gap_pca = analysis.PCA(gap, multiple_gaps = False)
        gap_no_onset_pca = analysis.PCA(gap_no_onset, multiple_gaps = False)
        gap_no_offset_pca = analysis.PCA(gap_no_offset, multiple_gaps = False)

        # Draw variance explained from the four subspaces: [on, off] x [original, exclude sustanined response]
        fig1 = Draw_PC_Variance()
        # Draw PC1-3 from the four subspaces: [on, off] x [original, exclude sustanined response]
        fig2 = Draw_PC()
        # Draw PC1-3 (3d) from the four subspaces: [on, off] x [original, exclude sustanined response]
        fig3 = Draw_Trajectory_3d()
        # Draw heatmap of data, projection, data exclude projection 
        fig4 = Draw_Data_in_Period_Heatmap()
        # Draw PC1-N heatmap of data projected to the period-specific subspace 
        fig5 = Draw_Projection_to_Subsplace_Heatmap()

        return fig1, fig2, fig3, fig4, fig5
        
    def Plot_Gap_Dependent_On_Response(self):
        def Align_PC_Sign(PC):
            for i in range(len(PC)):
                if Flip(PC[i][20:70], 0, 25): PC[i] *= -1     
            return PC
        
        def Draw_Principal_Angles():
            fig1, axs = plt.subplots(2, 3, figsize = (24, 15))
            axs = axs.flatten()
            gap_idx = [0,2,4,6,8,9]
            for count, gap_idx in enumerate([0,2,4,6,8,9]):
                sim = np.zeros((5,5))
                for i in range(5):
                    for j in range(i,5):
                        period1_pca, period2_pca = self.group.periods_pca[gap_idx][i], self.group.periods_pca[gap_idx][j]
                        angles = analysis.calculate_principal_angles(period1_pca.loading[:5].T, period2_pca.loading[:5].T)
                        sim[i,j] = 1 - np.mean(angles) / (np.pi/2)
                        sim[j,i] = sim[i,j]
                sns.heatmap(sim, ax = axs[count], cmap = 'binary', vmin=0, vmax=1, square=True, cbar = True)
                axs[count].set_aspect('auto')
                axs[count].set_xticklabels(['Noise1', 'Noise2', 'N2-On', 'N2-Both', 'Silence'], rotation = 0, fontsize = 16)
                axs[count].set_yticklabels(['Noise1', 'Noise2', 'N2-On', 'N2-Both', 'Silence'], fontsize = 16)
                axs[count].set_title('Gap = ' + str(round(self.group.gaps[gap_idx]*1000)) +'ms', fontsize = 24)
            plt.tight_layout()
            return fig1
        
        def Draw_Projection_to_Subspace():
            fig2 = plt.figure(figsize=(24, 18))
            gs = GridSpec(3, 3, figure=fig2, width_ratios=[6, 9, 9], height_ratios=[1, 1, 1], wspace=0.1, hspace=0.1)
            left_axes, mid_upper_axes, mid_lower_axes, right_upper_axes, right_lower_axes = [], [], [], [], []

            for row in range(3):
                left_ax = fig2.add_subplot(gs[row, 0])
                left_axes.append(left_ax)    

                mid_upper_ax = fig2.add_subplot(gs[row, 1])
                mid_upper_axes.append(mid_upper_ax)
                mid_lower_ax = fig2.add_axes(mid_upper_ax.get_position())
                mid_lower_axes.append(mid_lower_ax)

                pos = mid_upper_ax.get_position()
                mid_upper_ax.set_position([pos.x0, pos.y0 + pos.height*2/35, pos.width, pos.height * 33/35])
                mid_lower_ax.set_position([pos.x0, pos.y0, pos.width, pos.height*1/35])

                right_upper_ax = fig2.add_subplot(gs[row, 2])
                right_upper_axes.append(right_upper_ax)
                right_lower_ax = fig2.add_axes(right_upper_ax.get_position())
                right_lower_axes.append(right_lower_ax)

                pos = right_upper_ax.get_position()
                right_upper_ax.set_position([pos.x0, pos.y0 + pos.height*2/35, pos.width, pos.height * 33/35])
                right_lower_ax.set_position([pos.x0, pos.y0, pos.width, pos.height*1/35])

            for count, gap_idx in enumerate([2, 4, 9]):
                # Left panel: square heatmap
                sim = np.zeros((5,5))
                for i in range(5):
                    for j in range(i,5):
                        period1_pca, period2_pca = self.group.periods_pca[gap_idx][i], self.group.periods_pca[gap_idx][j]
                        angles = analysis.calculate_principal_angles(period1_pca.loading[:5].T, period2_pca.loading[:5].T)
                        sim[i,j] = 1 - np.mean(angles) / (np.pi/2)
                        sim[j,i] = sim[i,j]
                sns.heatmap(sim, ax = left_axes[count], cmap = 'binary', vmin=0, vmax=1, square=True, cbar = False)
                left_axes[count].set_aspect('auto')
                left_axes[count].set_xticklabels(['Noise1', 'Noise2', 'N2 w/o On', 'N2 w/o Both', 'Silence'], rotation = 0, fontsize = 11)
                left_axes[count].set_yticklabels(['Noise1', 'Noise2', 'N2 w/o On', 'N2 w/o Both', 'Silence'], fontsize = 11)
                left_axes[count].set_ylabel('Gap = ' + str(round(self.group.gaps[gap_idx]*1000)) +'ms', fontsize = 24)

                data = self.group.pop_response_stand[:, gap_idx, :]
                onset_pca = self.group.periods_pca[gap_idx][0]
                onset_post_pca =  self.group.periods_pca[gap_idx][1]
                
                onset_projection = onset_pca.loading @ data
                onset_post_projection = onset_post_pca.loading @ data

                #Mid panel: heatmap of projection onto N1 on-resp subspace
                PCs = 20
                sns.heatmap(Align_PC_Sign(onset_projection[:PCs,50:-150]), ax = mid_upper_axes[count], cmap = 'RdBu', cbar = False)
                sns.heatmap([self.group.gaps_label[gap_idx][50:-150]], ax=mid_lower_axes[count], cmap='Blues', vmin=0, vmax=1, cbar=False)
                
                # Right panel: heatmap of projection onto N2 on-resp subspace
                sns.heatmap(Align_PC_Sign(onset_post_projection[:PCs,50:-150]), ax = right_upper_axes[count], cmap = 'RdBu', cbar = False)
                sns.heatmap([self.group.gaps_label[gap_idx][50:-150]], ax=right_lower_axes[count], cmap='Blues', vmin=0, vmax=1, cbar=False)
            left_axes[0].set_title('Subspace Similarity', fontsize = 20)
            mid_upper_axes[0].set_title('Projection to Noise1 Onset Subspace', fontsize = 20)
            right_upper_axes[0].set_title('Projection to Noise2 Onset Subspace', fontsize = 20)

            # Remove ticks for cleaner look
            for axs in  mid_upper_axes + mid_lower_axes + right_upper_axes + right_lower_axes:
                axs.set_xticks([])
                axs.set_yticks([])
            
            return fig2
  
        def Draw_N1_N2_Similarity():
            fig3, axs = plt.subplots(1, 1, figsize = (6,6))
            sim = np.zeros(10)
            for count in range(10):
                period1_pca = self.group.periods_pca[count][0]
                period2_pca = self.group.periods_pca[count][1]
                angles = analysis.calculate_principal_angles(period1_pca.loading[:5].T, period2_pca.loading[:5].T)
                sim[count] = 1 - np.mean(angles) / (np.pi/2)

            axs.plot(np.array([0,1,2,4,8,16,32,64,128,256]), sim, color = 'darkblue')
            
            #axs.set_xticks([0,2,4,6,8],labels= ['0', '2', '8', '32', '128'], fontsize = 20)
            axs.set_xticks([0,50, 100, 150, 200, 250],labels= ['0', '50', '100', '150', '200', '250'], fontsize = 20)
            axs.set_yticks(np.arange(0.1, 0.7, 0.1),labels= ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], fontsize = 20)
            axs.set_ylabel('Similarity Index', fontsize = 24)
            axs.set_xlabel('Gap (ms)', fontsize = 24)
            plt.tight_layout()
            return fig3
            
        # Compare the similarity between the period-specific subspaces, with or without projection
        fig1 = Draw_Principal_Angles()
        # Compare N1 and N2 onset subspaces for 3 gaps: angles and projections (might be most intuitive)
        fig2 = Draw_Projection_to_Subspace()
        # Compare the similarity between the N1 and N2 onset subspaces for all gaps
        fig3 = Draw_N1_N2_Similarity()
        return fig1, fig2, fig3


class Decoder:
    def __init__(self, group):
        self.group = group
        self.gaps = group.gaps
        
    def Plot_Noise_Return_Silence(self, dim = 3):
        def calculate_distance(gap_idx, start, end):
            score_per_gap = self.group.pca.score_per_gap[:dim, gap_idx,:]
            origin = np.mean(score_per_gap[:,50:100], axis = 1)
            distance = []
            for t in range(start, end): 
                distance.append(np.sqrt(np.sum((origin - score_per_gap[:, t]) ** 2)))
            return np.array(distance)
        
        def smooth_scatter(x, y, method='lowess'):
            if len(y) < 3: return x, y
            # LOWESS smoothing
            smoothed = lowess(y, x, frac=0.3)
            return smoothed[:, 0], smoothed[:, 1]
        
        def smooth_scatter_peak(x, y):
            if len(y) < 3: return x, y
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]

            # Find approximate peak location
            window = min(len(y_sorted) // 5, 31)  # Adjust window size based on data length
            if window % 2 == 0:
                window += 1  # Ensure odd window size
            poly_order = min(2, window - 1)  # Ensure polynomial order is less than window size
            
            # Use Savitzky-Golay filter with different parameters for rising and falling parts
            y_smooth = savgol_filter(y_sorted, window, poly_order)
            peak_idx = np.argmax(y_smooth)
            
            # Separate rising and falling parts
            x_rise = x_sorted[:peak_idx+1]
            y_rise = y_sorted[:peak_idx+1]
            x_fall = x_sorted[peak_idx:]
            y_fall = y_sorted[peak_idx:]
            
            # Fit each part separately
            if len(x_rise) > 3:
                spl_rise = UnivariateSpline(x_rise, y_rise, k=3, s=len(y_rise)*0.1)
                y_rise_smooth = spl_rise(x_rise)
            else:
                y_rise_smooth = y_rise
                
            if len(x_fall) > 3:
                spl_fall = UnivariateSpline(x_fall, y_fall, k=3, s=len(y_fall)*0.1)
                y_fall_smooth = spl_fall(x_fall)
            else:
                y_fall_smooth = y_fall
            
            # Combine the parts
            x_smooth = np.concatenate([x_rise, x_fall[1:]])
            y_smooth = np.concatenate([y_rise_smooth, y_fall_smooth[1:]])
            return x_smooth, y_smooth
        
        fig1, axs = plt.subplots(2, 5, figsize = (30, 12))
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
                
            baseline = np.mean(calculate_distance(i, start = 50, end = 100))
            
            pre_distance = calculate_distance(i, start = pre_gap_start, end = pre_gap_end)
            axs[i].scatter(np.arange(-len(pre_distance),0,1), pre_distance, color = 'green', alpha = 0.3)
            smoothed_x, smoothed_y = smooth_scatter(np.arange(-len(pre_distance),0,1), pre_distance)
            axs[i].plot(smoothed_x, smoothed_y, linewidth = 4, color = 'darkgreen')
  
            distance = calculate_distance(i, start = gap_start, end = gap_end)
            axs[i].scatter(np.arange(len(distance)), distance, color = 'blue', alpha = 0.3)
            smoothed_x, smoothed_y = smooth_scatter_peak(np.arange(len(distance)), distance)
            axs[i].plot(smoothed_x, smoothed_y, linewidth = 4, color = 'darkblue')
            
            time = np.argsort(smoothed_y)[::-1][0]
            axs[i].axvline(x=time, linestyle = '--', color = 'red', label = 'Peak Time: ' + str(time) + 'ms')
            
            for t in range(len(smoothed_y)):
                if abs(smoothed_y[t] - baseline) < 1e-2: 
                    axs[i].plot([t,t], [-0.1, smoothed_y[t]], linestyle = '--', color = 'purple', label = 'Decay Time: ' + str(t) + 'ms')
                    break 
                
            post_distance = calculate_distance(i, start = post_gap_start, end = post_gap_end)
            axs[i].scatter(len(distance) + np.arange(len(post_distance)), post_distance, color = 'orange', alpha = 0.3)
            smoothed_x, smoothed_y = smooth_scatter_peak(len(distance) + np.arange(len(post_distance)), post_distance)
            axs[i].plot(smoothed_x, smoothed_y, linewidth = 4, color = 'brown')

            axs[i].legend(loc = 'upper right', fontsize = 20)
            axs[i].axhline(y=baseline, linestyle = '--', color = 'grey')
            
            if i > 0: 
                axs[i].set_xlim((-50, 300))
                axs[i].set_xticks(np.arange(-50, 305, 50), ['-50', '0', '50', '100', '150', '200', '250','300'], fontsize = 20)
            else: 
                axs[i].set_xlim((-50, 550))
                axs[i].set_xticks(np.arange(0, 505, 100), ['0', '100', '200', '300', '400', '500'], fontsize = 20)
            axs[i].set_ylim((0, 3))
            axs[i].set_xlabel('Time Since Gap Start (ms)', fontsize = 20)
            axs[i].set_ylabel('Distance to Noise On-Set', fontsize = 20)
            axs[i].set_title('Gap = ' + str(gap_dur) + 'ms', fontsize = 24)
        axs[0].set_xlabel('Time Since Post-Noise Off-Set (ms)', fontsize = 20)
        axs[0].set_title('Silence = 550 ms', fontsize = 24)
        plt.tight_layout()
        
        onset_pca= analysis.PCA(self.group.pop_response_stand[:,0, 100:200], multiple_gaps=False)
        offset_pca= analysis.PCA(self.group.pop_response_stand[:,0, 450:550], multiple_gaps=False)
        noise_pca= analysis.PCA(self.group.pop_response_stand[:,0, 350:450], multiple_gaps=False)
        background_pca= analysis.PCA(self.group.pop_response_stand[:,0, 0:99], multiple_gaps=False)

        subspaces = [onset_pca.loading, offset_pca.loading, noise_pca.loading, background_pca.loading]
        subspaces_label = ['On-Resp Subspace', 'Off-Resp Subspace', 'Noise-Resp Subspace', 'Background Subspace']
        
        fig2, axs = plt.subplots(2, 2, figsize = (12, 12))
        axs = axs.flatten()
        gap_dur = round(self.group.gaps[0]*1000)
        for i in range(4):
            subspace = subspaces[i]
            self.group.pca.loading = subspace
            self.group.pca.score = subspace @ (self.group.pca.data.reshape(self.group.pca.data.shape[0], -1) )
            self.group.pca.Separate_Multiple_Gaps()
            
            baseline = np.mean(calculate_distance(i, start = 50, end = 100))
            
            pre_distance = calculate_distance(0, start = 300, end = 351)
            axs[i].scatter(np.arange(-len(pre_distance),0,1), pre_distance, color = 'green', alpha = 0.3)
            smoothed_x, smoothed_y = smooth_scatter(np.arange(-len(pre_distance),0,1), pre_distance)
            axs[i].plot(smoothed_x, smoothed_y, linewidth = 4, color = 'darkgreen')
            
            distance = calculate_distance(0, start = 350 + gap_dur + 100, end = 1000)
            axs[i].scatter(np.arange(len(distance)), distance, color = 'blue', alpha = 0.3)
            smoothed_x, smoothed_y = smooth_scatter_peak(np.arange(len(distance)), distance)
            axs[i].plot(smoothed_x, smoothed_y, linewidth = 4, color = 'darkblue')
            
            time = np.argsort(smoothed_y)[::-1][0]
            axs[i].axvline(x=time, linestyle = '--', color = 'red', label = 'Peak Time: ' + str(time) + 'ms')
            
            for t in range(len(smoothed_y)):
                if abs(smoothed_y[t] - baseline) < 1e-2: 
                    axs[i].plot([t,t], [-0.1, smoothed_y[t]], linestyle = '--', color = 'purple', label = 'Decay Time: ' + str(t) + 'ms')
                    break 
                
            axs[i].legend(loc = 'upper right', fontsize = 20)
            axs[i].axhline(y=baseline, linestyle = '--', color = 'grey')
            
            axs[i].set_ylim((-0.1, 3.5))
            axs[i].set_xticks(np.arange(0, 600, 100), ['0', '100', '200', '300', '400', '500'], fontsize = 20)
            axs[i].set_yticks(np.arange(0,3.1, 0.5), ['0', '0.5','1.0','1.5','2.0','2.5','3.0'], fontsize = 20)
            axs[i].set_xlabel('Time Since Post-Noise Off-Set (ms)', fontsize = 20)
            axs[i].set_xlabel('Distance to Noise On-Set', fontsize = 20)
            axs[i].set_title(subspaces_label[i], fontsize = 24)
        plt.tight_layout()
        self.group.pca = analysis.PCA(self.group.pop_response_stand)

        return fig1, fig2

    def Get_PCs(self, gap_idx):
        PC, PCs = [0,1,2],[]
        for j in range(len(PC)):
            scores = self.group.pca.score_per_gap[PC[j]]
            score_per_gap = scores[gap_idx]
            if Flip(score_per_gap): score_per_gap = score_per_gap * (-1)
            PCs.append(score_per_gap)
        return  np.array(PCs)
        
    def Plot_Binary_Decoder(self):
        def Get_Prediction(gap_idx):
            PCs = self.Get_PCs(gap_idx)
            X = PCs.T  
            y = self.group.gaps_label[gap_idx]
            reg = LinearRegression()
            reg.fit(X, y) 
            s = reg.predict(X)
            return s 
        
        def Threshold_Prediction(s):
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(np.array(s[:700]).reshape(-1, 1))
            centers = kmeans.cluster_centers_
            sort_idx =np.argsort(centers.reshape(1,-1))[0]
            boundary = (centers[sort_idx[0]][0] + centers[sort_idx[1]][0])/2
            noises = np.zeros(len(s))
            for t in range(len(s)):
                if s[t] > boundary: noises[t] = 1 
            return noises, boundary
            
        gaps = [3, 9]
        fig, axs = plt.subplots(1, 2, figsize = (42, 6))
        for i in range(len(gaps)):
            gap_idx = gaps[i]
            
            s = Get_Prediction(gap_idx)
            s_thresholded, boundary = Threshold_Prediction(s)
            
            axs[i].plot(s, label = 's(t)')
            axs[i].plot(s_thresholded, color = 'red', label = 'Predicted Noise')
            axs[i].axhline(y = boundary, color = 'grey', linestyle = ':')
            ymin, ymax = axs[i].get_ylim()
            mask = self.group.gaps_label[gap_idx] == 1
            axs[i].fill_between(np.arange(len(self.group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = 'dimgrey', alpha = 0.1, label = 'True Noise')
            axs[i].legend(loc = 'upper right', fontsize = 18)
            axs[i].set_title('Gap = ' + str(round(self.group.gaps[gap_idx]*1000)) + ' ms', fontsize = 24)
        plt.tight_layout()
        return fig
        
    def Plot_HMM_Decoder(self, n_state = 4):
        gaps = [3, 9]
        fig, axs = plt.subplots(4, 2, figsize = (42, 6))
        for i in range(len(gaps)):
            gap_idx = gaps[i]
            
            PCs = self.Get_PCs(gap_idx)
            HMM = analysis.HMM(PCs, state = n_state)
            HMM.Fit_Model()
            
            colors = ['black', 'red', 'blue']
            colors_state = ['black', 'red', 'blue', 'green']
            labels = ['PC 1', 'PC 2', 'PC 3']

            axs[0,i].plot(np.arange(1000), PCs[0], color=colors[0])
            axs[1,i].plot(np.arange(1000), PCs[1], color=colors[1])
            axs[2,i].plot(np.arange(1000), PCs[2], color=colors[2])
            
            sort_idx = np.argsort(HMM.parameters[0].T[2])[::-1]
            print(HMM.parameters[0].T[2][sort_idx])
            for j in range(HMM.n_state):
                state = sort_idx[j]
                mask = HMM.states == state
                axs[3,i].fill_between(np.arange(len(HMM.states)), 0, 0.4, where=mask, color=colors_state[j], alpha=0.7)

            for j in range(3):
                ymin, ymax = axs[j,i].get_ylim()
                mask = self.group.gaps_label[gap_idx] == 1
                axs[j,i].fill_between(np.arange(len(self.group.gaps_label[gap_idx])), ymin, ymax, where=mask, color = 'dimgrey', alpha = 0.1, label = 'True Noise')     
                axs[j,i].set_xticks([0, 200, 400, 600, 800], labels=[0, 200, 400, 600, 800], fontsize=16)
                if j < 3:
                    axs[j,i].set_ylabel(labels[j], fontsize=24)
                else: 
                    axs[j,i].spines['left'].set_visible(False)
                    axs[j,i].spines['bottom'].set_visible(False)
            axs[3,i].set_xlabel('Time (ms)', fontsize=20)
            axs[0,i].set_title('Gap = ' + str(round(self.group.gaps[gap_idx]*1000)) + ' ms', fontsize = 24)
        plt.tight_layout()
        return fig
            
            


class System:
    def __init__(self, group, Model):
        self.group = group
        self.Model = Model
        self.model = Model.model 
        
    def style_3d_ax(self, ax):
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
        
    def Draw_Model(self, gap_idx):
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
                ymin, ymax = axs[j].get_ylim()
                axs[j].axhline(y=0, color = 'grey', linestyle = ':')
                axs[j].fill_between([100, 350], ymin, ymax, color='dimgrey', alpha=0.1)
                axs[j].fill_between([350+self.model.gap_dur, 450+self.model.gap_dur], ymin, ymax, color='dimgrey', alpha=0.1)
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
                                            ls=linestyle[k], c=linecolors[k], linewidth = 3, label = labels[k])
                axs[1].plot(PC1[starts[k]:ends[k]], PC2[starts[k]:ends[k]], PC3[starts[k]:ends[k]], 
                                            ls=linestyle[k], c=linecolors[k], linewidth = 3, label = labels[k])

            for i in range(2):
                axs[i].legend(loc = 'upper center', fontsize = 14)
                self.style_3d_ax(axs[i])
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
            axs[0].plot(np.arange(len(self.model.train_loss)), self.model.train_loss)
            axs[1].plot(np.arange(len(self.model.validate_loss)), self.model.validate_loss)
            axs[0].set_xlabel('Iter', fontsize = 16)
            axs[1].set_xlabel('Iter', fontsize = 16)
            axs[0].set_ylabel('Train Loss', fontsize = 16)
            axs[1].set_ylabel('Validaiton Loss', fontsize = 16)
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
                model.Run(noise = True)
                
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
                for t in range(len(distance)):
                    if abs(distance[t] - baseline) < 1e-2: 
                        axs[i].plot([t,t], [-0.1, distance[t]], linestyle = '--', color = 'purple', label = 'Decay Time: ' + str(t) + 'ms')
                        break 
                
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
        
        def Draw_Fix_Point():
            def Draw_Random_Trajectory(time, start_points, W, n_steps, point_color, axs):
                def derivative(point, W):
                    return ((W @ point.reshape(-1, 1) + (self.model.OnS[time] * self.model.OnRe)+ (self.model.OffS[time] * self.model.OffRe)).T * self.model.Nt.T).reshape(1, -1)[0]

                def simulate_trajectory(W, start_point, dt=1, n_steps=200):
                    trajectory = np.zeros((n_steps, 3))
                    trajectory[0] = start_point
                    
                    for i in range(1, n_steps):
                        curr_derivative = derivative(trajectory[i-1], W)
                        trajectory[i] = trajectory[i-1] + dt * curr_derivative
                        
                    return trajectory
                
                colors = np.linspace(0, 1, n_steps)
                for start in start_points:
                    traj = simulate_trajectory(W, start, dt = 1, n_steps = n_steps)
                    for i in range(len(traj)-1):
                        axs.plot3D(traj[i:i+2,0], traj[i:i+2,1], traj[i:i+2,2], 
                                color=plt.cm.plasma(colors[i]), 
                                alpha=0.6)
                    axs.scatter(*start, color=point_color, s=30)

                return axs 

            # Plot
            fig6, axs = plt.subplots(1,3,figsize=(34, 10), subplot_kw={'projection': '3d'})
            W = self.model.W
            eigenvals, eigenvecs = np.linalg.eig(W)
            
            #Draw fixed points
            fix_point = np.linalg.solve(W, np.zeros((3,1)))
            near_stable_points = []
            for i in [-0.1, 0, 0.1]:
                for j in [-0.1, 0, 0.1]:
                    for k in [-0.1, 0, 0.1]:
                        near_stable_points.append(np.array([i+fix_point[0,0],j+fix_point[1,0],k+fix_point[2,0]]))
            axs[0] = Draw_Random_Trajectory(0, near_stable_points, W, 100, 'darkblue', axs[0]) 
            axs[0].scatter(fix_point[0,0],fix_point[1,0],fix_point[2,0], color = 'brown', s = 30, label = 'Fixed Point')
            
            for i in range(3):
                real, imag = np.real(eigenvals[i]), np.imag(eigenvals[i])
                if imag == 0: label = str(round(real,4))
                elif imag > 0: label = str(round(real,4)) + ' + ' + str(round(imag,4)) + 'j'
                else: label = str(round(real,4)) + ' - ' + str(round(-imag,4)) + 'j'
                axs[0].plot([], [], color = 'black', label = label)  
            axs[0].set_xlim([fix_point[0,0]-0.12, fix_point[0,0]+0.12])
            axs[0].set_xticks([fix_point[0,0]-0.12, 0, fix_point[0,0]+0.12], [-0.12, 0, 0.12], fontsize = 16)
            axs[0].set_ylim([fix_point[1,0]-0.12, fix_point[1,0]+0.12])
            axs[0].set_yticks([fix_point[1,0]-0.12, 0, fix_point[1,0]+0.12], [-0.12, 0, 0.12], fontsize = 16)
            axs[0].set_zlim([fix_point[2,0]-0.12, fix_point[2,0]+0.12])  
            axs[0].set_zticks([fix_point[2,0]-0.12, 0, fix_point[2,0]+0.12], [-0.12, 0, 0.12], fontsize = 16)    
                        
            PC1 = gaussian_filter1d(self.model.PCs[0], sigma=sigma)
            PC2 = gaussian_filter1d(self.model.PCs[1], sigma=sigma)
            PC3 = gaussian_filter1d(self.model.PCs[2], sigma=sigma)

            if Flip(PC1): PC1 *= -1
            if Flip(PC2): PC2 *= -1
            if Flip(PC3): PC3 *= -1

            gap_dur = round(256+350)
            linecolors = ['grey', 'green', 'black', 'lightgreen', 'black']
            linestyle = ['-', '--', '-', '--', ':']
            labels = ['pre-N1', 'noise1','gap', 'noise2', 'post-N2']
            starts = [0, 100, 350, gap_dur, gap_dur + 100]
            ends = [100, 350, gap_dur, gap_dur+100, 1000]
            for k in range(3):
                axs[1].plot(PC1[starts[k]:ends[k]], PC2[starts[k]:ends[k]], PC3[starts[k]:ends[k]], 
                                                ls=linestyle[k], c=linecolors[k], linewidth = 2, alpha = 0.5)
                axs[2].plot(PC1[starts[k]:ends[k]], PC2[starts[k]:ends[k]], PC3[starts[k]:ends[k]], 
                                                ls=linestyle[k], c=linecolors[k], linewidth = 2, alpha = 0.5)


            for time in [105, 110, 115, 120, 125, 135, 150, 200, 275]:
                axs[1].scatter(PC1[time],PC2[time],PC3[time], color = 'magenta', s = 30)
                axs[1] = Draw_Random_Trajectory(time, [np.array([PC1[time],PC2[time],PC3[time]])], W, 200, 'magenta', axs[1])
                
                ext_input = self.model.OnS[time]*self.model.OnRe + self.model.OffS[time]*self.model.OffRe
                fix_point = np.linalg.solve(W, -ext_input)
                axs[1].scatter(fix_point[0,0],fix_point[1,0],fix_point[2,0], color = 'saddlebrown', s = 30)
            axs[1].scatter([], [], color = 'saddlebrown', s = 30, label = 'Fixed Point of Noise')
            OnFix = - np.linalg.inv(W) @ self.model.OnRe
            OffFix = - np.linalg.inv(W) @ self.model.OffRe
            axs[1].quiver(0, 0, 0, OnFix[0,0], OnFix[1,0], OnFix[2,0],  color='saddlebrown', arrow_length_ratio=0.1, alpha = 0.5)
            axs[1].quiver(0, 0, 0, OffFix[0,0], OffFix[1,0], OffFix[2,0],  color='sandybrown', arrow_length_ratio=0.3, alpha = 0.3)
            angle = analysis.calculate_vector_angle(OnFix.reshape(1,-1)[0], OffFix.reshape(1,-1)[0])
            axs[1].scatter([], [], color = 'saddlebrown', s = 30, label = 'Angle: ' + str(round(angle,1)))
            
            for time in [350, 355, 370, 390, 420, 500, 600]:
                axs[2].scatter(PC1[time],PC2[time],PC3[time], color = 'magenta', s = 30)
                axs[2] = Draw_Random_Trajectory(time, [np.array([PC1[time],PC2[time],PC3[time]])], W, 200, 'magenta', axs[2])
                
                ext_input = self.model.OnS[time]*self.model.OnRe + self.model.OffS[time]*self.model.OffRe
                fix_point = np.linalg.solve(W, -ext_input)
                axs[2].scatter(fix_point[0,0],fix_point[1,0],fix_point[2,0], color = 'sandybrown', s = 30)
            axs[2].scatter([], [], color = 'sandybrown', s = 30, label = 'Fixed Point of Gap')
            axs[2].quiver(0, 0, 0, OnFix[0,0], OnFix[1,0], OnFix[2,0],  color='saddlebrown', arrow_length_ratio=0.1, alpha = 0.3)
            axs[2].quiver(0, 0, 0, OffFix[0,0], OffFix[1,0], OffFix[2,0],  color='sandybrown', arrow_length_ratio=0.3, alpha = 0.5)
            axs[2].scatter([], [], color = 'sandybrown', s = 30, label = 'Angle: ' + str(round(angle,1)))
            
            for i in range(1,3):
                axs[i].scatter([], [], color = 'magenta', s = 30, label = 'Trajectory Points')
                axs[i].set_xlim((-1,3))
                axs[i].set_ylim((-0.5,2))
                axs[i].set_zlim((-1,1)) 

            for i in range(3):
                axs[i].legend(loc = 'upper left', fontsize = 18)  
                axs[i].set_xlabel('X', fontsize = 18)
                axs[i].set_ylabel('Y', fontsize = 18)
                axs[i].set_zlabel('Z', fontsize = 18)
            axs[0].set_title('Fixed Point', fontsize = 20)
            axs[1].set_title('Time-Varying Fixed Point: Noise', fontsize = 20)
            axs[2].set_title('Time-Varying Fixed Point: Gap', fontsize = 20)
            plt.tight_layout() 
            return fig6
                        
                        
        self.model.gap_idx = gap_idx
        self.model.Set_Gap_Dependent_Params()
        self.model.Run(noise = True)
        
        fig1 = Draw_Trace_2d() 
        fig2 = Draw_Trace_3d()
        fig3 = Draw_Parameters() 
        fig4 = Draw_Loss_with_Iter()
        fig5 = Draw_Gap_Duration_Recognition()
        fig6 = Draw_Fix_Point()
        
        return fig1, fig2, fig3, fig4, fig5, fig6
        
    def Draw_Simulation(self, SoundInput):
        def get_periods(S):
            """Get start, end, colors arrays for plotting periods"""
            unique_vals = np.unique(S)
            smaller_val, larger_val = min(unique_vals), max(unique_vals)
            
            # Initialize lists
            starts, ends, colors = [], [], []
            
            # Get periods
            current_val = S[0]
            period_start = 0
            
            for i in range(1, len(S)):
                if S[i] != current_val:
                    starts.append(period_start)
                    ends.append(i)
                    colors.append('black' if current_val == smaller_val else 'darkgreen')
                    period_start = i
                    current_val = S[i]
            
            # Add final period
            starts.append(period_start)
            ends.append(len(S))
            colors.append('black' if current_val == smaller_val else 'darkgreen')
            
            return starts, ends, colors
        
        def Draw_Trace_2d():
            colors_unit = ['black', 'red', 'blue']
            fig1, axs = plt.subplots(4, 1, sharex=True, figsize=(17, 8))
            axs[0].plot(np.arange(len(SoundInput)), N[0], color=colors_unit[0])
            axs[1].plot(np.arange(len(SoundInput)), N[1], color=colors_unit[1])
            axs[2].plot(np.arange(len(SoundInput)), N[2], color=colors_unit[2])
            axs[3].plot(np.arange(len(SoundInput)), OnS, color='darkgreen')
            axs[3].plot(np.arange(len(SoundInput)), OffS, color='limegreen')

            for j in range(4):
                ymin, ymax = axs[j].get_ylim()
                for t in range(1, len(starts), 2):
                    axs[j].fill_between([starts[t], ends[t]], ymin, ymax, color='dimgrey', alpha=0.1)
                axs[j].set_xticks([0 + 200*i for i in range(len(SoundInput)//200)], labels=[0 + 200*i for i in range(len(SoundInput)//200)], fontsize=16)
                if j <3:
                    axs[j].set_ylabel('Unit' + str(j+1), fontsize=24)
            axs[3].set_xlabel('Time (ms)', fontsize=20)
            plt.tight_layout()
            return fig1
            
        def Draw_Trace_3d():
            fig2, axs = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': '3d'})    
            x = gaussian_filter1d(N[0], sigma=sigma-1)
            y = gaussian_filter1d(N[1], sigma=sigma-1)
            z = gaussian_filter1d(N[2], sigma=sigma-1)

            for k in range(len(starts)):
                axs.plot(x[starts[k]:ends[k]], y[starts[k]:ends[k]], z[starts[k]:ends[k]], 
                                            c=linecolors[k], linewidth = 3)

            #axs.legend(loc = 'upper center', fontsize = 14)
            self.style_3d_ax(axs)
            axs.set_xlim(min(x), max(x))
            axs.set_ylim(min(y), max(y))
            axs.set_zlim(min(z), max(z))
            axs.set_title('Predicted Units', fontsize = 24)
            plt.tight_layout()   
            return fig2
        
        def Draw_Decode_Gap():
            self.model.Set_Params_of_Least_Loss()
            self.model.gap_idx = 9
            self.model.Run()
            X = self.model.PCs.T  
            y = self.group.gaps_label[9]

            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, y)
            
            s = reg.predict(self.Model.N.T)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(np.array(s[:700]).reshape(-1, 1))
            centers = kmeans.cluster_centers_
            sort_idx =np.argsort(centers.reshape(1,-1))[0]
            boundary = (centers[sort_idx[0]][0] + centers[sort_idx[1]][0])/2

            noises = np.zeros(len(s))
            for t in range(len(s)):
                if s[t] > boundary: noises[t] = 1
                
            fig3, axs = plt.subplots(1, 1, figsize = (30, 12))
            axs.plot(s)
            axs.plot(noises, color = 'red', label = 'Predicted Noise')
            axs.axhline(y = boundary, color = 'grey', linestyle = ':', label = 'Boundary')
            ymin, ymax = axs.get_ylim()
            for t in range(1, len(starts), 2):
                axs.fill_between([starts[t], ends[t]], ymin, ymax, color = 'dimgrey', alpha = 0.1)
            axs.fill_between([starts[1], ends[1]], ymin, ymax, color = 'dimgrey', alpha = 0.1, label = 'True Noise')
            axs.legend(loc = 'upper right', fontsize = 20)
            
            HMM = analysis.HMM(X.T, state = 4)
            HMM.Fit_Model()
            predicted_observation = self.Model.N.T
            predicted_states = HMM.model.most_likely_states(predicted_observation)
            colors_state = ['black', 'red', 'blue', 'green']
            sort_idx = np.argsort(HMM.parameters[0].T[2])[::-1]
            for j in range(HMM.n_state):
                state = sort_idx[j]
                mask = predicted_states == state
                axs.fill_between(np.arange(len(predicted_states)), -0.4, -0.25, where=mask, color=colors_state[j], alpha=0.7)
            axs.tick_params(axis='both', labelsize = 24)
            axs.set_xlabel('Time (ms)', fontsize = 24)
            plt.tight_layout()
            return fig3
                
        starts, ends, linecolors = get_periods(SoundInput)
        self.Model.Run(SoundInput)
        OnS, OffS = self.Model.OnS, self.Model.OffS
        N = self.Model.N   
        
        fig1 = Draw_Trace_2d()
        fig2 = Draw_Trace_3d()
        fig3 = Draw_Decode_Gap()
        
        return fig1, fig2, fig3
    
    def Draw_Gap_Threshold_Simulation(self):
        self.model.Set_Params_of_Least_Loss()
        self.model.gap_idx = 9
        self.model.Run()
        X = self.model.PCs.T  
        y = self.group.gaps_label[9]

        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        gap_detections = np.zeros(64-1)
        for gap_dur in range(1, 64):
            S = np.zeros(1000) + 10
            for t in range(100, 700): S[t] = self.model.S_on

            for t in range(350, 350+gap_dur): S[t] = 10

            self.Model.Run(SoundInput=S)

            # Generate continuous state prediction
            s = reg.predict(self.Model.N.T)
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(np.array(s[:700]).reshape(-1, 1))
            centers = kmeans.cluster_centers_
            sort_idx =np.argsort(centers.reshape(1,-1))[0]
            boundary = (centers[sort_idx[0]][0] + centers[sort_idx[1]][0])/2

            noises = np.zeros(len(s))
            for t in range(len(s)):
                if s[t] > boundary: noises[t] = 1 
                
            start, end = 0, 0    
            for t in range(350, 350+100):
                if noises[t] == 0 and noises[t-1] == 1: start = t
                if noises[t] == 1 and noises[t-1] == 0: 
                    end = t
                    break
            gap_detections[gap_dur-1] = end-start 
        
        fig, axs = plt.subplots(1, 1, figsize = (12, 12))
        axs.scatter(np.arange(1, 64, 1), gap_detections)
        axs.plot(np.arange(1, 64, 1), np.arange(1, 64, 1), color = 'red', linestyle = ":")
        axs.set_ylim(-1,65)
        axs.set_xlabel('Simulated Gap Duration (ms)', fontsize = 26)
        axs.set_ylabel('Predicted Gap Duration (ms)', fontsize = 26)
        axs.tick_params(axis='both', labelsize = 24)
        plt.tight_layout()

        return fig
    
    
class Summary:
    def __init__(self, groups):
        self.groups = groups
        self.geno_types = [group.geno_type for group in groups]
        self.hearing_type = [group.hearing_type for group in groups]
        
        self.colors = ['tab:orange', 'tab:orange', 'tab:grey', 'tab:grey']
        self.ls = ['--', '-', '--', '-']
        self.labels = [self.geno_types[i] + '_' + self.hearing_type[i]
                  for i in range(4)]
    
    def Plot_Unit_Type(self):
        fig, axs = plt.subplots(1,1,figsize=(5,5),layout='constrained')
        types = ['on', 'both', 'off', 'none']
        for i in range(4):
            Group = self.groups[i]
            num = np.array([len(Group.unit_type[Group.unit_type==types[j]])/len(Group.unit_type) for j in range(4)])

            a = 1
            fc = 'None'
            ec = self.colors[i]
            axs.bar(i, num[0], bottom=0, 
                    facecolor=fc, edgecolor=ec, hatch='/////', label='Onset only',alpha=a)
            axs.bar(i, num[1], bottom=num[0], 
                    facecolor=fc, edgecolor=ec, hatch='xxxxx', label='Onset & Offset',alpha=a)
            axs.bar(i, num[2], bottom=num[0:2].sum(), 
                    facecolor=fc, edgecolor=ec, hatch='\\\\\\\\\\', label='Offset only',alpha=a)
            axs.bar(i, num[3], bottom=num[0:3].sum(),
                    facecolor=fc, edgecolor=ec, hatch='', label='Neither',alpha=a)

        plt.ylim([0,1])
        plt.ylabel('Percentage (%)',fontsize=16)
        plt.yticks([0,1],[0, 100], fontsize=14)
        plt.xticks([0,1,2,3],['$\mathit{Df1}$/+\nNonHL', '$\mathit{Df1}$/+\nHL', 'WT\nNonHL', 'WT\nHL'],fontsize=16)
        return fig
    
    def Plot_Components_Correlation(self):
        fig, axs = plt.subplots(1, 2, figsize = (11, 5))
        for i in range(4):
            label = self.labels[i]
            group = self.groups[i]
            corre_on, corre_off = [], []
            for i in range(10):
                corre_on.append(abs(np.corrcoef(abs(group.pca.loading[i]), group.unit_onset)[0,1]))
                corre_off.append(abs(np.corrcoef(abs(group.pca.loading[i]), group.unit_offset)[0,1]))
            axs[0].plot(np.arange(1,11,1), corre_on, label = label)
            axs[1].plot(np.arange(1,11,1), corre_off, label = label)
        for i in range(2):
            axs[i].legend(loc = 'upper right', fontsize = 12)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xticks([2,4,6,8,10])
            axs[i].set_xlabel('PC#', fontsize = 14)
            axs[i].set_ylabel('Abs. Corre. Coef.', fontsize = 14)
        axs[0].set_yticks([0,0.5])
        axs[1].set_yticks([0,0.5])
        axs[0].set_title('On-set', fontsize = 20)
        axs[1].set_title('Off-set', fontsize = 20)
        plt.tight_layout()
        return fig
    
    def Plot_PCA_Variance(self):
        fig, axs = plt.subplots(1, 1, figsize=(
            5, 5), sharex=True, layout='constrained')
        for i in range(4):
            Group = self.groups[i]
            axs.plot(np.arange(10), Group.pca.variance[:10],
                        color=self.colors[i], ls=self.ls[i], label=self.labels[i])
        axs.legend(loc='upper right', fontsize=12)
        axs.set_xlabel('PC#', fontsize=14)
        axs.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=12)
        axs.set_ylabel('Variance Explained', fontsize=14)
        return fig

    def Plot_Distance_Evoke_Ratio(self, PC, print=False):
        def get_evoked_ratio(euclidian_distance):
            evoked_ratio = np.zeros(10)
            gap = np.array([0., 0.001, 0.002, 0.004, 0.008,
                           0.016, 0.032, 0.064, 0.128, 0.256])
            bin = 5/1000
            for i in range(10):
                evoked_ratio[i] = np.sum(euclidian_distance[i, round(
                    70+gap[i]/bin):round(90+gap[i]/bin)])/np.sum(euclidian_distance[i, 20:70])
            return evoked_ratio

        fig, axs = plt.subplots(1, 2, figsize=(
            10, 5), sharex=True, layout='constrained')
        for i in range(4):
            group_pca = Latent(self.groups[i])
            group_pca.Calculate_Distance(PC=PC)
            axs[0].plot(np.arange(9), get_evoked_ratio(group_pca.euclidean_distance)[
                        1:], color=self.colors[i], ls=self.ls[i], label=self.labels[i])
            axs[1].plot(np.arange(10), get_evoked_ratio(
                group_pca.step_distance), color=self.colors[i], ls=self.ls[i], label=self.labels[i])
        for i in range(2):
            axs[i].legend(loc='lower right', fontsize=12)
            axs[i].set_xlabel('Gap (ms)', fontsize=14)
            axs[i].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], labels=[
                              1, '', 4, '', 16, '', 64, '', 256], fontsize=12)
        axs[0].set_ylabel(
            '2nd evoked distance / \n1st evoked distance', fontsize=14)
        axs[0].set_title('Euclidean Distance', fontsize=16)
        axs[1].set_title('Step Distance', fontsize=16)
        plt.tight_layout()
        if print:
            plt.show()
        return fig

    def Plot_Angle(self, PC=[0, 1, 3]):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
        for i in range(4):
            group_pca = Latent(self.groups[i])
            group_pca.Calculate_Angle(PC=PC)
            axs[0].plot(np.arange(10), group_pca.angle_noise,
                     color=self.colors[i], ls=self.ls[i], label=self.labels[i])
            axs[1].plot(np.arange(10), group_pca.angle_gap,
                     color=self.colors[i], ls=self.ls[i], label=self.labels[i])
            axs[2].plot(np.arange(10), group_pca.angle_fix_gap,
                     color=self.colors[i], ls=self.ls[i], label=self.labels[i])
        for i in range(3):
            axs[i].legend(loc='lower left', fontsize=14)
            axs[i].set_xlabel('Gap (ms)', fontsize=20)
            axs[i].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], labels=[
                        1, '', 4, '', 16, '', 64, '', 256], fontsize=16)
            axs[i].set_ylim((0, 120))
        axs[0].set_ylabel('Angle Degree', fontsize=20)
        axs[0].set_title('Post-Gap Noise', fontsize=24)
        axs[1].set_title('Gap', fontsize=24)
        axs[2].set_title('Post-Gap Silence', fontsize=24)
        return fig

    def Plot_Travel_Distance_First_Step(self, PC):

        fig, axs = plt.subplots(1, 3, figsize=(18,6))
        for i in range(4):
            group_pca = Latent(self.groups[i])
            group_pca.Calculate_First_Step_per_Gap(PC)
            axs[0].plot(np.arange(10), group_pca.first_step_pre_gap, color=self.colors[i], ls=self.ls[i], label=self.labels[i])
            axs[1].plot(np.arange(10), group_pca.first_step_gap, color=self.colors[i], ls=self.ls[i], label=self.labels[i])
            axs[2].plot(np.arange(10), group_pca.first_step_post_gap, color=self.colors[i], ls=self.ls[i], label=self.labels[i])
        
        for i in range(3):
            axs[i].legend(loc = 'lower right', fontsize = 12)
            axs[i].set_xlabel('Gap (ms)', fontsize = 14)
            axs[i].set_xticks([0,1,2,3,4,5,6,7,8,9],labels=[0, 1,'',4,'',16,'',64,'',256], fontsize = 12)
        axs[0].set_ylabel('$\Delta$Travel Distance :First 10-20 ms', fontsize = 14)
        axs[0].set_title('Pre-Gap Noise', fontsize=24)
        axs[1].set_title('Gap', fontsize=24)
        axs[2].set_title('Post-Gap Noise', fontsize=24)
        plt.tight_layout()
        
        return fig