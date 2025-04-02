import os
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
import Function.projection_analysis as projection_analysis


basepath = '/Volumes/Zimo/Auditory/Data/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'
grouppath = '/Volumes/Research/GapInNoise/Data/Groups/'
gpfapath = '/Volumes/Research/GapInNoise/Data/GPFA/'
modelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel/'
newmodelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel_ss/'
subspacepath = '/Volumes/Research/GapInNoise/Data/Subspace/'
projectionpath = '/Volumes/Research/GapInNoise/Data/Projection/'

imagepath = '/Volumes/Research/GapInNoise/Images/Projection/'


import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially transparent artists will be rendered opaque")

# Default figure saving settings
fs = 10
custom_params = {
    "font.size": fs,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
}
sns.set_theme(style="ticks", rc=custom_params)

fig_dpi = 300
fig_format = 'eps'

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def SaveFig(fig, path):
    for fig_format in ['eps', 'png']:
        fig.savefig(path+f'.{fig_format}', dpi = fig_dpi)

def Load_Groups(group_labels):
    Groups, Labels = {},[]
    for geno_type in ['WT', 'Df1']: 
        for hearing_type in ['NonHL', 'HL']:
            label = geno_type + '_' + hearing_type
            if label in group_labels:
                Labels.append(label)
                with open(grouppath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                    Group = pickle.load(file)
                Groups[label] = Group
    return Groups, Labels

def Display_Group(params, Group, label, file_path = '../Images/Subspace/'):
    if params.Dimensionality_Reduction:
        sub_file_path = check_path(file_path + label + '/')
        fig_variance, fig_projection = projection_analysis.Low_Dim_Activity(Group)
        SaveFig(fig_variance, sub_file_path + 'Variance')
        SaveFig(fig_projection, sub_file_path + 'Projection')
        print('Dimensionality Reduction Completed!')
        plt.close('all')
        
    if params.Low_Dim_Activity_by_Space:
        sub_file_path = check_path(file_path + label + '/')
        fig_hist = projection_analysis.Low_Dim_Activity_by_Space(Group, short_gap = 3, long_gap = 9)
        SaveFig(fig_hist, sub_file_path + 'Histogram_by_Space')
        
    if params.Low_Dim_Activity_Divergence_by_Space:
        sub_file_path = check_path(file_path + label + '/')
        fig_KL = projection_analysis.Low_Dim_Activity_Divergence_by_Space(Group, short_gap = 3, long_gap = 9)
        SaveFig(fig_KL, sub_file_path + 'KLDivergenceby_Space')
        
    if params.Low_Dim_Activity_in_Different_Space:
        for space_name in ['On', 'Off']:
            sub_file_path = check_path(file_path + label + f'/{space_name}/')
            fig_projection, fig_KL = projection_analysis.Low_Dim_Activity_in_Different_Space(Group, short_gap = 3, long_gap = 9, 
                                                                                            space_name = space_name, period_length = 100, offset_delay = 10)
            SaveFig(fig_projection, sub_file_path + 'Projection_in_Different_Space')
            SaveFig(fig_KL, sub_file_path + 'KLDivergence_in_Different_Space')

    if params.Low_Dim_Activity_in_Space:
        sub_file_path = check_path(file_path + label + '/')
        fig_2d, fig_3d = projection_analysis.Low_Dim_Activity_Manifold(Group, short_gap = 5, long_gap = 9)
        SaveFig(fig_2d[0], sub_file_path + '2D_Short_Gap')
        SaveFig(fig_2d[1], sub_file_path + '2D_Long_Gap')
        SaveFig(fig_3d[0], sub_file_path + '3D_Short_Gap')
        SaveFig(fig_3d[1], sub_file_path + '3D_Long_Gap')
        print('Low_Dim_Activity_in_Space Completed')
        plt.close('all')
        
    if params.Predict_Off_using_Low_Dim_Activity:
        sub_file_path = check_path(file_path + label + '/')
        fig_Efficient_Dim, fig_Model_Prediction = projection_analysis.Binary_Classifier(Group, subspace = Group.pca.loading)
        SaveFig(fig_Efficient_Dim, sub_file_path + 'Efficient_Dim')
        SaveFig(fig_Model_Prediction, sub_file_path + 'Model_Prediction')
        print('Predict_Off_using_Low_Dim_Activity Completed')
        plt.close('all')
        
def main(subspace_params, group_labels):

    Groups, Labels = Load_Groups(group_labels)
    for label in Labels:
        print(f'Start Analysing {label}')
        Display_Group(subspace_params, Groups[label], label, file_path = imagepath)
        print('\n')
    
if __name__ == "__main__":
    main()