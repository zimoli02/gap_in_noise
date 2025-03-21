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

imagepath = '/Volumes/Research/GapInNoise/Images/Subspace/'


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
    if params.Projection:
        fig = 

def main(subspace_params, group_labels):

    Groups, Labels = Load_Groups(group_labels)
    for label in Labels:
        print(f'Start Analysing {label}')
        Display_Group(subspace_params, Groups[label], label, file_path = imagepath)
        print('\n')
    
if __name__ == "__main__":
    main()