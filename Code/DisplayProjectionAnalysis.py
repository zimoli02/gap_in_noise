import os
from PIL import Image
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
import Function.data as data
import Function.analysis as analysis 
import Function.subspace_analysis as subspace_analysis
import Function.projection_analysis as projection_analysis
import Function.plot as plot

basepath = '/Volumes/Zimo/Auditory/Data/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'
grouppath = '/Volumes/Research/GapInNoise/Data/Groups/'
gpfapath = '/Volumes/Research/GapInNoise/Data/GPFA/'
modelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel/'
newmodelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel_ss/'
subspacepath = '/Volumes/Research/GapInNoise/Data/Subspace/'
projectionpath = '/Volumes/Research/GapInNoise/Data/Projection/'

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

@dataclass
class DisplayParams:
    Projection: bool = False

def SaveFig(fig, path):
    for fig_format in ['eps', 'png']:
        fig.savefig(path+f'.{fig_format}', dpi = fig_dpi)

def Load_Groups():
    Groups, Labels = {},[]
    for geno_type in ['WT', 'Df1']: 
        for hearing_type in ['NonHL', 'HL']:
            label = geno_type + '_' + hearing_type
            Labels.append(label)
            with open(grouppath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                Group = pickle.load(file)
            Groups[label] = Group
    return Groups, Labels

def Display_Group(params, Group, label, file_path = '../Images/Subspace/'):
    if params.Projection:
        fig = 

def main():
    params = DisplayParams(
        Projection = True
    )
    
    Groups, Labels = Load_Groups()
    for label in Labels:
        if label != 'WT_NonHL': continue
        print(f'Start Analysing {label}')
        Display_Group(params, Groups[label], label, file_path = '../Images/Subspace/')
        print('\n')
    
if __name__ == "__main__":
    main()