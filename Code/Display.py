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
import Function.subspace_analysis as subspace_analysis
import DisplaySubspaceAnalysis as display_subspace_analysis



basepath = '/Volumes/Zimo/Auditory/Data/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'
grouppath = '/Volumes/Research/GapInNoise/Data/Groups/'
gpfapath = '/Volumes/Research/GapInNoise/Data/GPFA/'
modelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel/'
newmodelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel_ss/'
subspacepath = '/Volumes/Research/GapInNoise/Data/Subspace/'

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

@dataclass
class SubspaceParams:
    Subspace_Comparison_Simulation: bool = False
    Standard_Subspace_Comparison: bool = False
    Standard_Subspace_Location: bool = False
    Subspace_Comparison_per_Gap: bool = False
    Subspace_Capacity_Determination: bool = False
    Best_Subspace_Comparison: bool = False
    Best_Subspace_Comparison_All_Group_Property: bool = False
    
def main():

    subspace_params = SubspaceParams(
        Subspace_Comparison_Simulation = False,
        Standard_Subspace_Location = False,
        Standard_Subspace_Comparison = False,
        Subspace_Comparison_per_Gap = False,
        Subspace_Capacity_Determination = True,
        Best_Subspace_Comparison = True,
        Best_Subspace_Comparison_All_Group_Property = False
    )
    
    display_subspace_analysis(subspace_params, group_labels = ['WT_NonHL', 'Df1_NonHL'])
    
    

if __name__ == "__main__":
    main()