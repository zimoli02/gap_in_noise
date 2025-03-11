import os
from PIL import Image
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fs = 10
custom_params = {
    "font.size": fs,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
}
sns.set_theme(style="ticks", rc=custom_params)

import Function.data as data
import Function.analysis as analysis 
import Function.subspace_analysis as subspace_analysis
import Function.plot as plot

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.modules['data'] = sys.modules['Function.data']
sys.modules['analysis'] = sys.modules['Function.analysis']
sys.modules['subspace_analysis'] = sys.modules['Function.subspace_analysis']

basepath = '/Volumes/Zimo/Auditory/Data/'
gpfapath = '/Volumes/Research/GapInNoise/Data/GPFA/'
grouppath = '/Volumes/Research/GapInNoise/Data/Groups/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'
modelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel/'
newmodelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel_ss/'
subspacepath = '/Volumes/Research/GapInNoise/Data/SubspaceAnalysis/'

fig_dpi = 450

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

@dataclass
class DisplayParams:
    Standard_Subspace_Comparison: bool = False
    Standard_Subspace_Location: bool = False
    Subspace_Comparison_per_Gap: bool = False

def Display_Subspace_Comparison_Simulation(n_observation = 100, n_feature = 50, file_path = '../Images/'):
    sub_file_path = check_path(file_path + 'Simulation/')
    fig_Result = subspace_analysis.Subspace_Comparison_Simulation(n_observation = n_observation, n_feature = n_feature)
    for i, method in enumerate(['Pairwise', 'CCA', 'RV', 'Trace']):
        fig_Result[i].savefig(sub_file_path +  method + '.png', dpi = fig_dpi)
    print('Display_Subspace_Comparison_Simulation Completed!')

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

def Display_Group(params, Group, Label, file_path = '../Images/'):
    if params.Standard_Subspace_Location:
        for subspace_name in ['On', 'Off', 'SustainedNoise', 'SustainedSilence']:
            sub_file_path = check_path(file_path + '/StandardSubspace/' + subspace_name + '/')
            fig_Period_Location = subspace_analysis.Draw_Standard_Subspace_Location(Group, subspace_name, period_length = 100, offset_delay = 10)
            fig_Period_Location.savefig(sub_file_path + 'Period_Location.png', dpi = fig_dpi, bbox_inches='tight')
        print('Standard_Subspace_Location Completed!')
        plt.close('all')
    
    if params.Standard_Subspace_Comparison:
        for subspace_name in ['On', 'Off', 'SustainedNoise', 'SustainedSilence']:
            sub_file_path = check_path(file_path + '/StandardSubspace/' + subspace_name + '/' + Label + '/')
            fig_Result = subspace_analysis.Standard_Subspace_Comparison(Group, subspace_name, period_length = 100, offset_delay = 10)
            for i, method in enumerate(['Pairwise', 'CCA', 'RV', 'Trace']):
                fig_Result[i].savefig(sub_file_path +  method + '.png', dpi = fig_dpi, bbox_inches='tight')
        print('Standard_Subspace_Comparison Completed!')
        plt.close('all')

    if params.Subspace_Comparison_per_Gap:
        methods = ['Pairwise', 'CCA', 'RV', 'Trace']
        for subspace_name in ['On', 'Off', 'SustainedNoise', 'SustainedSilence']:
            sub_file_path = check_path(file_path + '/SubspaceEvolution/' + subspace_name + '/' + Label + '/')
            fig = subspace_analysis.Subspace_Similarity_for_All_Gaps(Group, subspace_name, methods, standard_period_length=100, period_length=100, offset_delay = 10)
            fig.savefig(sub_file_path  + 'Summary.png', dpi = fig_dpi, bbox_inches='tight')
        plt.close('all')
            
    

def main():
    
    #Display_Subspace_Comparison_Simulation(n_observation = 100, n_feature = 50, file_path = '../Images/Subspace/')
    
    params = DisplayParams(
        Standard_Subspace_Location = False,
        Standard_Subspace_Comparison = False,
        Subspace_Comparison_per_Gap = True
    )
    
    Groups, Labels = Load_Groups()
    for label in Labels:
        if label != 'WT_NonHL': continue
        print(f'Start Analysing {label}')
        Display_Group(params, Groups[label], label, file_path = '../Images/Subspace/')
        print('\n')
        
    
    '''Display_Group_Summary(unit_type = True, 
                          pc_corre = False,
                          pca_variance = False, 
                          distance = False, 
                          angle = False, 
                          travel = False, 
                          file_path = '../Images/AllGroup/')'''
    

if __name__ == "__main__":
    main()