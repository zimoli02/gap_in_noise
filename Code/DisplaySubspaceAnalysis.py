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
import Function.plot as plot


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

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

@dataclass
class DisplayParams:
    Subspace_Comparison_Simulation: bool = False
    Standard_Subspace_Comparison: bool = False
    Standard_Subspace_Location: bool = False
    Subspace_Comparison_per_Gap: bool = False
    Subspace_Capacity_Determination: bool = False
    Best_Subspace_Comparison: bool = False
    Best_Subspace_Comparison_All_Group_Property: bool = False

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

def Display_Subspace_Comparison_Simulation(n_observation = 100, n_feature = 50, file_path = '../Images/'):
    sub_file_path = check_path(file_path + 'Simulation/')
    fig_Result = subspace_analysis.Subspace_Comparison_Simulation(n_observation = n_observation, n_feature = n_feature)
    for i, method in enumerate(['Pairwise', 'CCA', 'RV', 'Trace']):
        SaveFig(fig_Result[i], sub_file_path +  method)
    print('Display_Subspace_Comparison_Simulation Completed!')

def Display_Group(params, Group, Label, file_path = '../Images/'):
    if params.Standard_Subspace_Location:
        for subspace_name in ['On', 'Off', 'SustainedNoise', 'SustainedSilence']:
            sub_file_path = check_path(file_path + '/StandardSubspace/' + subspace_name + '/')
            fig_Period_Location = subspace_analysis.Draw_Standard_Subspace_Location(Group, subspace_name, period_length = 50, offset_delay = 10)
            SaveFig(fig_Period_Location, sub_file_path + 'Period_Location')
        print('Standard_Subspace_Location Completed!')
        plt.close('all')
    
    if params.Standard_Subspace_Comparison:
        for subspace_name in ['On', 'Off', 'SustainedNoise', 'SustainedSilence']:
            sub_file_path = check_path(file_path + '/StandardSubspace/' + subspace_name + '/' + Label + '/')
            fig_Result = subspace_analysis.Standard_Subspace_Comparison(Group, subspace_name, period_length = 50, offset_delay = 10)
            for i, method in enumerate(['Pairwise', 'CCA', 'RV', 'Trace']):
                SaveFig(fig_Result[i], sub_file_path +  method)
        print('Standard_Subspace_Comparison Completed!')
        plt.close('all')

    if params.Subspace_Comparison_per_Gap:
        #methods = ['Pairwise', 'CCA', 'RV', 'Trace']
        methods = ['Trace']
        for subspace_name in ['On', 'Off', 'SustainedNoise', 'SustainedSilence']:
            sub_file_path = check_path(file_path + '/SubspaceEvolution/' + subspace_name + '/' + Label + '/')
            fig = subspace_analysis.Subspace_Similarity_for_All_Gaps(Group, subspace_name, methods, standard_period_length=50, period_length=50, offset_delay = 10)
            SaveFig(fig, sub_file_path +  'Summary')
        print('Subspace_Comparison_per_Gap Completed!')
        plt.close('all')
    
    if params.Subspace_Capacity_Determination:
        methods = ['Trace']
        for method in methods:
            sub_file_path = check_path(file_path + '/BestSubspace/' + method + '/' + Label + '/')
            fig_best_capacity = subspace_analysis.Period_Capacity_in_Subspace_Comparison(Group, method, max_on_capacity = 75, max_off_capacity = 100, max_timewindow = 100, offset_delay = 10) 
            SaveFig(fig_best_capacity, sub_file_path + 'Best_Capacity')
        print('Subspace_Capacity_Determination Completed!')
        plt.close('all')
        
    if params.Best_Subspace_Comparison:
        methods = ['Trace']
        for method in methods:
            sub_file_path = check_path(file_path + '/BestSubspace/' + method + '/' + Label + '/')
            fig_best_subspace_comparison, fig_explain_find_best_subspace, fig_on_properties, fig_off_properties = subspace_analysis.Best_Subspace_Comparison(Group, method) 
            
            SaveFig(fig_best_subspace_comparison, sub_file_path + 'Best_Subspace_Comparison')
            SaveFig(fig_explain_find_best_subspace, sub_file_path + 'Explain_Find_Best_Subspace')
            SaveFig(fig_off_properties[0], sub_file_path + 'Best_Off_Similarity')
            SaveFig(fig_off_properties[1], sub_file_path + 'Best_Off_Similarity_Peak')
            SaveFig(fig_on_properties[0], sub_file_path + 'Best_On_Similarity')
            SaveFig(fig_on_properties[1], sub_file_path + 'Best_On_Similarity_Peak')
        print('Best_Subspace_Comparison Completed!')
        plt.close('all')
            
            
    

def main():

    params = DisplayParams(
        Subspace_Comparison_Simulation = False,
        Standard_Subspace_Location = False,
        Standard_Subspace_Comparison = False,
        Subspace_Comparison_per_Gap = False,
        Subspace_Capacity_Determination = False,
        Best_Subspace_Comparison = False,
        Best_Subspace_Comparison_All_Group_Property = False
    )
    
    Groups, Labels = Load_Groups()
    for label in Labels:
        if label != 'WT_NonHL': continue
        print(f'Start Analysing {label}')
        Display_Group(params, Groups[label], label, file_path = '../Images/Subspace/')
        print('\n')


    if params.Subspace_Comparison_Simulation:
        Display_Subspace_Comparison_Simulation(n_observation = 100, n_feature = 50, file_path = '../Images/Subspace/')
        
    if params.Best_Subspace_Comparison_All_Group_Property:
        for method in ['Trace', 'RV']:
            fig1, fig2 = subspace_analysis.Best_Subspace_Comparison_All_Group_Property(Groups, method)
            SaveFig(fig1, '../Images/Subspace/BestSubspace/' + method + '/OnSummary')
            SaveFig(fig2, '../Images/Subspace/BestSubspace/' + method + '/OffSummary')

if __name__ == "__main__":
    main()