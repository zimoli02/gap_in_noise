import os
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
import Function.unit_analysis as unit_analysis


basepath = '/Volumes/Zimo/Auditory/Data/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'
grouppath = '/Volumes/Research/GapInNoise/Data/Groups/'
gpfapath = '/Volumes/Research/GapInNoise/Data/GPFA/'
modelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel/'
newmodelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel_ss/'
subspacepath = '/Volumes/Research/GapInNoise/Data/Subspace/'

imagepath = '/Volumes/Research/GapInNoise/Images/UnitType/'

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


def main(unit_params, group_labels):
    Groups, Labels = Load_Groups(group_labels)
    
    if unit_params.Example_Unit_Responsiveness:
        figs = unit_analysis.Draw_Example_Unit()
        types = ['On','Off', 'Both', 'None']
        sub_file_path = check_path(imagepath + 'ExampleUnit/')
        for i in range(4): 
            SaveFig(figs[i], sub_file_path + f'{types[i]}')
        print('Example_Unit_Responsiveness Completed')
        plt.close('all')
    
    if unit_params.All_Unit_Response_Type:
        fig = unit_analysis.Draw_Unit_Response_Type_All_Group(Groups)
        sub_file_path = check_path(imagepath + 'AllGroup/')
        SaveFig(fig, sub_file_path + 'Response')
        print('All_Unit_Response_Type Completed')
        plt.close('all')
        
    if unit_params.All_Unit_Spike_Type:
        fig = unit_analysis.Draw_Unit_Spike_Type_All_Group(Groups)
        sub_file_path = check_path(imagepath + 'AllGroup/')
        SaveFig(fig, sub_file_path + 'Spike')
        print('All_Unit_Spike_Type Completed')
        plt.close('all')
        
    if unit_params.Responsiveness_Comparison:
        fig_On, fig_Off = unit_analysis.Draw_Responsiveness_Comparison(percent=0.5)
        sub_file_path = check_path(imagepath + 'AllGroup/')
        SaveFig(fig_On, sub_file_path + 'Compare_On')
        SaveFig(fig_Off, sub_file_path + 'Compare_Off')
        print('Responsiveness_Comparison Completed')
        plt.close('all')

if __name__ == "__main__":
    main()