import os
from PIL import Image
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[0] / 'Function'
sys.path.insert(0, str(functions_dir))
import neuron
import result

gpfapath = '/Volumes/Zimo/Auditory/GPFA/'
grouppath = '/Volumes/Zimo/Auditory/Groups/'
recordingpath = '/Volumes/Zimo/Auditory/Recordings/'
basepath = '/Volumes/Zimo/Auditory/Data/'
mouse = pd.read_csv('Mouse_Tones.csv')

def create_gif(input_folder, output_file, duration=500):
    images = []

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        img = Image.open(file_path)
        images.append(img)
    
    images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=0)

def Display_Group_Summary(unit_type, pc_corre, pca_variance, distance, angle, travel, file_path):
    Groups = []
    for geno_type in ['WT', 'Df1']:
        for hearing_type in ['HL', 'NonHL']:
            Group = neuron.Group(geno_type, hearing_type)
            print(geno_type, hearing_type)
            Groups.append(Group)
            
    GroupSummary = result.Summary(Groups)
    
    if unit_type:
        GroupSummary.Plot_Unit_Type().savefig(file_path + 'UnitTypeSummary.png')
        
    if pc_corre:
        GroupSummary.Plot_Components_Correlation().savefig(file_path + 'ComponentCorreSummary.png')
    
    if pca_variance:
        GroupSummary.Plot_PCA_Variance().savefig(file_path + 'VarianceSummary.png')
    
    if distance:
        GroupSummary.Plot_Distance_Evoke_Ratio(PC = [0,1,3]).savefig(file_path + 'DistanceSummary.png')
    
    if angle:
        GroupSummary.Plot_Angle(PC = [0,1,3]).savefig(file_path + 'AngleSummary.png')
    
    if travel:
        GroupSummary.Plot_Travel_Distance_First_Step(PC = [0,1,3]).savefig(file_path + 'TravelSummary.png')
        
def Display_Group(response_per_gap, pc_corre, pca_score, trajectory_3d, travel_degree, trajectory_3d_by_event, trajectory_event, distance, angle, principal_angle, onoff, on_gap_dependent, return_background, model, file_path):
    Groups = []
    for geno_type in ['WT', 'Df1']:
        for hearing_type in ['HL', 'NonHL']:
            file_path_sub = file_path + geno_type + '_' + hearing_type + '/'

            with open(grouppath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                Group = pickle.load(file)
            Groups.append(Group)
            print(Group.geno_type, Group.hearing_type)
            
            '''offset_pca= neuron.PCA(Group.pop_response_stand[:,0, 450:550], multiple_gaps=False)
            Group.pca.loading = offset_pca.loading
            Group.pca.score = offset_pca.loading @ (Group.pca.data.reshape(Group.pca.data.shape[0], -1))
            Group.pca.Separate_Multiple_Gaps()'''

            Plot = result.PCA(Group)
            
            if response_per_gap:
                PC_idx = 0
                for idx in range(10):
                    Plot.Plot_Neural_Data(gap_idx = idx, PC_idx = PC_idx).savefig(file_path_sub + 'Response/' + str(idx) + '.png')
                    
                input_folder = file_path_sub + 'Response'
                output_file = file_path_sub + 'Responses_PC' + str(PC_idx + 1) + '.gif'
                create_gif(input_folder, output_file)
                print(f"GIF created successfully: {output_file}")
            
            if pca_score:
                Plot.Plot_Projection(PC = [0,1,2,3]).savefig(file_path_sub + 'Component/Projection.png')
            
            if pc_corre:
                Plot.Plot_Components_Correlation().savefig(file_path_sub + 'Component/Correlation.png')

            if trajectory_3d:
                Trajectory_3d, Step_Distance_Event_per_Gap = Plot.Plot_Trajectory_3d(PC = [0,1,2])
                Trajectory_3d.savefig(file_path_sub + 'Trajectory/3d.png')
                Step_Distance_Event_per_Gap.savefig(file_path_sub + 'Trajectory/3d_Event_Step_Distance_per_Gap.png')
                
            if trajectory_3d_by_event:
                Plot.Plot_Trajectory_3d_by_Event(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/3d_by_Event.png')
            
            if trajectory_event:
                Trajectory_Event, Step_Distance_Event, Euclidean_Distance_Event = Plot.Plot_Trajectory_3d_Event(PC = [0,1,2])
                Trajectory_Event.savefig(file_path_sub + 'Trajectory/3d_Event.png')
                Step_Distance_Event.savefig(file_path_sub + 'Trajectory/3d_Event_Step_Distance.png')
                Euclidean_Distance_Event.savefig(file_path_sub + 'Trajectory/3d_Event_Euclidean_Distance.png')
                
            if travel_degree:
                Plot.Plot_Step_Degree(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/Step_Degree.png')
                
            if distance:
                Plot.Plot_Distance(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/Distances.png')
            
            if angle:
                Plot.Plot_Angle(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/3d_Angle.png')
    
            if principal_angle:
                Plot.Plot_Principal_Angle().savefig(file_path_sub + 'Subspace/PrincipalAngle.png')
    
            if onoff:
                fig1, fig2, fig3, fig4, fig5 = Plot.Plot_OnOff_Period()
                fig1.savefig(file_path_sub + 'OnOff/Variance.png')
                fig2.savefig(file_path_sub + 'OnOff/Component_Exclude_Noise.png')
                fig3.savefig(file_path_sub + 'OnOff/3d_Exclude_Noise.png')
                fig4.savefig(file_path_sub + 'OnOff/Data_Exclude_Noise.png')
                fig5.savefig(file_path_sub + 'OnOff/Data_Projection_to_All.png')
            
            if on_gap_dependent:
                fig1, fig2, fig3 = Plot.Plot_Gap_Dependent_On_Response()
                fig1.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_PrincipalAngle.png')
                fig2.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_Projection.png', bbox_inches='tight', pad_inches=0.05)
                fig3.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_Similarity.png')
                
            if return_background:
                fig1, fig2 = Plot.Plot_Noise_Return_Silence()
                fig1.savefig(file_path_sub + 'OnOff/Noise_Return_Silence.png')
                fig2.savefig(file_path_sub + 'OnOff/Noise_Return_Silence_Diff_Subspace.png')
                
            if model:
                Model = result.DynamicalSystem(Group, gap_idx = 8)
                Model.opti_start, Model.opti_end = 0, 350 + Model.gap_dur
                Model.Optimize_Params()
                Model.Run()
                fig1, fig2, fig3 = Model.Draw()
                fig1.savefig(file_path_sub+'Model/Trajectory.png')
                fig2.savefig(file_path_sub+'Model/Trajectory_3d.png')
                fig3.savefig(file_path_sub+'Model/Params.png')

def main():
    
    #Display_Single_Recording(file_path = '../Images/SingleMouse/')
    
    Display_Group(response_per_gap = False, 
                  pc_corre = False,
                  pca_score = False, 
                  trajectory_3d = False, 
                  travel_degree=False,
                  trajectory_3d_by_event = False, 
                  trajectory_event = False, 
                  distance = False, 
                  angle = False, 
                  principal_angle = False,
                  onoff = False,
                  on_gap_dependent = False, 
                  return_background = False,
                  model = True,
                  file_path = '../Images/')
    
    '''
    Display_Group_Summary(unit_type = True, 
                          pc_corre = True,
                          pca_variance = True, 
                          distance = False, 
                          angle = False, 
                          travel = False, 
                          file_path = '../Images/AllGroup/')
    '''

if __name__ == "__main__":
    main()