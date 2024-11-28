import os
from PIL import Image
import pickle

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
import Function.plot as plot

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.modules['data'] = sys.modules['Function.data']
sys.modules['analysis'] = sys.modules['Function.analysis']

basepath = '/Volumes/Zimo/Auditory/Data/'
gpfapath = '/Volumes/Research/GapInNoise/Data/GPFA/'
grouppath = '/Volumes/Research/GapInNoise/Data/Groups/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'
modelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel/'


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
        for hearing_type in ['NonHL', 'HL']:
            with open(grouppath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                Group = pickle.load(file)
            print(Group.geno_type, Group.hearing_type)
            Groups.append(Group)
            
    GroupSummary = plot.Summary(Groups)
    
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
        
def Display_Group(response_per_gap, pc_corre, pca_score, trajectory_3d, travel_degree, trajectory_3d_by_event, trajectory_event, distance, angle, principal_angle, onoff, on_gap_dependent, decoder, model, file_path):
    for geno_type in ['WT', 'Df1']:
        for hearing_type in ['NonHL', 'HL']:
            file_path_sub = file_path + geno_type + '_' + hearing_type + '/'

            with open(grouppath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                Group = pickle.load(file)
            print(Group.geno_type, Group.hearing_type)
            
            Plot_Neural_Data = plot.NeuralData(Group)
            Plot_Latent = plot.Latent(Group)
            Plot_Decoder = plot.Decoder(Group)
            
            if response_per_gap:
                # Plot the heatmap of neural activities in each example trial (averaged across 45 trials)
                PC_idx = 0
                for idx in range(10):
                    Plot_Neural_Data.Plot_Heatmap(gap_idx = idx, sort = ['PC', PC_idx]).savefig(file_path_sub + 'Response/' + str(idx) + '.png') 
                input_folder = file_path_sub + 'Response'
                output_file = file_path_sub + 'Responses_PC' + str(PC_idx + 1) + '.gif'
                create_gif(input_folder, output_file)

            if pca_score:
                # Plot the projection of data on the first 4 principal components
                Plot_Latent.Plot_Projection(PC = [0,1,2,3]).savefig(file_path_sub + 'Component/Projection.png')
            
            if pc_corre:
                # Plot the correlation between loadings of principal components with on/off-response z-scores
                Plot_Latent.Plot_Components_Correlation().savefig(file_path_sub + 'Component/Correlation.png')

            if trajectory_3d:
                fig1, fig2 = Plot_Latent.Plot_Trajectory_3d(PC = [0,1,2])
                fig1.savefig(file_path_sub + 'Trajectory/3d.png')
                fig2.savefig(file_path_sub + 'Trajectory/3d_Event_Step_Distance_per_Gap.png')
                
            if trajectory_3d_by_event:
                Plot_Latent.Plot_Trajectory_3d_by_Event(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/3d_by_Event.png')
            
            if trajectory_event:
                fig1, fig2, fig3 = Plot_Latent.Plot_Trajectory_3d_Event(PC = [0,1,2])
                fig1.savefig(file_path_sub + 'Trajectory/3d_Event.png')
                fig2.savefig(file_path_sub + 'Trajectory/3d_Event_Step_Distance.png')
                fig3.savefig(file_path_sub + 'Trajectory/3d_Event_Euclidean_Distance.png')
                
            if travel_degree:
                Plot_Latent.Plot_Step_Degree(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/Step_Degree.png')
                
            if distance:
                Plot_Latent.Plot_Distance(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/Distances.png')
            
            if angle:
                Plot_Latent.Plot_Angle(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/3d_Angle.png')
    
            if principal_angle:
                # Compare how similar two subspaces are by taking the average principal angles
                Plot_Latent.Plot_Principal_Angle(dim=5).savefig(file_path_sub + 'Subspace/PrincipalAngle.png')
    
            if onoff:
                fig1, fig2, fig3, fig4, fig5 = Plot_Latent.Plot_OnOff_Period()
                # Draw variance explained from the four subspaces: [on, off] x [original, exclude sustanined response]
                fig1.savefig(file_path_sub + 'OnOff/Variance.png')
                # Draw PC1-3 from the four subspaces: [on, off] x [original, exclude sustanined response]
                fig2.savefig(file_path_sub + 'OnOff/Component_Exclude_Noise.png')
                # Draw PC1-3 (3d) from the four subspaces: [on, off] x [original, exclude sustanined response]
                fig3.savefig(file_path_sub + 'OnOff/3d_Exclude_Noise.png')
                # Draw heatmap of data, projection, data exclude projection 
                fig4.savefig(file_path_sub + 'OnOff/Data_Exclude_Noise.png')
                # Draw PC1-N heatmap of data projected to the period-specific subspace 
                fig5.savefig(file_path_sub + 'OnOff/Data_Projection_to_All.png')
            
            if on_gap_dependent:
                fig1, fig2, fig3 = Plot_Latent.Plot_Gap_Dependent_On_Response()
                # Compare the similarity between the period-specific subspaces, with or without projection
                fig1.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_PrincipalAngle.png')
                # Compare N1 and N2 onset subspaces for 3 gaps: angles and projections (might be most intuitive)
                fig2.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_Projection.png', bbox_inches='tight', pad_inches=0.05)
                # Compare the similarity between the N1 and N2 onset subspaces for all gaps
                fig3.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_Similarity.png')
            
            
            if decoder:
                fig1, fig2 = Plot_Decoder.Plot_Noise_Return_Silence()
                fig1.savefig(file_path_sub + 'OnOff/Noise_Return_Silence.png')
                fig2.savefig(file_path_sub + 'OnOff/Noise_Return_Silence_Diff_Subspace.png')
                
                fig = Plot_Decoder.Plot_Binary_Decoder()
                fig.savefig(file_path_sub + 'Decoder/Binary.png')
                
                fig = Plot_Decoder.Plot_HMM_Decoder()
                fig.savefig(file_path_sub + 'Decoder/HMM.png')
                
            if model:
                Update = False 
                
                if Update:
                    Model = analysis.Model(Group)
                    Model.Train(cross_validate = True)
                    with open(modelpath + Group.geno_type + '_' + Group.hearing_type + '.pickle', 'wb') as file:
                        pickle.dump(Model, file)
                else:
                    with open(modelpath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                        Model = pickle.load(file)
                    #Model.model.Set_Params_Median()
                    Model.model.Set_Params_of_Least_Loss()
                
                Plot_Model = plot.System(Group, Model)
                fig1, fig2, fig3, fig4, fig5, fig6 = Plot_Model.Draw_Model(gap_idx = 9)
                fig1.savefig(file_path_sub+'Model/Trajectory.png')
                fig2.savefig(file_path_sub+'Model/Trajectory_3d.png')
                fig3.savefig(file_path_sub+'Model/Params.png')
                fig4.savefig(file_path_sub+'Model/Loss_with_Iter.png')
                fig5.savefig(file_path_sub+'Model/Gap_Duration_Recognition.png')
                fig6.savefig(file_path_sub+'Model/Fix_Point.png')
                
                '''fig = Plot_Model.Draw_Gap_Threshold_Simulation()
                fig.savefig(file_path_sub+'Model/Gap_Threshold_Simulation.png')
                
                S_on = 60 
                if Group.hearing_type == 'HL': S_on = 75
                S = np.zeros(2000) + 10
                for t in range(100, 350): S[t] = S_on
                for t in range(500, 750): S[t] = S_on
                for t in range(800, 850): S[t] = S_on
                for t in range(870, 920): S[t] = S_on
                for t in range(940, 990): S[t] = S_on
                for t in range(1250, 1550): S[t] = S_on
                
                fig1, fig2, fig3 = Plot_Model.Draw_Simulation(S)
                fig1.savefig(file_path_sub+'Model/Simulated_Trajectory.png')
                fig2.savefig(file_path_sub+'Model/Simulated_Trajectory_3d.png')
                fig3.savefig(file_path_sub+'Model/Simulated_Gap_Decoding.png')'''
                

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
                  decoder = False,
                  model = True,
                  file_path = '../Images/')
    
    
    '''Display_Group_Summary(unit_type = True, 
                          pc_corre = False,
                          pca_variance = False, 
                          distance = False, 
                          angle = False, 
                          travel = False, 
                          file_path = '../Images/AllGroup/')'''
    

if __name__ == "__main__":
    main()