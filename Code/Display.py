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
newmodelpath = '/Volumes/Research/GapInNoise/Data/TrainedModel_ss/'
subspacepath = '/Volumes/Research/GapInNoise/Data/SubspaceAnalysis/'
@dataclass
class DisplayParams:
    response_per_gap: bool = False
    pc_corre: bool = False
    pca_score: bool = False
    trajectory_3d: bool = False
    travel_degree: bool = False
    trajectory_3d_by_event: bool = False
    trajectory_event: bool = False
    distance: bool = False
    angle: bool = False
    principal_angle: bool = False
    onoff: bool = False
    on_gap_dependent: bool = False
    subspace: bool = False
    decoder: bool = False
    model: bool = False

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
        
def Display_Group(params: DisplayParams, file_path):
    for geno_type in ['WT', 'Df1']: 
        for hearing_type in ['NonHL', 'HL']:
            file_path_sub = file_path + geno_type + '_' + hearing_type + '/'

            with open(grouppath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                Group = pickle.load(file)
            print(Group.geno_type, Group.hearing_type)
            
            Plot_Neural_Data = plot.NeuralData(Group)
            Plot_Latent = plot.Latent(Group)
            Plot_Decoder = plot.Decoder(Group)
            
            if params.response_per_gap:
                # Plot the heatmap of neural activities in each example trial (averaged across 45 trials)
                PC_idx = 0
                for idx in range(10):
                    Plot_Neural_Data.Plot_Heatmap(gap_idx = idx, sort = ['PC', PC_idx]).savefig(file_path_sub + 'Response/' + str(idx) + '.png') 
                input_folder = file_path_sub + 'Response'
                output_file = file_path_sub + 'Responses_PC' + str(PC_idx + 1) + '.gif'
                create_gif(input_folder, output_file)

            if params.pca_score:
                # Plot the projection of data on the first 4 principal components
                Plot_Latent.Plot_Projection(PC = [0,1,2,3]).savefig(file_path_sub + 'Component/Projection.png')
            
            if params.pc_corre:
                # Plot the correlation between loadings of principal components with on/off-response z-scores
                Plot_Latent.Plot_Components_Correlation().savefig(file_path_sub + 'Component/Correlation.png')

            if params.trajectory_3d:
                fig1, fig2 = Plot_Latent.Plot_Trajectory_3d(PC = [0,1,2])
                fig1.savefig(file_path_sub + 'Trajectory/3d.png')
                fig2.savefig(file_path_sub + 'Trajectory/3d_Event_Step_Distance_per_Gap.png')
                
            if params.trajectory_3d_by_event:
                Plot_Latent.Plot_Trajectory_3d_by_Event(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/3d_by_Event.png')
            
            if params.trajectory_event:
                fig1, fig2, fig3 = Plot_Latent.Plot_Trajectory_3d_Event(PC = [0,1,2])
                fig1.savefig(file_path_sub + 'Trajectory/3d_Event.png')
                fig2.savefig(file_path_sub + 'Trajectory/3d_Event_Step_Distance.png')
                fig3.savefig(file_path_sub + 'Trajectory/3d_Event_Euclidean_Distance.png')
                
            if params.travel_degree:
                Plot_Latent.Plot_Step_Degree(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/Step_Degree.png')
                
            if params.distance:
                Plot_Latent.Plot_Distance(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/Distances.png')
            
            if params.angle:
                Plot_Latent.Plot_Angle(PC = [0,1,2]).savefig(file_path_sub + 'Trajectory/3d_Angle.png')
    
            if params.principal_angle:
                # Compare how similar two subspaces are by taking the average principal angles
                fig1= Plot_Latent.Plot_Principal_Angle(dim=5)
                fig1.savefig(file_path_sub + 'Subspace/PrincipalAngle.png')
                #fig2.savefig(file_path_sub + 'Subspace/PrincipalAngle_LeaveOneOut.png')
                #fig3.savefig(file_path_sub + 'Subspace/PrincipalAngle_LeaveOneOut_onoff.png')
    
            if params.onoff:
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
            
            if params.on_gap_dependent:
                fig1, fig2, fig3 = Plot_Latent.Plot_Gap_Dependent_On_Response()
                # Compare the similarity between the period-specific subspaces, with or without projection
                fig1.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_PrincipalAngle.png')
                # Compare N1 and N2 onset subspaces for 3 gaps: angles and projections (might be most intuitive)
                fig2.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_Projection.png', bbox_inches='tight', pad_inches=0.05)
                # Compare the similarity between the N1 and N2 onset subspaces for all gaps
                fig3.savefig(file_path_sub + 'OnOff/Gap_Dependent_OnResp_Similarity.png')
            
            if params.subspace:
                
                try:
                    with open(subspacepath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                        Subspace = pickle.load(file)
                except FileNotFoundError:
                    Subspace = analysis.Subspace(Group)
                    Subspace.Fit_Gap_Prediction_Model()
                    Subspace.Compare_Period_Length()
                    Subspace.Get_Prediction_along_Trial()
                    
                    with open(subspacepath + geno_type + '_' + hearing_type + '.pickle', 'wb') as file:
                        pickle.dump(Subspace, file)
                
                Plot_Subspace = plot.PlotSubspace(Group, Subspace)
                fig_3D, fig_2D = Plot_Subspace.Draw_Similarity_Index()
                fig_scatter, fig_shuffle_r2 = Plot_Subspace.Draw_Model_Prediction()
                fig_different_r2 = Plot_Subspace.Draw_R2_with_Different_Period()
                fig_feature, fig_prediction = Plot_Subspace.Draw_Prediction_Along_Trial()
                
                fig_3D.savefig(file_path_sub + 'Subspace/Similarity_Index_3D.png')
                fig_2D.savefig(file_path_sub + 'Subspace/Similarity_Index_2D.png')
                fig_scatter.savefig(file_path_sub + 'Subspace/Model_Prediction_Scatter.png')
                fig_shuffle_r2.savefig(file_path_sub + 'Subspace/Model_Prediction_Shuffle_R2.png')
                fig_different_r2.savefig(file_path_sub + 'Subspace/R2_with_Different_Period.png')
                fig_feature.savefig(file_path_sub + 'Subspace/Prediction_Along_Trial_Feature.png')
                fig_prediction.savefig(file_path_sub + 'Subspace/Prediction_Along_Trial.png')
            
            if params.decoder:
                fig1, fig2 = Plot_Decoder.Plot_Noise_Return_Silence()
                fig1.savefig(file_path_sub + 'OnOff/Noise_Return_Silence.png')
                fig2.savefig(file_path_sub + 'OnOff/Noise_Return_Silence_Diff_Subspace.png')
                
                fig = Plot_Decoder.Plot_Binary_Decoder()
                fig.savefig(file_path_sub + 'Decoder/Binary.png')
                
                fig = Plot_Decoder.Plot_HMM_Decoder()
                fig.savefig(file_path_sub + 'Decoder/HMM.png')
                
            if params.model:
                Update = False 
                
                if Update:
                    Model = analysis.Model(Group)
                    Model.Train(cross_validate = True)
                    with open(modelpath + Group.geno_type + '_' + Group.hearing_type + '.pickle', 'wb') as file:
                        pickle.dump(Model, file)
                else:
                    with open(modelpath + geno_type + '_' + hearing_type + '.pickle', 'rb') as file:
                        Model = pickle.load(file)
                    with open(newmodelpath + geno_type + '_' + hearing_type + '.pkl', 'rb') as file:
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
    
    params = DisplayParams(
        subspace = True
    )
    
    Display_Group(params, file_path = '../Images/')
    
    
    '''Display_Group_Summary(unit_type = True, 
                          pc_corre = False,
                          pca_variance = False, 
                          distance = False, 
                          angle = False, 
                          travel = False, 
                          file_path = '../Images/AllGroup/')'''
    

if __name__ == "__main__":
    main()