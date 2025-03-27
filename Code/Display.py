from dataclasses import dataclass

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
import DisplaySubspaceAnalysis
import DisplayProjectionAnalysis

import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially transparent artists will be rendered opaque")

@dataclass
class SubspaceParams:
    Subspace_Comparison_Simulation: bool = False
    Standard_Subspace_Comparison: bool = False
    Standard_Subspace_Location: bool = False
    Subspace_Comparison_per_Gap: bool = False
    Subspace_Capacity_Determination: bool = False
    Best_Subspace_Comparison: bool = False
    Best_Subspace_Comparison_All_Group_Property: bool = False
@dataclass
class ProjectionParams:
    Dimensionality_Reduction: bool = False
    Low_Dim_Activity_in_Space: bool = False
    Predict_Off_using_Low_Dim_Activity: bool = False
    
def main():

    subspace_params = SubspaceParams(
        Subspace_Capacity_Determination = True
    )
    DisplaySubspaceAnalysis.main(subspace_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])
    
    projection_params = ProjectionParams(
        Predict_Off_using_Low_Dim_Activity = True
    )
    #DisplayProjectionAnalysis.main(projection_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])
    

if __name__ == "__main__":
    main()