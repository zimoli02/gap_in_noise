from dataclasses import dataclass

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
import DisplaySubspaceAnalysis
import DisplayProjectionAnalysis
import DisplayUnitAnalysis

import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially transparent artists will be rendered opaque")

@dataclass
class UnitParams:
    Example_Unit_Responsiveness: bool = False
    All_Unit_Response_Type: bool = False
    All_Unit_Spike_Type: bool = False
    Responsiveness_Comparison: bool = False
 
@dataclass
class ProjectionParams:
    Dimensionality_Reduction: bool = False
    Low_Dim_Activity_in_Space: bool = False
    Predict_Off_using_Low_Dim_Activity: bool = False   

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



    projection_params = ProjectionParams(
        Predict_Off_using_Low_Dim_Activity = True
    )
    #DisplayProjectionAnalysis.main(projection_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])

    subspace_params = SubspaceParams(
        Best_Subspace_Comparison_All_Group_Property = True
    )
    #DisplaySubspaceAnalysis.main(subspace_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])

    unit_params = UnitParams(
        Example_Unit_Responsiveness = True,
        All_Unit_Response_Type = True,
        All_Unit_Spike_Type = True,
        Responsiveness_Comparison = True
    )
    DisplayUnitAnalysis.main(unit_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])

if __name__ == "__main__":
    main()