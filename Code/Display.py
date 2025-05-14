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
    Single_Units: bool = False
    All_Unit_Response_Type: bool = False
    All_Unit_Spike_Type: bool = False
    Responsiveness_Comparison: bool = False
 
@dataclass
class ProjectionParams:
    Period_Selection_Explanation: bool = False
    Analysis_Explanation: bool = False
    Dimensionality_Reduction: bool = False
    Low_Dim_Activity_by_Space: bool = False
    Low_Dim_Activity_Divergence_by_Space: bool = False
    Low_Dim_Activity_in_Different_Space: bool = False
    Low_Dim_Activity_in_Space: bool = False
    Predict_Off_using_Low_Dim_Activity: bool = False   

@dataclass
class SubspaceParams:
    Subspace_Comparison_Simulation: bool = False
    Standard_Subspace_Comparison: bool = False
    Standard_Subspace_Location: bool = False
    Subspace_Comparison_per_Gap: bool = False
    Analysis_Comparison: bool = False
    Subspace_Capacity_Determination: bool = False
    Best_Subspace_Comparison: bool = False
    Subspace_Comparison_All_Group_Property: bool = False
    Best_Subspace_Comparison_All_Group_Property: bool = False

def main():

    projection_params = ProjectionParams(
        Dimensionality_Reduction = True,
        Low_Dim_Activity_by_Space = True,
        Low_Dim_Activity_Divergence_by_Space = True,
        Low_Dim_Activity_in_Different_Space = True

    )
    #DisplayProjectionAnalysis.main(projection_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])
    #DisplayProjectionAnalysis.main(projection_params, group_labels = ['WT_NonHL'])

    subspace_params = SubspaceParams(
        Analysis_Comparison = True
    )
    #DisplaySubspaceAnalysis.main(subspace_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])
    DisplaySubspaceAnalysis.main(subspace_params, group_labels = ['WT_NonHL'])

    unit_params = UnitParams(
        Single_Units=True
    )
    #DisplayUnitAnalysis.main(unit_params, group_labels = ['WT_NonHL', 'WT_HL', 'Df1_NonHL', 'Df1_HL'])

if __name__ == "__main__":
    main()