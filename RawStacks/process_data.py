from inputs import Inputs
import utils_v2 as utils
import os
from sgr_calculator_v5 import sgr_calculator
from ov_mass_calculator import overall_mass_calculator, plot_ov_mass
import sgr_utils_v2

def process_data():
    
    # Create the inputs instance and load necessary data
    inputs = Inputs()
    inputs.load_excel_file()
    
    # Get inputs params
    exp_folder_path = inputs.exp_folder_path
    treatment_conds_df = inputs.treatment_conds_df
    data_folder = inputs.data_folder
    
    # Combine all the location data (saves and returns for futher processing)
    all_cell_data = utils.concat_track_files(os.path.join(exp_folder_path, data_folder))

    # Calculate SGRs
    cell_sgr_df, well_sgr_df, conc_sgr_df = sgr_calculator(all_cell_data, exp_folder_path, data_folder=data_folder)

    # EC50 params
    ec50_params_df = sgr_utils_v2.EC50_params(well_sgr_df, p_cutoff=0.01)

    # Save EC50 figures
    sgr_utils_v2.plot_EC50_curves(conc_sgr_df, ec50_params_df, save_fig=True, exp_folder_path=exp_folder_path, data_folder=data_folder)

    # Save Violins
    sgr_utils_v2.plot_violins(cell_sgr_df, treatment_conds_df=treatment_conds_df, 
                              save_fig=True, exp_folder_path=exp_folder_path, data_folder=data_folder)


    well_data_df, conc_data_df = overall_mass_calculator(exp_folder_path)
    
    plot_ov_mass(conc_data_df)