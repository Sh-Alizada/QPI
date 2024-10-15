import os
import pandas as pd
import numpy as np
import utils_v2 as utils
from my_decorators import timer
import matplotlib.pyplot as plt


@timer
def overall_mass_calculator(exp_folder_path, medfilt_size=5, data_folder='data_folder', 
                            file_pattern='frame_data_loc_(?P<location>\d+)', location_range=None, use_frame_nums=False):
    
    folder_path = os.path.join(exp_folder_path, data_folder)
    
    # Get the files
    all_files_df = utils.find_files(folder_path, pattern=file_pattern, location=location_range)

    # Initialize a list to store all aggregated DataFrames
    aggregated_data_list = []
    
    # Loop through the files
    for file in all_files_df['filename']:
        
        file_path = os.path.join(folder_path, file)
        
        # Read data from file
        data = pd.read_csv(file_path)
        
        data['overall_mass'] = utils.medfilt1_truncate(data['pixel_sums'], medfilt_size)
        data['overall_mass'] = data['overall_mass']/data['overall_mass'][0]
        
        columns_to_aggregate = ['frame', 'time(h)', 'overall_mass']
        non_aggregated_columns = ['location', 'well', 'cell_type', 'treatment', 'concentration', 'conc_units', 'control']
 
        # Create a new DataFrame with non-aggregated columns first and aggregated columns next
        aggregated_data = pd.DataFrame(
            {**{col: data[col].iloc[0] for col in non_aggregated_columns},  # First, include non-aggregated columns
             **{col: [data[col].tolist()] for col in columns_to_aggregate}}  # Then aggregate the selected columns
        )
        
        # Append the aggregated data to the list
        aggregated_data_list.append(aggregated_data)
    
    # Concatenate all aggregated DataFrames
    final_aggregated_data = pd.concat(aggregated_data_list, ignore_index=True)
    
    # Group by 'well' and aggregate the pixel_sums and other list-like columns by taking the mean and std deviation
    well_data_df = final_aggregated_data.groupby('well').agg({
        'location': 'first',  # Keep first location or use another rule if needed
        'cell_type': 'first',
        'treatment': 'first',
        'concentration': 'first',
        'conc_units': 'first',
        'control': 'first',
        'frame': lambda x: list(map(np.mean, zip(*x))),  # Mean over frames
        'time(h)': lambda x: list(map(np.mean, zip(*x))),  # Mean over time points
        'overall_mass': [lambda x: list(map(np.mean, zip(*x))), lambda x: list(map(np.std, zip(*x)))]  # Mean and std over pixel sums
    }).reset_index()

    # Rename the columns
    well_data_df.columns = ['well', 'location', 'cell_type', 'treatment', 'concentration', 'conc_units', 'control', 
                                 'frame_mean', 'time_mean', 'overall_mass', 'overall_mass_sd']

    # Now group by 'treatment' (drug) and 'concentration', and aggregate the same way
    conc_data_df = well_data_df.groupby(['treatment', 'concentration']).agg({
        'conc_units': 'first',
        'cell_type': 'first',
        'control': 'first',
        'frame_mean': lambda x: list(map(np.mean, zip(*x))),  # Mean over frames
        'time_mean': lambda x: list(map(np.mean, zip(*x))),  # Mean over time points
        'overall_mass': [lambda x: list(map(np.mean, zip(*x))), lambda x: list(map(np.std, zip(*x)))]  # Mean and std over pixel sums
    }).reset_index()
    
    conc_data_df.columns = ['treatment', 'concentration', 'conc_units', 'cell_type', 'control', 
                                 'frame_mean', 'time_mean', 'overall_mass', 'overall_mass_sd']
    
    return well_data_df, conc_data_df


def plot_ov_mass(conc_data_df):
    """
    Plot each drug's concentrations in separate figures, and add 'control'=='Yes' data to all figures.
    """
    # Separate control data (control == 'Yes')
    control_data = conc_data_df[conc_data_df['control'] == 'Yes']
    
    # Get unique drug treatments
    drugs = conc_data_df[conc_data_df['control'] == 'No']['treatment'].unique()

    # Loop through each drug and plot concentrations
    for drug in drugs:
        # Filter for the current drug (excluding controls)
        drug_data = conc_data_df[(conc_data_df['treatment'] == drug) & (conc_data_df['control'] == 'No')]
        
        # Create a new figure for the current drug
        plt.figure(figsize=(10, 6))
        
        # Plot each concentration for the drug
        for _, row in drug_data.iterrows():
            time_points = row['time_mean']
            mean_mass = row['overall_mass']
            std_mass = row['overall_mass_sd']
            
            # Plot the mean with shaded standard deviation
            plt.plot(time_points, mean_mass, label=f"{row['concentration']} {row['conc_units']}")
            plt.fill_between(time_points, np.array(mean_mass) - np.array(std_mass), 
                             np.array(mean_mass) + np.array(std_mass), alpha=0.2)

        # Add the control data to the plot
        for _, row in control_data.iterrows():
            time_points = row['time_mean']
            mean_mass = row['overall_mass']
            std_mass = row['overall_mass_sd']
            
            # Plot the control mean with shaded standard deviation
            plt.plot(time_points, mean_mass, label=f"Control ({row['concentration']} {row['conc_units']})", linestyle='--')
            plt.fill_between(time_points, np.array(mean_mass) - np.array(std_mass), 
                             np.array(mean_mass) + np.array(std_mass), alpha=0.2)

        # Final plot adjustments
        plt.xlabel('Time (h)')
        plt.ylabel('Mean Overall Mass')
        plt.title(f'Mean Overall Mass Over Time for {drug} and Control')
        plt.legend(title='Concentration')
        plt.grid(True)
        
        # Show the plot for the current drug
        plt.show()
