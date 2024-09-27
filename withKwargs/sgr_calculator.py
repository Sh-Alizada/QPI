import pandas as pd
import numpy as np
import math
from scipy.special import erfcinv
from filter_tracks import filter_tracks
import utils


def sgr_calculator(df, exp_folder_path, data_folder='data_folder', min_frames=20, min_mass=20, start_frame=1, end_frame=None):

    
    # Get the filtered DataFrame using cellhunt_v3
    filtered_tracks_df = filter_tracks(df, min_frames=min_frames, min_mass=min_mass, start_frame=start_frame, end_frame=end_frame)
    
    
    # Extract unique cell IDs
    cells_ID = filtered_tracks_df['id'].unique()
    num_cells = len(cells_ID)
    
    # Initialize arrays to store results
    cells_mod = [None] * num_cells
    xav = np.zeros(num_cells)
    x2 = np.zeros(num_cells)
    yav = np.zeros(num_cells)
    xy = np.zeros(num_cells)
    b_raw = np.zeros(num_cells)
    a_raw = np.zeros(num_cells)
    r2 = np.zeros(num_cells)
    TSS = np.zeros(num_cells)
    LRSS = np.zeros(num_cells)
    syx = np.zeros(num_cells)
    syxy = np.zeros(num_cells)
    sb = np.zeros(num_cells)
    
    # Group the DataFrame by 'id' for faster operations
    grouped = filtered_tracks_df.groupby('id')
    
    # Cell information
    cells_drug = np.zeros(num_cells)
    cells_conc = np.zeros(num_cells)
    cells_loc = np.zeros(num_cells)
    cells_well = np.zeros(num_cells)
    
    for i, cell_id in enumerate(cells_ID):
        # Get the data for the current cell
        cell_data = grouped.get_group(cell_id)
        
        # Set each cell initial time to zero
        times = cell_data['Time (h)'].values
        masses = cell_data['Mass (pg)'].values
        times_zeroed = times - times[0]
        cell_mod = np.column_stack((times_zeroed, masses))
        cells_mod[i] = cell_mod
        
        # Precompute common values
        xav[i] = xav_i = np.mean(times_zeroed)
        yav[i] = yav_i = np.mean(masses)
        x = times_zeroed - xav_i
        y = masses - yav_i
        
        # Calculate squared x and product xy once
        x2[i] = np.sum(x**2)
        xy[i] = np.sum(x * y)
        
        # Calculate regression parameters
        b_raw[i] = b = xy[i] / x2[i]
        a_raw[i] = a = yav_i - b * xav_i
        TSS[i] = TSS_val = np.sum(y**2)
        LRSS[i] = LRSS_val = xy[i]**2 / x2[i]
        syx[i] = syx_val = np.sqrt((TSS_val - LRSS_val) / (len(cell_mod) - 2))
        syxy[i] = syx_val / yav_i
        sb[i] = np.sqrt(syx_val**2 / x2[i])
        r2[i] = LRSS_val / TSS_val
    
        # Add drug and concentration data
        cells_drug[i] = cell_data['condition_drugID'].iloc[0]
        cells_conc[i] = cell_data['condition_concentration (um)'].iloc[0]
        cells_loc[i] = cell_data['Location ID'].iloc[0]
        cells_well[i] = cell_data['cells_well'].iloc[0]
        
    # Normalize growth rate by y intercept
    b = b_raw / a_raw
    c = -1 / (np.sqrt(2) * erfcinv(3 / 2))
    
    # Create a DataFrame from the results
    regressions_results_df = pd.DataFrame({
        'cell_id': cells_ID,
        'cells_drug': cells_drug,
        'cells_conc': cells_conc,
        'cells_loc': cells_loc,
        'cells_well': cells_well,
        'b': b,
        'xav': xav,
        'x2': x2,
        'yav': yav,
        'xy': xy,
        'b_raw': b_raw,
        'a_raw': a_raw,
        'r2': r2,
        'TSS': TSS,
        'LRSS': LRSS,
        'syx': syx,
        'syxy': syxy,
        'sb': sb,
        'c': [c] * num_cells,  # c is the same for all cells
        'cells_mod': cells_mod
    })
    
    # Remove false cells
    for jj in range(1, int(regressions_results_df['cells_drug'].max()) + 1):
        drug_conc = regressions_results_df.loc[regressions_results_df['cells_drug'] == jj, 'cells_conc'].unique()
        for conc in drug_conc:
            mask = (regressions_results_df['cells_drug'] == jj) & (regressions_results_df['cells_conc'] == conc)
            MADsyx = c * np.median(np.abs(regressions_results_df.loc[mask, 'syx'] - np.median(regressions_results_df.loc[mask, 'syx'])))
            MADb = c * np.median(np.abs(regressions_results_df.loc[mask, 'b'] - np.median(regressions_results_df.loc[mask, 'b'])))
            z = (regressions_results_df['b'] >= np.median(regressions_results_df.loc[mask, 'b']) + 3 * MADb) | (regressions_results_df['b'] <= np.median(regressions_results_df.loc[mask, 'b']) - 3 * MADb)
            y = regressions_results_df['syx'] >= np.median(regressions_results_df.loc[mask, 'syx']) + 3 * MADsyx
            zy = z | y
            remove_mask = mask & zy
            regressions_results_df = regressions_results_df[~remove_mask]
    
    # Additional calculations
    rm = np.array([cell[-3, 1] / cell[2, 1] for cell in regressions_results_df['cells_mod']])
    l2mass = np.array([cell[-3, 1] for cell in regressions_results_df['cells_mod']])
    lastmass = np.array([cell[-1, 1] for cell in regressions_results_df['cells_mod']])
    meanmass = np.array([np.mean(cell[:, 1]) for cell in regressions_results_df['cells_mod']])
    
    regressions_results_df['rm'] = rm
    regressions_results_df['l2mass'] = l2mass
    regressions_results_df['lastmass'] = lastmass
    regressions_results_df['meanmass'] = meanmass
    
    # Calculate the number of cells for each location and well
    loc_cell_amount = regressions_results_df['cells_loc'].value_counts().rename('loc_cell_amount')
    well_cell_amount = regressions_results_df['cells_well'].value_counts().rename('well_cell_amount')
    
    # Add the counts back to the DataFrame
    regressions_results_df = regressions_results_df.join(loc_cell_amount, on='cells_loc')
    regressions_results_df = regressions_results_df.join(well_cell_amount, on='cells_well')
    
    #Store individual cell grwoth rate data
    cell_growth_rate_df = regressions_results_df[['cell_id', 'cells_drug', 'cells_conc', 'cells_loc', 'cells_well', 'b', 'loc_cell_amount', 'well_cell_amount']].copy()
    cell_growth_rate_df.rename(columns={'b': 'cell_sgr'}, inplace=True) # Rename the 'b' column to 'SGR'
    
    
    # Create an array to represent the plate with mean growth rates
    plate = utils.generate_plate(plate_height, plate_length, starting_corner) # Generate the plate layout
    mean_growth_rate_per_well = regressions_results_df.groupby('cells_well')['b'].mean() # Group by well and calculate the mean growth rate 'b' for each well
    growth_rate_plate = np.full((plate_height, plate_length), np.nan)
    for well, mean_b in mean_growth_rate_per_well.items():
        well_position = np.argwhere(plate == well) # Find the row and column in the plate corresponding to the well number
        if well_position.size > 0:
            row, col = well_position[0]
            if well not in wells_to_remove:
                growth_rate_plate[row, col] = mean_b
    
    
    # Calculate mean and standard deviation of growth rates for each drug and concentration
    grouped = regressions_results_df.groupby(['cells_drug', 'cells_conc', 'cells_well'])
    well_growth_rate_df = grouped['b'].mean().reset_index()
    well_growth_rate_df.rename(columns={'b': 'well_sgr'}, inplace=True) # Rename the 'b' column to 'SGR_well'
    
    # Create a DataFrame to store the mean and standard deviation for each drug and concentration
    concentration_growth_rate = []
    
    for (drug, conc), group in well_growth_rate_df.groupby(['cells_drug', 'cells_conc']):
        wells = group['cells_well'].values
        mean_b = group['well_sgr'].mean()
        std_b = group['well_sgr'].std()
        concentration_growth_rate.append([drug, conc, wells, mean_b, std_b])
    
    concentration_growth_rate_df = pd.DataFrame(concentration_growth_rate, columns=['drug', 'concentration', 'wells', 'concentration_sgr', 'sgr_stddev'])
