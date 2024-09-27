import pandas as pd
import numpy as np
import os
from my_decorators import timer

@timer
def sgr_calculator(df, exp_folder_path, data_folder='data_folder', min_frames=20, min_mass=20, start_frame=1, end_frame=None):
    """
    Calculates specific growth rate (sgr) for cell tracking data and saves results in CSV files.

    Parameters:
    - df (pd.DataFrame): DataFrame containing cell data with columns 'location', 'cell', 'time', 'mass', etc.
    - exp_folder_path (str): Path to the experiment folder where results will be saved.
    - data_folder (str, optional): Folder name where results will be saved (default is 'data_folder').
    - min_frames (int, optional): Minimum number of frames a cell must be tracked to be included (default is 20).
    - min_mass (float, optional): Minimum mean mass for cells to be considered (default is 20).
    - start_frame (int, optional): Starting frame for filtering the time window (default is 1).
    - end_frame (int, optional): Ending frame for filtering the time window. If None, it uses the maximum frame in the data (default is None).
    - save (bool, optional): If True, saves the results to CSV files (default is True).

    Returns:
    - cell_sgr_df (pd.DataFrame): DataFrame containing sgr values for each cell.
    - well_sgr_df (pd.DataFrame): DataFrame containing average sgr values for each well.
    - conc_sgr_df (pd.DataFrame): DataFrame containing average sgr values and standard deviation for each concentration.
    """
    
    # Filter tracks
    filtered_tracks_df = filter_tracks(df, min_frames=min_frames, min_mass=min_mass, start_frame=start_frame, end_frame=end_frame)
    
    # Calculate zeroed times
    filtered_tracks_df['times_zeroed'] = filtered_tracks_df['time'] - filtered_tracks_df.groupby(['location', 'cell'])['time'].transform('first')
    
    # Calculate regression parameters
    grouped = filtered_tracks_df.groupby(['location', 'cell', 'treatment', 'concentration', 'well'])
    results = []
    for name, group in grouped:
        params = get_regression_params(group['times_zeroed'].values, group['mass'].values)
        params.update(dict(zip(['location', 'cell', 'treatment', 'concentration', 'well'], name)))
        results.append(params)
    regression_res_df = pd.DataFrame(results)

    # Apply false cell removal based on regression results
    regression_res_df = remove_false_cells(regression_res_df)

    # Prepare sgr DataFrames
    cell_sgr_df = regression_res_df[['cell', 'location', 'well', 'treatment', 'concentration', 'sgr']].copy()  # Explicitly create a copy
    cell_sgr_df.sort_values(by=['location', 'cell'], inplace=True)
    well_sgr_df = cell_sgr_df.groupby(['well', 'treatment', 'concentration'], as_index=False)['sgr'].mean()
    conc_sgr_df = well_sgr_df.groupby(['treatment', 'concentration']).agg(wells=('well', list),
                                                                          sgr=('sgr', 'mean'),
                                                                          std_dev=('sgr', 'std')).reset_index()

    # Save results
    data_folder_path = os.path.join(exp_folder_path, data_folder)
    os.makedirs(data_folder_path, exist_ok=True)  # Make the directory if it doesn't exist
    
    file_path_cell = os.path.join(data_folder_path, f"cell_sgr_minf{min_frames}_minm{min_mass}.csv")
    cell_sgr_df.to_csv(file_path_cell, index=False)
    
    file_path_well = os.path.join(data_folder_path, f"well_sgr_minf{min_frames}_minm{min_mass}.csv")
    well_sgr_df.to_csv(file_path_well, index=False)   
    
    file_path_conc = os.path.join(data_folder_path, f"concentration_sgr_minf{min_frames}_minm{min_mass}.csv")
    conc_sgr_df.to_csv(file_path_conc, index=False)   

    return cell_sgr_df, well_sgr_df, conc_sgr_df

@timer
def filter_tracks(df, min_frames=20, min_mass=20, start_frame=1, end_frame=None):
    """
    Filters cell tracking data based on a specified number of frames and mean mass.

    Parameters:
    - df (pd.DataFrame): DataFrame containing cell tracking data with columns like 'location', 'cell', 'frame', and 'mass'.
    - min_frames (int, optional): Minimum number of frames a cell must be tracked to be included (default is 20).
    - min_mass (float, optional): Minimum mean mass for cells to be considered (default is 20).
    - start_frame (int, optional): Start frame for the analysis window (default is 1).
    - end_frame (int, optional): End frame for the analysis window. If None, uses the max frame in the data (default is None).

    Returns:
    - pd.DataFrame: Filtered DataFrame containing only the cells that meet the criteria.
    """
    # Set the end_frame to the max frame in the dataset if not provided
    if end_frame is None:
        end_frame = df['frame'].max()

    # Apply frame filtering early to reduce the data size
    df_f = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]

    # Group by location and cell, and calculate frame counts and mean mass
    cell_stats = df_f.groupby(['location', 'cell'], as_index=False).agg(
        frame_count=('frame', 'size'),
        mean_mass=('mass', 'mean')
    )

    # Filter cells based on min_frames and min_mass in one step
    valid_cells = cell_stats.query('frame_count >= @min_frames and mean_mass >= @min_mass')

    # Use a merge based on the filtered cells to reduce the final DataFrame
    result = df.merge(valid_cells[['location', 'cell']], on=['location', 'cell'])

    return result

def get_regression_params(times_zeroed, masses):
    """
    Calculates linear regression parameters for given time and mass data.

    Parameters:
    - times_zeroed (np.ndarray): Array of time values with zero reference.
    - masses (np.ndarray): Array of mass values corresponding to the times.

    Returns:
    - dict: A dictionary containing regression parameters such as 'b_raw', 'a_raw', 'r2', 'syx', 'syxy', 'sb', 'mean_mass', and 'sgr'.
    """
    xav = np.mean(times_zeroed)
    yav = np.mean(masses)

    # Deviations from mean
    x = times_zeroed - xav
    y = masses - yav

    # Sums of squares and products
    sum_xx = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Slope (b_raw) and intercept (a_raw)
    b_raw = sum_xy / sum_xx
    a_raw = yav - b_raw * xav

    # Normalized slope (sgr)
    sgr = b_raw / a_raw

    # Goodness of fit (r-squared)
    TSS = np.sum(y ** 2)
    LRSS = (sum_xy ** 2) / sum_xx
    r2 = LRSS / TSS

    # Residual standard deviation (syx) and standard error (sb)
    syx = np.sqrt((TSS - LRSS) / (len(times_zeroed) - 2))
    sb = np.sqrt(syx ** 2 / sum_xx)

    # ??
    syxy = syx / yav

    return {
        'b_raw': b_raw,
        'a_raw': a_raw,
        'r2': r2,
        'syx': syx,
        'syxy': syxy,
        'sb': sb,
        'mean_mass': yav,
        'sgr': sgr,
    }

def remove_false_cells(df):
    """
    Removes cells from the DataFrame that are considered outliers based on MAD filtering of regression parameters.

    Parameters:
    - df (pd.DataFrame): DataFrame containing regression results with 'syx' and 'sgr' columns.

    Returns:
    - pd.DataFrame: Filtered DataFrame with outlier cells removed.
    """
    # Apply the MAD filtering on the grouped data
    filtered_df = df.groupby(['treatment', 'concentration'], group_keys=False).apply(filter_by_mad)

    # Reset index, which turns 'treatment' and 'concentration' into columns
    filtered_df = filtered_df.reset_index()

    # Drop the level_2 index that may have been introduced by apply()
    if 'level_2' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns='level_2')

    return filtered_df

def filter_by_mad(group, max_MAD_factor=3):
    """
    Filters cells within a group based on Median Absolute Deviation (MAD) from the median 'sgr' and 'syx'.

    Parameters:
    - group (pd.DataFrame): DataFrame for a specific group of cells.
    - max_MAD_factor (float, optional): The factor by which MAD outliers are determined (default is 3).

    Returns:
    - pd.DataFrame: Filtered DataFrame with outlier cells removed.
    """
    syx = group['syx'].values
    sgr = group['sgr'].values
    
    # MAD constant
    c = 1.4826  # Approximation of 1 / (sqrt(2) * erfcinv(3/2))

    # Calculate MAD for syx and sgr
    median_syx = np.median(syx)
    median_sgr = np.median(sgr)
    
    MADsyx = c * np.median(np.abs(syx - median_syx))
    MADsgr = c * np.median(np.abs(sgr - median_sgr))
    
    # Filter out cells that are outliers based on MAD
    syx_remove = np.abs(syx - median_syx) > max_MAD_factor * MADsyx
    sgr_remove = np.abs(sgr - median_sgr) > max_MAD_factor * MADsgr
    
    return group[~(syx_remove | sgr_remove)]
