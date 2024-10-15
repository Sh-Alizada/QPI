import pandas as pd
import numpy as np
from scipy.special import erfcinv
import os
from my_decorators import timer



@timer
def sgr_calculator(df, exp_folder_path, data_folder='data_folder', min_frames=20, min_mass=20, start_frame=1, end_frame=None):
    
    # Filter tracks
    filtered_tracks_df = filter_tracks(df, min_frames=min_frames, min_mass=min_mass, start_frame=start_frame, end_frame=end_frame)
    
    # Calculate zeroed times
    filtered_tracks_df['times_zeroed'] = filtered_tracks_df.groupby(['location', 'cell'])['time'].transform(lambda x: x - x.iloc[0])

    # Apply the combined regression function
    # I had to further group by concentration and well to get somehow add them to the regression_results_df
    regression_results_df = filtered_tracks_df.groupby(['location', 'cell', 'treatment', 'concentration','well']).apply(
        lambda group: pd.Series(get_regression_params(group['times_zeroed'].values, group['mass'].values))
    , include_groups=False).reset_index()

    # Apply false cell removal based on regression results
    # regression_results_df = remove_false_cells(regression_results_df)
    
    # data_folder_path =  os.path.join(exp_folder_path, data_folder)
    # os.makedirs(data_folder_path, exist_ok=True) # make the directory if not exist
    # file_path_data = os.path.join(data_folder_path, "sgr_regression_params.csv") # assign tracking data file name
    # regression_results_df.to_csv(file_path_data, index=False) # save tracking data

    regression_results_df['loc_cell_amnt'] = regression_results_df['location'].map(regression_results_df['location'].value_counts())
    regression_results_df['well_cell_amnt'] = regression_results_df['well'].map(regression_results_df['well'].value_counts())

    cell_sgr_df = regression_results_df[['cell', 'location', 'well', 'concentration', 
                                         'sgr', 'loc_cell_amnt', 'well_cell_amnt']].copy().sort_values(by=['cell', 'location'])
    
    

    return cell_sgr_df

@timer
def filter_tracks(df, min_frames=20, min_mass=20, start_frame=1, end_frame=None):
    """
    Function to filter cell tracking data based on given criteria.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing cell tracking data.
    - min_tracked_frames (int): Minimum number of frames a cell must be tracked to be included.
    - min_mass (float): Minimum mean mass for cells to be considered.
    - start_frame (int): Start frame for the analysis window.
    - end_frame (int): End frame for the analysis window (if None, uses max frame).
    
    Returns:
    - pd.DataFrame: Filtered DataFrame containing only the cells that meet the criteria.
    """
    if end_frame is None:
        end_frame = df['frame'].max()

    # Filter the DataFrame to include only the frames within the specified frame range
    target = (df['frame'] >= start_frame) & (df['frame'] <= end_frame)
    filtered_df = df.loc[target]

    # Group by location and cell, then calculate the number of frames and mean mass for each cell
    cell_stats = filtered_df.groupby(['location', 'cell']).agg(frame_count=('frame', 'size'), mean_mass=('mass', 'mean'))

    # Filter cells based on the minimum number of tracked frames and minimum mean mass
    valid_cells = cell_stats[(cell_stats['frame_count'] >= min_frames) & (cell_stats['mean_mass'] >= min_mass)].index

    # Create a mask to filter the DataFrame for valid cells
    filtered_df = filtered_df.set_index(['location', 'cell']).loc[valid_cells].reset_index()

    return filtered_df

def get_regression_params(times_zeroed, masses):
    
    # Calculate average time and mass
    xav = np.mean(times_zeroed)
    yav = np.mean(masses)
    
    # Calculate differences
    x = times_zeroed - xav
    y = masses - yav
    
    # Calculate sums of squares and cross products
    x2 = np.sum(x ** 2)
    xy = np.sum(x * y)
    
    # Calculate b_raw and a_raw
    b_raw = xy / x2
    a_raw = yav - b_raw * xav
    
    # Calculate normalized growth rate b
    sgr = b_raw / a_raw
    
    # Calculate TSS and LRSS
    TSS = np.sum(y ** 2)
    LRSS = xy ** 2 / x2
    
    # Calculate r2
    r2 = LRSS / TSS
    
    # Calculate syx
    syx = np.sqrt((TSS - LRSS) / (len(times_zeroed) - 2))
    
    # Calculate syxy
    syxy = syx / yav
    
    # Calculate sb
    sb = np.sqrt(syx ** 2 / x2)
    
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
    # Group by treatment and concentration and apply MAD filtering
    filtered_df = df.groupby(['treatment', 'concentration'], group_keys=True).apply(filter_by_mad, include_groups=False)
    filtered_df = filtered_df.reset_index(level=2, drop=True)  # Adjust depending on your use case
    return filtered_df.reset_index()

# MAD calculation
def filter_by_mad(group, max_MAD_factor=3):
    syx = group['syx'].values
    sgr = group['sgr'].values
    
    # Calculate medians
    median_syx = np.median(syx)
    median_sgr = np.median(sgr)
    
    # MAD for syx and sgr
    MADsyx = np.median(np.abs(syx - median_syx))
    MADsgr = np.median(np.abs(sgr - median_sgr))
    
    # Remove cells that are outliers based on MAD
    syx_remove = np.abs(syx - median_syx) > max_MAD_factor * MADsyx
    sgr_remove = np.abs(sgr - median_sgr) > max_MAD_factor * MADsgr
    
    # Return the filtered group
    return group[~(syx_remove | sgr_remove)] 





































