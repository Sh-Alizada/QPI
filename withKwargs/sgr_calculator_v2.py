import pandas as pd
import numpy as np
from scipy.special import erfcinv
from filter_tracks import filter_tracks
from my_decorators import timer

@timer
def sgr_calculator(df, data_folder='data_folder', min_frames=20, min_mass=20, start_frame=1, end_frame=None):

    
    # Get the filtered DataFrame using cellhunt_v3
    filtered_tracks_df = filter_tracks(df, min_frames=min_frames, min_mass=min_mass, start_frame=start_frame, end_frame=end_frame)
    
    # Subtract the initial time from each time to get times_zeroed
    filtered_tracks_df['times_zeroed'] = filtered_tracks_df.groupby(['location', 'cell'])['time'].transform(lambda x: x - x.iloc[0])
    
    # Apply the regression functions and retain drug and concentration information
    regression_results_df = filtered_tracks_df.groupby(['location', 'cell', 'treatment', 'concentration']).apply(
        lambda group: pd.Series(get_regression_params(group['times_zeroed'].values, group['mass'].values))
    , include_groups=False).reset_index()

    
    # Apply false cell removal based on regression results
    regression_results_df = remove_false_cells(regression_results_df)

    return regression_results_df

# Function to calculate xav_i
def calc_xav_i(times_zeroed):
    return np.mean(times_zeroed)

# Function to calculate yav_i
def calc_yav_i(masses):
    return np.mean(masses)

# Function to calculate b_raw
def calc_b_raw(times_zeroed, masses):
    xav_i = calc_xav_i(times_zeroed)
    yav_i = calc_yav_i(masses)
    x = times_zeroed - xav_i
    y = masses - yav_i
    x2 = np.sum(x**2)
    xy = np.sum(x * y)
    return xy / x2

# Function to calculate a_raw
def calc_a_raw(times_zeroed, masses):
    b_raw = calc_b_raw(times_zeroed, masses)
    xav_i = calc_xav_i(times_zeroed)
    yav_i = calc_yav_i(masses)
    return yav_i - b_raw * xav_i

# Function to calculate r2
def calc_r2(times_zeroed, masses):
    xav_i = calc_xav_i(times_zeroed)
    yav_i = calc_yav_i(masses)
    x = times_zeroed - xav_i
    y = masses - yav_i
    TSS = np.sum(y**2)
    LRSS = np.sum(x * y)**2 / np.sum(x**2)
    return LRSS / TSS

# Function to calculate syx
def calc_syx(times_zeroed, masses):
    xav_i = calc_xav_i(times_zeroed)
    yav_i = calc_yav_i(masses)
    x = times_zeroed - xav_i
    y = masses - yav_i
    TSS = np.sum(y**2)
    LRSS = np.sum(x * y)**2 / np.sum(x**2)
    return np.sqrt((TSS - LRSS) / (len(times_zeroed) - 2))

# Function to calculate syxy
def calc_syxy(times_zeroed, masses):
    syx = calc_syx(times_zeroed, masses)
    yav_i = calc_yav_i(masses)
    return syx / yav_i

# Function to calculate sb
def calc_sb(times_zeroed, masses):
    syx = calc_syx(times_zeroed, masses)
    xav_i = calc_xav_i(times_zeroed)
    x = times_zeroed - xav_i
    x2 = np.sum(x**2)
    return np.sqrt(syx**2 / x2)

# Function to calculate normalized growth rate b
def calc_b(times_zeroed, masses):
    b_raw = calc_b_raw(times_zeroed, masses)
    a_raw = calc_a_raw(times_zeroed, masses)
    return b_raw / a_raw

# Function to get the regression values in a dict
def get_regression_params(times_zeroed, masses):
    return {
        'b_raw': calc_b_raw(times_zeroed, masses),
        'a_raw': calc_a_raw(times_zeroed, masses),
        'r2': calc_r2(times_zeroed, masses),
        'syx': calc_syx(times_zeroed, masses),
        'syxy': calc_syxy(times_zeroed, masses),
        'sb': calc_sb(times_zeroed, masses),
        'b': calc_b(times_zeroed, masses),
    }

def calculate_mad(group, max_MAD_factor=3):
    c = -1 / (np.sqrt(2) * erfcinv(3 / 2)) # Constant for Median Absolute Deviation (MAD)
    MADsyx = c * np.median(np.abs(group['syx'] - np.median(group['syx'])))
    return group[np.abs(group['syx'] - np.median(group['syx'])) <= max_MAD_factor * MADsyx]

@timer
def remove_false_cells(df):
    # Group by drug and concentration and apply the MAD filtering
    filtered_df = df.groupby(['treatment', 'concentration'], group_keys=True).apply(calculate_mad, include_groups=False)
    # Reset the index, but keep the group indexes
    filtered_df = filtered_df.reset_index(level=2, drop=True)
    return filtered_df.reset_index()