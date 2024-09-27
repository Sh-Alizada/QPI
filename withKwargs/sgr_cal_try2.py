import numpy as np
from numba import njit
import numpy as np
from scipy.special import erfcinv
from filter_tracks import filter_tracks
from my_decorators import timer
import pandas as pd

@njit
def calc_b_raw_numba(times_zeroed, masses):
    xav_i = np.mean(times_zeroed)
    yav_i = np.mean(masses)
    x = times_zeroed - xav_i
    y = masses - yav_i
    x2 = np.sum(x**2)
    xy = np.sum(x * y)
    return xy / x2 if x2 != 0 else 0  # Numba optimized

@timer
def sgr_calculator(df, data_folder='data_folder', min_frames=20, min_mass=20, start_frame=1, end_frame=None):

    # Get the filtered DataFrame using cellhunt_v3
    filtered_tracks_df = filter_tracks(df, min_frames=min_frames, min_mass=min_mass, start_frame=start_frame, end_frame=end_frame)

    # Subtract initial time from each time to get times_zeroed
    filtered_tracks_df['times_zeroed'] = filtered_tracks_df.groupby(['location', 'cell'])['time'].transform(lambda x: x - x.iloc[0])

    # Compute xav_i and yav_i directly
    xav_i = filtered_tracks_df.groupby(['location', 'cell'])['times_zeroed'].transform('mean')
    yav_i = filtered_tracks_df.groupby(['location', 'cell'])['mass'].transform('mean')

    # Calculate x_diff and y_diff
    filtered_tracks_df['x_diff'] = filtered_tracks_df['times_zeroed'] - xav_i
    filtered_tracks_df['y_diff'] = filtered_tracks_df['mass'] - yav_i

    # Calculate group-level sums for regression parameters
    x2_sum = filtered_tracks_df.groupby(['location', 'cell'])['x_diff'].transform(lambda x: np.sum(x**2))
    xy_sum = (filtered_tracks_df['x_diff'] * filtered_tracks_df['y_diff']).groupby([filtered_tracks_df['location'], filtered_tracks_df['cell']]).transform('sum')
    TSS = filtered_tracks_df.groupby(['location', 'cell'])['y_diff'].transform(lambda y: np.sum(y**2))

    # Calculate group sizes
    group_sizes = filtered_tracks_df.groupby(['location', 'cell']).size()

    # Ensure group_sizes has the same index as TSS and LRSS
    LRSS = xy_sum**2 / x2_sum
    syx = np.sqrt((TSS - LRSS) / (group_sizes - 2))  # Make sure group_sizes is aligned
    syxy = syx / yav_i
    sb = syx / np.sqrt(x2_sum)
    b_raw = np.where(x2_sum != 0, xy_sum / x2_sum, 0)
    a_raw = yav_i - b_raw * xav_i
    b = np.where(a_raw != 0, b_raw / a_raw, 0)

    # Compile results into a DataFrame
    regression_results_df = pd.DataFrame({
        'location': filtered_tracks_df['location'],
        'cell': filtered_tracks_df['cell'],
        'treatment': filtered_tracks_df['treatment'],
        'concentration': filtered_tracks_df['concentration'],
        'b_raw': b_raw,
        'a_raw': a_raw,
        'r2': LRSS / TSS,
        'syx': syx,
        'syxy': syxy,
        'sb': sb,
        'b': b
    }).drop_duplicates()

    # Apply false cell removal based on regression results
    regression_results_df = remove_false_cells(regression_results_df)

    return regression_results_df

def calculate_mad(group, max_MAD_factor=3):
    c = -1 / (np.sqrt(2) * erfcinv(3 / 2))  # Constant for Median Absolute Deviation (MAD)
    MADsyx = c * np.median(np.abs(group['syx'] - np.median(group['syx'])))
    return group[np.abs(group['syx'] - np.median(group['syx'])) <= max_MAD_factor * MADsyx]


def remove_false_cells(df):
    # Group by treatment and concentration and apply the MAD filtering
    filtered_df = df.groupby(['treatment', 'concentration'], group_keys=True).apply(calculate_mad, include_groups=False)
    # Reset the index, but keep the group indexes
    filtered_df = filtered_df.reset_index(level=2, drop=True)
    return filtered_df.reset_index()