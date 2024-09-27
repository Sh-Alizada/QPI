import numpy as np
from scipy.special import erfcinv
from filter_tracks import filter_tracks
from my_decorators import timer

@timer
def sgr_calculator(df, data_folder='data_folder', min_frames=20, min_mass=20, start_frame=1, end_frame=None):

    # Filter tracks using cellhunt_v3
    filtered_tracks_df = filter_tracks(df, min_frames=min_frames, min_mass=min_mass, start_frame=start_frame, end_frame=end_frame)

    # Subtract initial time from each time to get times_zeroed
    filtered_tracks_df['times_zeroed'] = filtered_tracks_df.groupby(['location', 'cell'])['time'].transform(lambda x: x - x.iloc[0])

    # Calculate xav_i and yav_i for each group
    grouped = filtered_tracks_df.groupby(['location', 'cell'])
    filtered_tracks_df['xav_i'] = grouped['times_zeroed'].transform('mean')
    filtered_tracks_df['yav_i'] = grouped['mass'].transform('mean')

    # Calculate x and y differences from means
    filtered_tracks_df['x_diff'] = filtered_tracks_df['times_zeroed'] - filtered_tracks_df['xav_i']
    filtered_tracks_df['y_diff'] = filtered_tracks_df['mass'] - filtered_tracks_df['yav_i']

    # Vectorized calculation of b_raw, a_raw, r2, syx, syxy, sb, and b
    group_sums = filtered_tracks_df.groupby(['location', 'cell', 'treatment', 'concentration']).agg(
        x2_sum=('x_diff', lambda x: np.sum(x**2)),
        xy_sum=('x_diff', lambda x: np.sum(x * filtered_tracks_df.loc[x.index, 'y_diff'])),
        TSS=('y_diff', lambda y: np.sum(y**2))
    )

    group_sums['b_raw'] = group_sums['xy_sum'] / group_sums['x2_sum']
    group_sums['a_raw'] = filtered_tracks_df.groupby(['location', 'cell'])['yav_i'].first() - group_sums['b_raw'] * filtered_tracks_df.groupby(['location', 'cell'])['xav_i'].first()
    group_sums['LRSS'] = group_sums['xy_sum']**2 / group_sums['x2_sum']
    group_sums['r2'] = group_sums['LRSS'] / group_sums['TSS']
    group_sums['syx'] = np.sqrt((group_sums['TSS'] - group_sums['LRSS']) / (grouped.size() - 2))
    group_sums['syxy'] = group_sums['syx'] / filtered_tracks_df.groupby(['location', 'cell'])['yav_i'].first()
    group_sums['sb'] = group_sums['syx'] / np.sqrt(group_sums['x2_sum'])
    group_sums['b'] = group_sums['b_raw'] / group_sums['a_raw']

    # Merge results back into the original DataFrame if needed
    regression_results_df = group_sums.reset_index()

    # Apply false cell removal based on regression results
    regression_results_df = remove_false_cells(regression_results_df)

    return regression_results_df

def calculate_mad(group, max_MAD_factor=3):
    c = -1 / (np.sqrt(2) * erfcinv(3 / 2))  # Constant for Median Absolute Deviation (MAD)
    MADsyx = c * np.median(np.abs(group['syx'] - np.median(group['syx'])))
    return group[np.abs(group['syx'] - np.median(group['syx'])) <= max_MAD_factor * MADsyx]

@timer
def remove_false_cells(df):
    # Group by treatment and concentration and apply the MAD filtering
    filtered_df = df.groupby(['treatment', 'concentration'], group_keys=True).apply(calculate_mad, include_groups=False)
    # Reset the index, but keep the group indexes
    filtered_df = filtered_df.reset_index(level=2, drop=True)
    return filtered_df.reset_index()
