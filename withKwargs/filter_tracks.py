from my_decorators import timer

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
