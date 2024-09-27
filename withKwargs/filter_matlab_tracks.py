import pandas as pd
import numpy as np
import scipy.io as sio

def filter_tracks(froot, minp, StartFrame, EndFrame, minmass):
    """
    Function to filter cell tracking data based on given criteria.
    
    Parameters:
    froot (str): Path to the folder containing the data file.
    minp (int): Minimum number of occurrences for a cell to be considered.
    StartFrame (int): Start frame for the analysis window.
    EndFrame (int): End frame for the analysis window.
    minmass (float, optional): Minimum mean mass for cells to be considered. Default is 110.
    
    Returns:
    pd.DataFrame: Filtered DataFrame containing only the cells that meet the criteria.
    """
    
    # Load tracks and tracksColHeaders from the .mat file
    mat_contents = sio.loadmat(froot + 'data_allframes.mat')  # Load .mat file contents
    tracks = mat_contents['tracks']  # Extract 'tracks' data from the .mat file
    tracksColHeaders = [str(h[0]) for h in mat_contents['tracksColHeaders'][0]]  # Extract and convert column headers to strings
    # tracksColHeaders.append('QDF')  # Adding QDF as a placeholder, adjust if necessary

    # Convert the tracks data to a pandas DataFrame
    df = pd.DataFrame(tracks, columns=tracksColHeaders)  # Create a DataFrame from tracks data using column headers
    
    # Define column names
    frameloc = 'Frame ID'  # Define the column name for frame IDs
    cellsloc = 'id'  # Define the column name for cell IDs
    massloc = 'Mass (pg)'  # Define the column name for cell mass, adjust if necessary

    # Filter the DataFrame to include only the frames within the specified frame range
    target = (df[frameloc] >= StartFrame) & (df[frameloc] <= EndFrame)  # Create a boolean mask for rows within the frame range
    filtered_df = df.loc[target]  # Apply the mask to filter the DataFrame
    
    # Count occurrences of each cell ID and filter based on minp
    Ncount = filtered_df[cellsloc].value_counts()  # Count occurrences of each cell ID
    targetcells = Ncount[Ncount >= minp].index  # Select cell IDs that appear at least 'minp' times

    # Filter the DataFrame to include only target cells
    newtracks = filtered_df[filtered_df[cellsloc].isin(targetcells)]  # Filter DataFrame to include only rows with target cell IDs

    # Calculate mean mass for each cell and filter out those with mean mass <= minmass
    mean_mass = newtracks.groupby(cellsloc)[massloc].mean()  # Calculate mean mass for each cell ID
    valid_cells = mean_mass[mean_mass > minmass].index  # Select cell IDs with mean mass greater than 'minmass'
    targetcells = np.intersect1d(targetcells, valid_cells)  # Ensure cells meet both occurrence and mass criteria

    # Filter the DataFrame to include only valid cells
    final_tracks = filtered_df[filtered_df[cellsloc].isin(targetcells)]  # Filter DataFrame to include only rows with valid cell IDs

    return final_tracks  # Return the final filtered DataFrame




