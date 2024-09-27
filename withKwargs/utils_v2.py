import os
import re
import scipy.io
from scipy.signal import medfilt
import numpy as np
import pandas as pd
# import natsort
import trackpy as tp
from natsort import natsorted
from my_decorators import timer
import tifffile
import h5py
import datetime

def matlab_serial_to_datetime(matlab_serial_date):
    """
    Convert a MATLAB serial date number to a Python datetime object.

    MATLAB serial date numbers represent dates as a continuous count of days 
    from a starting date. This function converts the MATLAB serial date number 
    to a Python datetime object, while discarding any fractional seconds.

    Parameters:
    matlab_serial_date (float): MATLAB serial date number to be converted.

    Returns:
    datetime.datetime: Corresponding Python datetime object with microseconds set to zero.
    """
    
    python_datetime = datetime.datetime.fromordinal(int(matlab_serial_date)) + datetime.timedelta(days=matlab_serial_date % 1) - datetime.timedelta(days=366)
    # Create a new datetime object without microseconds
    return python_datetime.replace(microsecond=0)

def datetime_to_matlab_serial(date):
    """
    Convert a Python datetime object to MATLAB serial date number.
    
    Parameters:
    -----------
    date : datetime.datetime
        The datetime object to be converted to MATLAB serial date number.
    
    Returns:
    --------
    float
        The MATLAB serial date number corresponding to the given datetime object.
    """
    # MATLAB serial date number starts from January 0, year 0000
    matlab_start_date = datetime.datetime(1, 1, 1)
    
    # Convert the datetime object to the number of days since MATLAB's start date
    delta = date - matlab_start_date
    matlab_serial_date = delta.days + delta.seconds / (24 * 3600) + 366 + 1
    
    return matlab_serial_date

def load_mat_file(file_path, variable_names=None):
    '''
    Load a MATLAB .mat file, handling both older formats and MATLAB v7.3 files.

    Parameters:
    -----------
    file_path : str
        The path to the .mat file to be loaded.
        
    variable_names : str or list of str, optional
        A specific variable name or a list of variable names to load from the .mat file. 
        If not provided, the function will return all variables as a dictionary.

    Returns:
    --------
    data : dict
        A dictionary containing the requested variables as NumPy arrays. 
        If variable_names is not provided, all variables in the .mat file are returned.
        If a variable is not found, it will not be included in the dictionary.
        If an error occurs, None is returned.
        
    Notes:
    ------
    - For MATLAB v7.3 files, the function uses h5py to load the data, and the arrays are transposed
      to match MATLAB's column-major order.
    - For older MATLAB formats, the function uses scipy.io.loadmat. 
    '''
    
    try:
        # Try to load the .mat file using scipy.io.loadmat
        data = scipy.io.loadmat(file_path)

        if variable_names:
            # Ensure variable_names is a list
            if isinstance(variable_names, str):
                variable_names = [variable_names]

            result = {}
            for var in variable_names:
                if var in data:
                    result[var] = data[var]
                else:
                    print(f"Variable '{var}' not found in the file.")
            
            return result if result else None
        else:
            # Return all data as a dictionary
            return data

    except NotImplementedError: # If there's a NotImplementedError, it means the file is in v7.3 format
        # Use h5py instead
        with h5py.File(file_path, 'r') as mat_file:
            if variable_names:
                # Ensure variable_names is a list
                if isinstance(variable_names, str):
                    variable_names = [variable_names]

                result = {}
                for var in variable_names:
                    if var in mat_file:
                        result[var] = np.array(mat_file[var]).T
                    else:
                        print(f"Variable '{var}' not found in the file.")
                
                return result if result else None
            else:
                # Load all datasets into a dictionary
                data = {}
                for key in mat_file.keys():
                    data[key] = np.array(mat_file[key]).T  # Transpose each array

                return data

    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        return None
    
def validate_filter(value, name):
    """Validates the filter parameter (location/frame)."""
    if isinstance(value, list):
        if len(value) != 2 or not all(isinstance(i, int) for i in value):
            raise ValueError(f"{name} list must contain exactly two integers [min_{name}, max_{name}].")
    elif value is not None and not isinstance(value, int):
        raise ValueError(f"{name} must be an int, list of two ints, or None.")

def is_value_valid(value, group_value):
    """Checks if a group value is within the specified filter range or matches an exact value."""
    if value is None:
        return True
    if isinstance(value, int):
        return group_value == value
    return value[0] <= group_value <= value[1]

def find_files(froot, pattern=r'pos(?P<location>\d+)_frame(?P<frame>\d+)', **filters):
    """
    Find and return matching files with dynamic group extraction,
    using os.listdir and compiled regex. Optionally filter files by any group ranges.
    
    Parameters:
        froot (str): The root directory containing files.
        pattern (str): The regex pattern to match filenames. 
                       Defaults to r'pos(?P<location>\d+)_frame(?P<frame>\d+)'.
        **filters: Keyword arguments corresponding to the regex group names. 
                   Can be int, list of [min, max], or None.
    
    Returns:
        pd.DataFrame: DataFrame where each row is a file with columns based on the regex groups.
    
    Raises:
        ValueError: If a filter parameter is a list with invalid length or non-integer values,
                    or if a filter keyword doesn't match any group name in the regex.
    """
    # Validate and compile the regex pattern
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")
    
    # Extract group names from the regex pattern
    group_names = list(regex.groupindex.keys())
    
    # Ensure that the filter keys match the regex group names
    for group_name in filters.keys():
        if group_name not in group_names:
            raise ValueError(f"Filter key '{group_name}' does not match any group name in the regex pattern.")
        validate_filter(filters[group_name], group_name)
    
    # List all files in the directory
    try:
        all_files = os.listdir(froot)
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory '{froot}' does not exist.")
    
    # Filter and extract data
    matching_files = []
    for filename in all_files:
        match = regex.search(filename)
        if match is not None:
            group_data = {group: int(value) for group, value in match.groupdict().items()}
            if all(is_value_valid(filters.get(group), value) for group, value in group_data.items()):
                matching_files.append([filename] + list(group_data.values()))
    
    # Naturally sort the files by all extracted groups
    matching_files = natsorted(matching_files, key=lambda x: tuple(x[1:]))
    
    # Convert the list of data to a DataFrame
    df = pd.DataFrame(matching_files, columns=['filename'] + group_names)
    
    return df

def cross_correlation_using_fft(frame1, frame2):
    """
    Computes the cross-correlation of two frames using FFT.
    
    Parameters:
    - frame1: np.ndarray, the first image/frame (e.g., reference frame).
    - frame2: np.ndarray, the second image/frame to compare (e.g., shifted frame).
    
    Returns:
    - cross_corr: np.ndarray, the cross-correlation array.
    """
    # Ensure the frames are in float format for FFT
    frame1 = frame1.astype(np.float64)
    frame2 = frame2.astype(np.float64)
    
    # Calculate the cross-correlation using FFT
    f1 = np.fft.fft2(frame1)            # Compute the FFT of the first frame
    f2 = np.fft.fft2(frame2)            # Compute the FFT of the second frame
    f2_conj = np.conj(f2)               # Compute the complex conjugate of the second frame's FFT
    cross_corr = np.fft.ifft2(f1 * f2_conj)  # Multiply and inverse FFT to get the cross-correlation
    
    # Take the real part of the cross-correlation result
    cross_corr = np.fft.fftshift(np.real(cross_corr))  # Shift the zero-frequency component to the center
    
    return cross_corr

def find_shift(frame1, frame2):
    """
    Finds the pixel shift between two frames using cross-correlation.
    
    Parameters:
    - frame1: np.ndarray, the first image/frame (e.g., reference frame).
    - frame2: np.ndarray, the second image/frame to compare (e.g., shifted frame).
    
    Returns:
    - shift_y: int, the amount by which `frame2` should be shifted along the y-axis (vertical shift) to match `frame1`.
    - shift_x: int, the amount by which `frame2` should be shifted along the x-axis (horizontal shift) to match `frame1`.
    
    Note:
    The returned shift values indicate how much `frame2` is shifted relative to `frame1`.
    A positive value indicates that `frame2` is shifted down or to the right relative to `frame1`.
    A negative value indicates that `frame2` is shifted up or to the left relative to `frame1`.
    """
    # Compute the cross-correlation using the FFT-based method
    cross_corr = cross_correlation_using_fft(frame1, frame2)
    
    # Find the peak in the cross-correlation array
    max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    
    # Calculate the shift by adjusting the peak index relative to the center
    shift_y, shift_x = np.array(max_idx) - np.array(frame1.shape) // 2
    
    # Negate the shifts to match the direction of frame movement
    shift_y, shift_x = -shift_y, -shift_x
    
    return shift_x, shift_y

@timer
def save_ome_tiff(file_path, image_array, metadata=None):
    
    """
    Save a NumPy array as an OME-TIFF file.

    Parameters:
    - image_array: np.ndarray
        The NumPy array to save.
    - file_path: str
        The file path where the OME-TIFF file will be saved.
    - metadata: dict, optional
        A dictionary containing metadata to include in the OME-TIFF file.

    Example:
    save_as_ome_tiff(image_array, 'output.ome.tiff')
    """

    if metadata is None:
        metadata = {}

    # Save the array as an OME-TIFF
    tifffile.imwrite(
        file_path,
        image_array,
        imagej=True,
        photometric='minisblack',
        compression='zstd',
        metadata=metadata
    )
    
    return None

def get_drug_concentration(well_number, drug_names_df, well_map_df, drug_map_df, concentration_map_df):
    """
    Retrieves the corresponding drug and concentration for a given well number
    from the provided plate, drug map, and concentration map DataFrames.
    
    Parameters:
    well_number (int): The well number to look up.
    plate_df (DataFrame): DataFrame representing the plate.
    drug_map_df (DataFrame): DataFrame representing the drug map.
    concentration_map_df (DataFrame): DataFrame representing the concentration map.
    
    Returns:
    tuple: A tuple containing the drug number and concentration value.
    """
    # Find the location of the well number in the plate
    location = None
    for i, row in well_map_df.iterrows():
        if well_number in row.values:
            location = (i, row[row == well_number].index[0])
            break
    
    if location is None:
        raise ValueError(f"Well number {well_number} not found in the plate.")
    
    # Retrieve the corresponding drug and concentration
    drug = drug_map_df.iat[location[0], location[1]]
    drug_name = drug_names_df[drug-1][0]
    concentration = concentration_map_df.iat[location[0], location[1]]
    
    return drug_name, drug, concentration

@timer    
def track_cells(location, df, exp_folder_path, data_folder='data_folder', 
                mass_factor=1, search_radius=40, adaptive_step=0.95, adaptive_stop=10, tracking_memory=1):
    
    # Add z dimension to track the cells with mass in addition to x and y
    if df.get('mass', None) is None:
        return
    else:
        df['z'] = df['mass'] * mass_factor # z is adjusted mass with mass factor
        
        
        tp.quiet() # stop printing number of trajectories
        # tp.link tracks the cells using columns  ['z', 'y', 'x', 'frame'] and adds 'particle' column indicating the cell IDs
        df = tp.link(df, search_range=search_radius, memory=tracking_memory, adaptive_step=adaptive_step, 
                            adaptive_stop=adaptive_stop, pos_columns=['z','y','x'], t_column='frame')
        
        df.drop('z', axis=1, inplace=True) # remove z column
        df.insert(0, 'cell', df.pop('particle')) # change column name to cell and move to the first location
        df.sort_values(by=['cell', 'frame'], inplace=True) # sort tracking data by cell ID and frame number
        
        data_folder_path =  os.path.join(exp_folder_path, data_folder)
        os.makedirs(data_folder_path, exist_ok=True) # make the directory if not exist
        file_path_data = os.path.join(data_folder_path, f"cell_data_loc_{location}.csv") # assign tracking data file name
        df.to_csv(file_path_data, index=False) # save tracking data
    
@timer
def concat_track_files(folder_path, pattern='cell_data_loc_(?P<location>\d+)', location_range=None):
    
    track_files=find_files(folder_path, pattern=pattern, location=location_range)
    
    tracks_df=[]
    
    # Load each CSV file and append to the list
    for filename in track_files['filename']:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        tracks_df.append(df)
    
    # Concatenate all dataframes while ignoring headers of subsequent files
    combined_tracks_df = pd.concat(tracks_df, ignore_index=True)
    
    # Save the concatenated dataframe to a new CSV file
    file_path = os.path.join(folder_path, 'all_cell_data.csv')
    combined_tracks_df.to_parquet(file_path, engine='pyarrow', index=False, compression='zstd')

    return combined_tracks_df



def medfilt1_truncate(x, filter_size):
    """
    Applies a 1-D median filter to the input signal with edge truncation.

    Parameters:
    x (array-like): The input signal.
    filter_size (int): The size of the filter window over which to compute the median.

    Returns:
    numpy.ndarray: The filtered signal with truncated edges.
    """
    # Apply the median filter to the entire signal
    y = medfilt(x, kernel_size=filter_size)
    
    # Compute the half window size
    half_window = (filter_size - 1) // 2
    
    # Handle the edges manually:
    # For the start of the signal, compute the median over smaller segments
    for i in range(half_window):
        y[i] = np.median(x[:i + half_window + 1])
    
    # For the end of the signal, compute the median over smaller segments
    for i in range(-half_window, 0):
        y[i] = np.median(x[i - half_window:])
    
    return y






















