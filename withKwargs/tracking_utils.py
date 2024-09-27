import os
import re
import scipy.io
import numpy as np
import pandas as pd
import natsort
import trackpy as tp
from natsort import natsorted
from my_decorators import timer
import tifffile
import h5py
import datetime

def conv_matlab_date(matlab_serial_date):
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


def load_mat_file(file_path, variable_name=None):
    '''
    Load a MATLAB .mat file, handling both older formats and MATLAB v7.3 files.

    Parameters:
    -----------
    file_path : str
        The path to the .mat file to be loaded.
        
    variable_name : str, optional
        The name of a specific variable to load from the .mat file. If not provided,
        the function will return all variables as a dictionary.

    Returns:
    --------
    data : dict or np.ndarray
        If variable_name is not provided, a dictionary containing all variables in the .mat file is returned.
        If variable_name is provided, the corresponding variable is returned as a NumPy array.
        If the variable is not found, or if an error occurs, None is returned.
        
    Notes:
    ------
    - For MATLAB v7.3 files, the function uses h5py to load the data, and the arrays are transposed
      to match MATLAB's column-major order.
    - For older MATLAB formats, the function uses scipy.io.loadmat. 
    '''
    
    try:
        # Try to load the .mat file using scipy.io.loadmat
        data = scipy.io.loadmat(file_path)

        if variable_name:
            # Check if the variable exists in the loaded data
            if variable_name in data:
                return data[variable_name]
            else:
                print(f"Variable '{variable_name}' not found in the file.")
                return None
        else:
            # Return all data as a dictionary
            return data

    except NotImplementedError:
        # If there's a NotImplementedError, it means the file is in v7.3 format
        with h5py.File(file_path, 'r') as mat_file:
            if variable_name:
                if variable_name in mat_file:
                    data = np.array(mat_file[variable_name]).T
                    return data
                else:
                    print(f"Variable '{variable_name}' not found in the file.")
                    return None
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
    
def find_files(froot, pattern=r'pos(?P<location>\d+)_frame(?P<frame>\d+)', location=None, frame=None):
    """
    Find and return matching files with locations and frames as integers,
    using os.listdir and compiled regex. Optionally filter files by location and frame ranges.
    
    Args:
        froot (str): The root directory containing files.
        pattern (str): The regex pattern to match filenames. 
                        Defaults to r'pos(?P<location>\d+)_frame(?P<frame>\d+) for "*pos1_frame1*"
        
        location (int, list, or None): Location to filter files. If int, filters by exact location.
                                      If list, filters by location range [min_location, max_location].
                                      If None, no filtering by location.
        frame (int, list, or None): Frame to filter files. If int, filters by exact frame.
                                    If list, filters by frame range [min_frame, max_frame].
                                    If None, no filtering by frame.
    
    Returns:
        np.ndarray: Nx3 array where each row is [filename, position, frame].
    
    Raises:
        ValueError: If location or frame parameters are lists with invalid lengths or non-integer values.
    """
    # Validate and compile the regex pattern
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")
    
    # Validate the filtering parameters
    validate_filter(location, "location")
    validate_filter(frame, "frame")
    
    # List all files in the directory
    try:
        all_files = os.listdir(froot)
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory '{froot}' does not exist.")
    
    # List comprehension to filter and extract data
    matching_files = [
        [filename, int(match.group('location')), int(match.group('frame'))]
        for filename in all_files
        if (match := regex.search(filename)) is not None
        and (is_location_valid(location, int(match.group('location'))))
        and (is_frame_valid(frame, int(match.group('frame'))))
    ]
    
    # Naturally sort the files by location and frame
    matching_files = natsorted(matching_files, key=lambda x: (x[1], x[2]))
    
    return np.array(matching_files, dtype=object)

def validate_filter(value, name):
    """
    Validate that the filter value is either an integer, a list with exactly two integers, or None.
    
    Args:
        value (int, list, or None): The filter value to validate.
        name (str): The name of the filter (for error messages).
    
    Raises:
        ValueError: If the value is invalid.
    """
    if value is not None:
        if isinstance(value, int):
            return
        elif isinstance(value, list):
            if len(value) != 2:
                raise ValueError(f"{name.capitalize()} range must be a list with exactly two elements.")
            if not all(isinstance(v, int) for v in value):
                raise ValueError(f"All elements in the {name} range must be integers.")
        else:
            raise ValueError(f"{name.capitalize()} must be an integer, list of two integers, or None.")

def is_location_valid(location, pos):
    """
    Check if the location is valid based on the location parameter.
    
    Args:
        location (int, list, or None): Location filter.
        pos (int): Position to check.
    
    Returns:
        bool: True if the location is valid, False otherwise.
    """
    if location is None:
        return True
    elif isinstance(location, int):
        return pos == location
    elif isinstance(location, list) and len(location) == 2:
        min_loc, max_loc = location
        return min_loc <= pos <= max_loc
    else:
        return False

def is_frame_valid(frame, frm):
    """
    Check if the frame is valid based on the frame parameter.
    
    Args:
        frame (int, list, or None): Frame filter.
        frm (int): Frame to check.
    
    Returns:
        bool: True if the frame is valid, False otherwise.
    """
    if frame is None:
        return True
    elif isinstance(frame, int):
        return frm == frame
    elif isinstance(frame, list) and len(frame) == 2:
        min_frm, max_frm = frame
        return min_frm <= frm <= max_frm
    else:
        return False

def load_frame_and_time(file_path):
    '''
    Loads the raw phase image and its timestamp from a .mat file
    
    Parameters
    ----------
    file_path : path to a .mat file .

    Returns
    -------
    raw_image : raw phase image as saved in .mat file.
    time : timestamp of a raw phase image.

    '''
    
    # Load the .mat file
    data = scipy.io.loadmat(file_path)
    
    # Extract the Phase data
    raw_image = data['Phase']
    time = data['timestamp']

    return raw_image, time


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

def read_maps_from_excel(file_path):
    """
    Reads the well_map, drug_map, and concentration_map from an Excel file.
    
    Args:
    file_path (str): The path to the Excel file.
    
    Returns:
    tuple: A tuple containing DataFrames for well_map, drug_map, and concentration_map.
    """
    # Read the data from the Excel file
    drug_names_df = pd.read_excel(file_path, sheet_name='Drug Names', header=None)
    well_map_df = pd.read_excel(file_path, sheet_name='Well Map', header=None)
    drug_map_df = pd.read_excel(file_path, sheet_name='Drug Map', header=None)
    concentration_map_df = pd.read_excel(file_path, sheet_name='Conc Map', header=None)
    
    return drug_names_df, well_map_df, drug_map_df, concentration_map_df


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
    
    Args:
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


    
def track_cells(data_frame, mass_factor, search_radius, tracking_memory, location, folder):
    
    #Add z dimension to track the cells with mass in addition to x and y
    data_frame['z'] = data_frame['mass'] * mass_factor # z is adjusted mass with mass factor
    
    #tp.link tracks the cells using columns  ['z', 'y', 'x', 'frame'] and adds 'particle' column indicating the cell IDs
    tp.quiet() # stop printing number of trajectories
    data_frame = tp.link(data_frame, search_range=search_radius, memory=tracking_memory) # link the cells
    data_frame.drop('z', axis=1, inplace=True) # remove z column
    data_frame.insert(0, 'cell', data_frame.pop('particle')) # change column name to cell and move to the first location
    data_frame.sort_values(by=['cell', 'frame'], inplace=True) # sort tracking data by cell ID and frame number
    
    file_path_data = os.path.join(folder, f"tracking_data_loc_{location}.csv") # assign tracking data file name
    data_frame.to_csv(file_path_data, index=False) # save tracking data
    

def concat_track_files(folder_path):
    # Get list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Sort the CSV files naturally
    csv_files = natsort.natsorted(csv_files)
    
    # Initialize an empty list to hold dataframes
    dataframes = []
    
    # Load each CSV file and append to the list
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # Concatenate all dataframes while ignoring headers of subsequent files
    big_dataframe = pd.concat(dataframes, ignore_index=True)
    
    # Save the concatenated dataframe to a new CSV file
    output_path = os.path.join(folder_path, 'data_allframes.csv')
    big_dataframe.to_csv(output_path, index=False)
    
def extract_maps_from_excel(file_path, keys):
    
    # Function to find the start and end of each map
    def find_map_range(df, start_header, key_locations):
        start_index = df[df == start_header].stack().index[0][0] + 1
        start_col = df[df == start_header].stack().index[0][1]

        start_value = df.iloc[start_index, start_col]
        start_type = type(start_value)

        # Finding the end row by checking for non-empty cells in the start column and ensuring we do not cross into another key's area
        end_index = start_index
        while end_index < len(df) and not pd.isna(df.iloc[end_index, start_col]):
            current_value = df.iloc[end_index, start_col]
            if (end_index, start_col) in key_locations or type(current_value) != start_type:
                break
            end_index += 1

        # Finding the end column by checking for non-empty cells in the start row and ensuring we do not cross into another key's area
        end_col = start_col
        while end_col < df.shape[1] and not pd.isna(df.iloc[start_index, end_col]):
            current_value = df.iloc[start_index, end_col]
            if (start_index, end_col) in key_locations or type(current_value) != start_type:
                break
            end_col += 1

        return start_index, end_index, start_col, end_col

    def convert_to_float64(x):
        try:
            return np.float64(x)
        except ValueError:
            return x
        
    # Read the entire sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

    # Apply the function to the entire DataFrame
    df = df.map(convert_to_float64)

    # Find locations of all keys in the sheet
    key_locations = set()
    for key in keys:
        locations = df[df == key].stack().index.tolist()
        key_locations.update(locations)

    # Initialize a dictionary to store the DataFrames
    maps = {}

    # Loop through each key to extract the corresponding map
    for key in keys:
        start_index, end_index, start_col, end_col = find_map_range(df, key, key_locations)
        extracted_map = df.iloc[start_index:end_index, start_col:end_col].reset_index(drop=True)
        # Reset column names to default integer-based names
        extracted_map.columns = range(extracted_map.shape[1])
        maps[key] = extracted_map

    return maps