import pandas as pd
import numpy as np

def read_exp_file(file_path, sheet_name='exp_conditions', 
                  excel_key_dict=None, verbose=True):
    """
    Extract and process data from an Excel file, returning combined data structures.

    Parameters
    ----------
    file_path : str
        The path to the Excel file.
    sheet_name : str, optional
        The name of the sheet to read from (default is 'exp_conditions').
    excel_key_dict : dict, optional
        Dictionary where the keys are consistent and the values correspond to tuples of keys in the Excel file
        and the corresponding add_to_start offsets.
        Default is:
            {
                'map': ('map', (1, 1)),
                'map_keys': ('map_keys', (1, 1)),
                'experiment_name': ('experiment_name', (1, 0)),
                'experiment_folder': ('experiment_folder', (1, 0)),
                'date': ('date(MM/DD/YY)', (1, 0)),
                'dose_time': ('dose_time(hh:mm:ss)', (1, 0)),
                'treatment_conds': ('treatment_conds', (0, 1))
            }

    Returns
    -------
    tuple
        Contains three elements:
            - maps_dict : dict
                Dictionary of segmented maps based on the map keys.
            - treatment_conds_df : pd.DataFrame
                Additional treatment conditions.
            - exp_info_df : pd.DataFrame
                Additional experiment information (e.g., name, date, folder, dosing time).

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If any of the keys cannot be found or the extracted data is invalid.
    RuntimeError
        If an error occurs while reading the file.
    """
    # Default dictionary with combined Excel key and add_to_start offset if not provided
    if excel_key_dict is None:
        excel_key_dict = {
            'map': ('map', (1, 1)),
            'map_keys': ('map_keys', (1, 1)),
            'experiment_name': ('experiment_name', (1, 0)),
            'experiment_folder': ('experiment_folder', (1, 0)),
            'date': ('date(MM/DD/YY)', (1, 0)),
            'dose_time': ('dose_time(hh:mm:ss)', (1, 0)),
            'treatment_conds': ('treatment_conds', (0, 1))
        }
    
    try:
        # Attempt to read the Excel file into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except FileNotFoundError:
        # Handle case where the file does not exist
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
    except Exception as e:
        # Handle other potential errors during file reading
        raise RuntimeError(f"An error occurred while reading the file: {e}")
    
    # Extract data using both the Excel key and the add_to_start offset
    extracted_data = {}
    for key, (excel_key, add_to_start) in excel_key_dict.items():
        try:
            # Extract the data for the given key and its associated offset
            extracted_data[key] = extract_data_from_excel(df, (excel_key, add_to_start))
        except ValueError as ve:
            raise ValueError(f"Error extracting data for key '{key}': {ve}")
    
    mapped_data_dict = extracted_data

    extracted_map = mapped_data_dict.get('map')
    map_keys = mapped_data_dict.get('map_keys')
    
    if extracted_map is None or map_keys is None:
        raise ValueError("The 'map' or 'map_keys' key is missing from the extracted data.")
    
    num_of_keys = len(map_keys)
    
    plate_length = extracted_map.shape[1]
    plate_height = get_plate_height(df)
    plate_shape = [plate_height, plate_length]
    
    validate_map_shape(extracted_map, plate_shape, num_of_keys)  # Validate map dimensions
    
    maps_dict = split_maps(extracted_map, map_keys, plate_shape)  # Split the map into segments
    
    # Process additional drug condition data
    treatment_conds_df = mapped_data_dict.get('treatment_conds')
    if treatment_conds_df is None:
        raise ValueError("The 'treatment_conds' key is missing from the extracted data.")
        
    # Assuming that the first row is the treatment condition names, assign them to the column names
    treatment_conds_df.columns = treatment_conds_df.iloc[0]  # Set column names from the first row
    treatment_conds_df = treatment_conds_df[1:]  # Remove the first row from the DataFrame
    treatment_conds_df.reset_index(drop=True, inplace=True)  # Reset index

    # Initialize the DataFrame for additional experiment information
    exp_info_df = pd.DataFrame({
        'plate_shape': [plate_shape]
    })
    for key, (excel_key, _) in excel_key_dict.items():
        if key not in ['map', 'map_keys', 'treatment_conds', 'plate_shape']:
            # Concatenate the DataFrame for each key into exp_info_df
            temp_df = mapped_data_dict[key]
            
            # Check if temp_df is None before trying to modify it
            if temp_df is None:
                raise ValueError(f"Data for key '{key}' is missing or extraction failed.")
            
            temp_df.columns = [key]  # Prefix columns with key
            if key == 'date':
                temp_df[key] = pd.to_datetime(temp_df[key]).dt.strftime('%m/%d/%y')
            exp_info_df = pd.concat([exp_info_df, temp_df], axis=1, join='outer')

    if verbose:
        for key, df in maps_dict.items():
            print(f"{key}:")
            print(df.to_string(index=False, header=False))
            print()  # Add a blank line after each map
        print(treatment_conds_df.to_string(index=False))
        print()  # Add a blank line
        print(exp_info_df.to_string(index=False))    

    return maps_dict, treatment_conds_df, exp_info_df


def extract_data_from_excel(df, key_with_offset):
    """
    Extract data for the specified key from the DataFrame using the given offset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to extract data from.
    key_with_offset : tuple
        A tuple where the first element is the key to search for in the DataFrame,
        and the second element is a tuple specifying the offset to apply.

    Returns
    -------
    pd.DataFrame
        Extracted data for the given key.

    Raises
    ------
    ValueError
        If the extracted data for the key is empty, the key is not found, or the data contains only NaN values.
    """
    # Unpack the key and its corresponding offset
    key, add_to_start = key_with_offset
    
    # Find the range of the map associated with the key, applying the offset
    start_row, end_row, start_col, end_col = find_map_range(df, key, add_to_start)
    
    # Validate that the key was found and has data associated with it
    if start_row is None or end_row is None or start_col is None or end_col is None:
        raise ValueError(f"Failed to extract data for key '{key}'. Key not found in the DataFrame.")
    
    # Extract the data within the identified range and reset column names
    data = df.iloc[start_row:end_row, start_col:end_col].reset_index(drop=True)
    data.columns = range(data.shape[1])  # Reset column names to start from 0
    
    # Check if the extracted data is empty or contains only NaN values
    if data.empty or data.isna().all().all():
        raise ValueError(f"Extracted data for key '{key}' is empty or contains only NaN values.")
    
    return data

def get_plate_height(df, key='map', add_to_start=(1, 0)):
    """
    Determines the height of a plate based on the number of consecutive rows 
    containing single letter labels (e.g., 'A', 'B', etc.) in a specified column 
    of a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the plate map and associated data.
    
    key : str, optional
        The key (default is 'map') to locate the starting point in the DataFrame. 
        The function will search for the first occurrence of this key.
    
    add_to_start : tuple of int, optional
        A tuple (add_to_row, add_to_col) specifying the number of rows and columns 
        to add to the starting point found by the key to locate the row indexing 
        of the plate (default is (1, 0)).
    
    Returns:
    --------
    height : int
        The number of consecutive rows in the specified column containing single 
        letter labels (representing well labels).
    """
    
    # Find the starting position of the key in the DataFrame
    key_start_row, key_start_col = df[df == key].stack().index[0]
    
    # Extract the column starting from the determined row and column
    df_rows = df.iloc[key_start_row + add_to_start[0]:, key_start_col + add_to_start[1]]
    
    # Filter to keep only non-NaN values and those that are single letters
    df_rows = df_rows[~pd.isna(df_rows) & df_rows.str.match(r'^[A-Za-z]$')]
    
    # Calculate the height, i.e., the number of rows with single letters
    height = len(df_rows)
    
    return height


def find_map_range(df, key, add_to_start=(1, 1)):
    """
    Find the range of the map associated with the given key in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to search in.
    key : str
        The key to locate within the DataFrame.
    add_to_start : tuple, optional
        Offset to apply to the found key's location, by default (1, 1).

    Returns
    -------
    tuple
        Starting and ending rows and columns of the map as (start_row, end_row, start_col, end_col).
        Returns (None, None, None, None) if the key is not found.

    Raises
    ------
    IndexError
        If the key is not found in the DataFrame.
    """
    try:
        # Locate the starting position of the key
        start_row, start_col = df[df == key].stack().index[0]
        start_row += add_to_start[0]  # Apply row offset
        start_col += add_to_start[1]  # Apply column offset
    
        # Determine the end row based on non-empty cells below the starting row
        end_row = start_row
        while end_row < df.shape[0] and not pd.isna(df.iloc[end_row, start_col]):
            end_row += 1
    
        # Determine the end column based on non-empty cells to the right of the starting column
        end_col = start_col
        while end_col < df.shape[1] and not pd.isna(df.iloc[start_row, end_col]):
            end_col += 1
    
        # Ensure end_row and end_col advance by at least one position to cover the map range
        if end_col == start_col:
            end_col += 1
        if end_row == start_row:
            end_row += 1
            
        return start_row, end_row, start_col, end_col

    except IndexError:
        # If the key is not found, return None values and print a message
        print(f"Key '{key}' not found in DataFrame.")
        return None, None, None, None
   

def split_maps(extracted_map, map_keys, plate_shape):
    """
    Split the extracted map into separate maps based on the map keys and plate shape.
    
    Parameters
    ----------
    extracted_map : pd.DataFrame
        The DataFrame containing the extracted map.
    map_keys : pd.DataFrame
        The DataFrame containing the map keys.
    plate_shape : list
        The shape of the plate as [height, width].
    
    Returns
    -------
    dict
        A dictionary where each key corresponds to a segment of the map. The keys are derived from `map_keys`,
        and the values are DataFrames representing the segmented maps.
    
    Raises
    ------
    ValueError
        If the number of keys does not match the number of segments or if the plate shape is invalid.
    """
    num_of_keys = len(map_keys)
    maps_dict = {}
    
    # Determine the row indices for each key's segment and extract the corresponding array segment
    for i in range(num_of_keys):
        row_indices = np.arange(i, extracted_map.shape[0], num_of_keys)
        array_segment = extracted_map.iloc[row_indices, :].reset_index(drop=True)
        maps_dict[map_keys.iloc[i, 0]] = array_segment

    return maps_dict

    
def validate_map_shape(extracted_map, plate_shape, num_of_keys):
    """
    Validate the shape of the extracted map against the expected dimensions.
    
    Parameters
    ----------
    extracted_map : pd.DataFrame
        The DataFrame containing the map to be validated.
    plate_shape : list
        The expected shape of the plate as [height, width].
    num_of_keys : int
        The number of map keys expected.
    
    Raises
    ------
    ValueError
        If the dimensions of the extracted map do not match the expected height or width.
        Includes details about the expected and actual dimensions for troubleshooting.
    """
    # Calculate the expected map dimensions
    expected_height = num_of_keys * plate_shape[0]
    expected_width = plate_shape[1]

    # Validate the extracted map dimensions against the expected dimensions
    if extracted_map.shape != (expected_height, expected_width):
        raise ValueError(
            f"Map dimensions do not match expected values.\n"
            f"Expected map shape (height, width): ({expected_height}, {expected_width})\n"
            f"Current map shape (height, width): {extracted_map.shape}\n"
            f"Plate shape (height, width): {plate_shape}\n"
            f"Number of map keys: {num_of_keys}\n"
            "Please check the 'instructions' sheet in the Excel file to correctly set up the 'conditions' sheet."
        )

def get_loc_conditions(location: int, maps_dict: dict, treatment_conds_df=None):
    """
    Retrieve conditions for a specific location from the provided maps and treatment DataFrame.
    
    Parameters
    ----------
    location : int
        The location number to retrieve conditions for.
    maps_dict : dict
        A dictionary containing dataframes with well numbers, imaging locations, and other maps.
    treatment_conds_df : pd.DataFrame, optional
        A DataFrame containing additional treatment conditions. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the conditions for the specified location.
    
    Raises
    ------
    ValueError
        If the location number exceeds the maximum available location.
    KeyError
        If required columns are missing in `maps_dict` or `treatment_conds_df`.
    TypeError
        If any input parameter is of the incorrect type.
    """
    
    # Validate inputs
    if not isinstance(location, (int, np.integer)):
        raise TypeError("The location parameter must be an integer.")
    if not isinstance(maps_dict, dict):
        raise TypeError("The maps_dict parameter must be a dictionary.")
    if treatment_conds_df is not None and not isinstance(treatment_conds_df, pd.DataFrame):
        raise TypeError("The treatment_df parameter must be a pandas DataFrame.")
    if location <= 0:
        raise ValueError(f"The location parameter must be a positive integer. Given value: {location}")
    
    required_map_keys = {'well_num', 'image_locs'}
    if not required_map_keys.issubset(maps_dict.keys()):
        missing_keys = required_map_keys - maps_dict.keys()
        raise KeyError(f"maps_dict is missing the following required keys: {', '.join(missing_keys)}")
    
    if treatment_conds_df is not None:
        if 'treatment' not in treatment_conds_df.columns:
            raise KeyError("treatment_df is missing the 'treatment' column.")
    
    # Flatten the well_nums
    flattened_well_nums = maps_dict['well_num'].values.flatten()
    
    
    is_digit = [isinstance(item, (int, float)) for item in flattened_well_nums]
    integer_well_nums =  flattened_well_nums[is_digit]
    
    flattened_image_locs = maps_dict['image_locs'].values.flatten()[is_digit]

    # Flatten the rest of the maps in maps_dict
    flattened_maps = {}
    for key, df in maps_dict.items():
        if key not in ['well_num', 'image_locs']:
            flattened_maps[key] = df.values.flatten()[is_digit]
    
    # Sort the well numbers and apply the same sorting to the image locations
    sorted_indices = np.argsort(integer_well_nums)
    sorted_well_nums = integer_well_nums[sorted_indices]
    sorted_image_locs = flattened_image_locs[sorted_indices]

    # Sort the other flattened maps according to sorted well numbers
    for key in flattened_maps.keys():
        flattened_maps[key] = flattened_maps[key][sorted_indices]

    # Initialize array to store ending locations
    ending_locs = np.zeros(len(sorted_well_nums), dtype=int)

    # Fill the ending_locs array with cumulative sum of image locations
    current_location = 0
    for i, num_locs in enumerate(sorted_image_locs):
        current_location += num_locs
        ending_locs[i] = current_location

    # Find the smallest ending location that is >= the given location
    index = np.searchsorted(ending_locs, location, side='left')
    if index >= len(ending_locs):
        raise ValueError(f"Location number exceeds the maximum available location. Given value: {location}")

    # Create a DataFrame to hold the conditions for the current location
    location_conds_df = pd.DataFrame({
        'location': [location],
        'well': [sorted_well_nums[index]]
    })

    # Add other maps from sorted flattened maps directly to the DataFrame
    for key, flattened_data in flattened_maps.items():
        location_conds_df[key] = [flattened_data[index]]

    if treatment_conds_df is not None:
        treatment = location_conds_df.get('treatment', [None])[0]
        if treatment:
            # Find the matching condition in the additional conditions DataFrame
            logic = treatment_conds_df['treatment'] == treatment
            if logic.any():  # Check if there are any matching treatments
                for column in treatment_conds_df.columns:
                    if column != 'treatment':
                        location_conds_df[column] = treatment_conds_df.loc[logic, column].reset_index(drop=True)
            else:
                # Handle case where treatment does not match any in the DataFrame
                for column in treatment_conds_df.columns:
                    if column != 'treatment':
                        location_conds_df[column] = np.nan
        else:
            # If treatment is None, add NaNs for all additional conditions
            for column in treatment_conds_df.columns:
                if column != 'treatment':
                    location_conds_df[column] = np.nan
                    
    return location_conds_df




if __name__ == '__main__':
    
    file_path=r"C:\Users\shukr\Desktop\NP_files_for_Python\plateMap_v2_empty.xlsx"
    
    maps_dict, treatment_conds_df, exp_info_df = read_exp_file(file_path, verbose=False)

    loc = 40
    
    location_conds_df = get_loc_conditions(loc, maps_dict, treatment_conds_df=treatment_conds_df)

    print(location_conds_df)


















