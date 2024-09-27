import pandas as pd
import numpy as np

def read_exp_file(file_path, sheet_name='exp_conditions', 
                  keys_to_extract = ['map', 'map_keys', 'treatment_conds', 'experiment_name', 'date', 'experiment_folder', 'plate_shape'], 
                  add_to_starts = [(1,1), (1,1), (0,1), (1,0), (1,0), (1,0), (1,0)]):
    """
    Main function to extract and process data from an Excel file, returning combined data structures.
    
    Parameters:
        file_path (str): The path to the Excel file.
        keys_to_extract (list): The keys to search for in the Excel file.
        add_to_starts (list): List of offsets to apply to index of a key to get to the top left corner of the array to be read.
        
    Returns:
        tuple: Contains three elements:
            - map_dict (dict): A dictionary of segmented maps based on the map keys.
            - cond_dict (dict): A dictionary of segmented drug conditions.
            - info_dict (dict): A dictionary containing the rest of the extracted data.
    
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If any of the keys cannot be found or the extracted data is invalid.
    """
    try:
        # Attempt to read the Excel file into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except FileNotFoundError:
        # Handle case where the file does not exist
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
    except Exception as e:
        # Handle other potential errors during file reading
        raise RuntimeError(f"An error occurred while reading the file: {e}")
    
    # Extract data associated with the specified keys
    data_dict = extract_data_from_excel(df, keys=keys_to_extract, add_to_starts=add_to_starts)
    
    # Extract and process map-related data
    try:
        plate_shape = [int(x) for x in data_dict['plate_shape'].values[0][0] if str(x).isdigit()]
    except KeyError:
        raise ValueError("The 'plate_shape' key is missing from the extracted data.")
    
    extracted_map = data_dict.get('map')
    map_keys = data_dict.get('map_keys')
    
    if extracted_map is None or map_keys is None:
        raise ValueError("The 'map' or 'map_keys' key is missing from the extracted data.")
    
    num_of_keys = len(map_keys)
    validate_map_shape(extracted_map, plate_shape, num_of_keys)  # Validate map dimensions
    maps_dict = split_maps(extracted_map, map_keys, plate_shape)  # Split the map into segments
    
    # Process additional drug condition data
    add_conds_df = data_dict.get('treatment_conds')
    if add_conds_df is None:
        raise ValueError("The 'drug_conditions' key is missing from the extracted data.")
        
    # Assuming that the first row is the treatment condition names, assign them to the column names
    add_conds_df.columns = add_conds_df.iloc[0]  # Set column names from the first row
    add_conds_df = add_conds_df[1:]  # Remove the first row from the DataFrame
    add_conds_df.reset_index(drop=True, inplace=True)  # Reset index

    # Initialize the DataFrame for additional experiment information
    exp_info_df = pd.DataFrame({
        'plate_shape': [plate_shape]
    })
    for key in keys_to_extract:
        if key not in ['map', 'map_keys', 'treatment_conds', 'plate_shape']:
            # Concatenate the DataFrame for each key into info_df
            temp_df = data_dict[key]
            temp_df.columns =  temp_df.columns = [key]  # Prefix columns with key
            exp_info_df = pd.concat([exp_info_df, temp_df], axis=1, join='outer')

    return maps_dict, add_conds_df, exp_info_df

def extract_data_from_excel(df, keys=None, add_to_starts=None):
    """
    Extracts data for specified keys from the DataFrame and returns them in a dictionary.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to extract data from.
        keys (list): List of keys to search for in the DataFrame.
        add_to_starts (list): List of tuples specifying the offset to start searching for each key.
        
    Returns:
        dict: A dictionary where each key is mapped to its corresponding extracted DataFrame.
        
    Raises:
        ValueError: If the extracted data for any key is empty or the key is not found.
    """
    if not isinstance(keys, list) or not isinstance(add_to_starts, list):
        raise ValueError("Assign coorect lists of keys and add_to_starts.")
    
    data_dict = {}
    for i, key in enumerate(keys):
        # Find the range of the map associated with each key
        start_row, end_row, start_col, end_col = find_map_range(df, key, add_to_starts[i])
        
        # Validate that the key was found and has data associated with it
        if start_row is None or end_row is None or start_col is None or end_col is None:
            raise ValueError(f"Failed to extract data for key '{key}'. Key not found in the DataFrame.")
        
        # Extract the data within the identified range and reset column names
        data = df.iloc[start_row:end_row, start_col:end_col].reset_index(drop=True)
        data.columns = range(data.shape[1])  # Reset column names to start from 0
        
        # Check if the extracted data is empty or contains only NaN values
        if data.empty or data.isna().all().all():
            raise ValueError(f"Extracted data for key '{key}' is empty or contains only NaN values.")
        
        data_dict[key] = data

    return data_dict

def find_map_range(df, key, add_to_start=(1, 1)):
    """
    Finds the range of the map associated with the given key in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to search in.
        key (str): The key to locate within the DataFrame.
        add_to_start (tuple): Offset to apply to the found key's location.
    
    Returns:
        tuple: Starting and ending rows and columns of the map as (start_row, end_row, start_col, end_col).
               Returns (None, None, None, None) if the key is not found.
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
    Splits the extracted map into segments based on the map keys and plate shape.
    
    Parameters:
        extracted_map (pd.DataFrame): The DataFrame containing the extracted map.
        map_keys (pd.DataFrame): The DataFrame containing the map keys.
        plate_shape (list): The shape of the plate as [height, width].
    
    Returns:
        dict: A dictionary where each key corresponds to a segment of the map.
    """
    num_of_keys = len(map_keys)
    map_dict = {}
    
    # Determine the row indices for each key's segment and extract the corresponding array segment
    for i in range(num_of_keys):
        row_indices = np.arange(i, extracted_map.shape[0], num_of_keys)
        array_segment = extracted_map.iloc[row_indices, :].reset_index(drop=True)
        map_dict[map_keys.iloc[i, 0]] = array_segment

    return map_dict

    
def validate_map_shape(extracted_map, plate_shape, num_of_keys):
    """
    Validates the shape of the extracted map against the expected dimensions.
    
    Parameters:
        extracted_map (pd.DataFrame): The DataFrame containing the map.
        plate_shape (list): The expected shape of the plate as [height, width].
        num_of_keys (int): The number of map keys expected.
    
    Raises:
        ValueError: If the dimensions of the extracted map do not match the expected values.
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

def get_loc_conditions(location: int, map_dict: dict, add_conds_df: pd.DataFrame = None):
    """
    Retrieve the conditions for a given location from the provided maps and treatment DataFrame.

    Parameters:
        location (int): The location number to find conditions for.
        map_dict (dict): A dictionary of dataframes containing well numbers, imaging locations, and other maps with corresponding plate shape.
        treatment_df (pd.DataFrame, optional): A DataFrame containing treatment information. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the conditions for the given location.
    
    Raises:
        ValueError: If location number exceeds the maximum available location.
        KeyError: If required columns are missing in map_dict or treatment_df.
        TypeError: If inputs are of incorrect types.
    """
    
    # Validate inputs
    if not isinstance(location, int):
        raise TypeError("The location parameter must be an integer.")
    if not isinstance(map_dict, dict):
        raise TypeError("The map_dict parameter must be a dictionary.")
    if add_conds_df is not None and not isinstance(add_conds_df, pd.DataFrame):
        raise TypeError("The treatment_df parameter must be a pandas DataFrame.")
    
    required_map_keys = {'well_num', 'image_locs'}
    if not required_map_keys.issubset(map_dict.keys()):
        missing_keys = required_map_keys - map_dict.keys()
        raise KeyError(f"map_dict is missing the following required keys: {', '.join(missing_keys)}")
    
    if add_conds_df is not None:
        if 'treatments' not in add_conds_df.columns:
            raise KeyError("treatment_df is missing the 'treatments' column.")
        if 'conc_units' not in add_conds_df.columns:
            raise KeyError("treatment_df is missing the 'conc_units' column.")
    
    # Flatten the DataFrames
    flattened_well_nums = map_dict['well_num'].values.flatten()
    is_integer = [isinstance(item, int) for item in flattened_well_nums]
    integer_well_nums =  flattened_well_nums[is_integer]
    
    flattened_image_locs = map_dict['image_locs'].values.flatten()[is_integer]

    # Flatten the rest of the maps in map_dict
    flattened_maps = {}
    for key, df in map_dict.items():
        if key not in ['well_num', 'image_locs']:
            flattened_maps[key] = df.values.flatten()[is_integer]

    
    
    
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
        raise ValueError("Location number exceeds the maximum available location.")

    # Create a DataFrame to hold the conditions for the current location
    location_conds_df = pd.DataFrame({
        'location': [location],
        'well': [sorted_well_nums[index]]
    })

    # Add other maps from sorted flattened maps directly to the DataFrame
    for key, flattened_data in flattened_maps.items():
        location_conds_df[key] = [flattened_data[index]]

    if add_conds_df is not None:
        treatment = location_conds_df.get('treatment', [None])[0]
        if treatment:
            # Find the matching condition in the additional conditions DataFrame
            logic = add_conds_df['treatments'] == treatment
            if logic.any():  # Check if there are any matching treatments
                for column in add_conds_df.columns:
                    if column != 'treatments':
                        location_conds_df[column] = add_conds_df.loc[logic, column].reset_index(drop=True)
            else:
                # Handle case where treatment does not match any in the DataFrame
                for column in add_conds_df.columns:
                    if column != 'treatments':
                        location_conds_df[column] = np.nan
        else:
            # If treatment is None, add NaNs for all additional conditions
            for column in add_conds_df.columns:
                if column != 'treatments':
                    location_conds_df[column] = np.nan
    return location_conds_df




























