import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
import numpy as np
import os


def make_sgr_map(well_sgr_df, plate, excel_file_path):
    """
    Maps SGR values from a well_sgr_df DataFrame to a plate DataFrame, modifies the plate
    with lettered rows and numbered columns, and saves the updated plate as a new sheet in an existing Excel file.

    Parameters:
    well_sgr_df (pd.DataFrame): A DataFrame containing two columns: 'well' (well number) and 'sgr' (sgr value).
    plate (pd.DataFrame): A DataFrame representing the experimental plate with well numbers.
    excel_file_path (str): The file path to the existing Excel file where the new sheet will be saved.
    
    """
    
    # Enable future behavior in Pandas to avoid warnings about downcasting
    pd.set_option('future.no_silent_downcasting', True)
    
    # Create a mapping dictionary from well numbers to SGR values
    mapping_dict = dict(zip(well_sgr_df['well'], well_sgr_df['sgr']))
    
    # Replace well numbers in the plate with their corresponding SGR values using the mapping dictionary
    plate_with_sgrs = plate.replace(mapping_dict)
    
    # Modify the row indices to be capital letters (A, B, C, ...)
    plate_with_sgrs.index = [chr(i) for i in range(65, 65 + plate_with_sgrs.shape[0])]

    # Set the column indices to be numbers starting from 1
    plate_with_sgrs.columns = list(range(1, plate_with_sgrs.shape[1] + 1))
    
    # Save the updated plate DataFrame as a new sheet named 'sgr_map' in the existing Excel file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
        plate_with_sgrs.to_excel(writer, sheet_name='sgr_map', index=True)
        

def plot_sgr_violin(cell_sgr_df, treatment_conds_df=None, save_fig=False, exp_folder_path=None):
    """
    Function to generate violin plots of SGRs by concentration for each drug,
    optionally including control data from specified controls. All plots will have the same y-axis scale.

    Parameters:
    - cell_sgr_df: DataFrame containing columns ['cell', 'location', 'well', 'treatment', 'concentration', 'sgr']
    - treatment_conds_df: DataFrame containing columns ['treatment', 'concentration', 'control']. If None, controls are excluded.
    - save_fig: Boolean indicating whether to save the figure. If True, exp_folder_path should not be None
    - exp_folder_path: Root directory to check for the response_data folder and save the figure. Should not be None if save_fig=True
    """

    if save_fig and exp_folder_path is None:
        raise ValueError("exp_folder_path must be specified if save_fig is True")

    # Set the Matplotlib environment to SVG
    set_matplotlib_formats('svg')

    # Initialize control_data as an empty DataFrame
    control_data = pd.DataFrame()

    if treatment_conds_df is not None:
        # Identify control treatments (where 'control' column is 'Yes')
        control_treatments = treatment_conds_df[treatment_conds_df['control'] == 'Yes']['treatment'].unique()

        # Extract control data for assigned controls
        control_data = []
        for control in control_treatments:
            temp_data = cell_sgr_df[cell_sgr_df['treatment'] == control].copy()
            temp_data['concentration'] = f'Control_{control}'  # Label as control for plotting
            temp_data['type'] = 'Control'
            control_data.append(temp_data)
        if control_data:
            control_data = pd.concat(control_data)

    # Find global min and max SGR values to set y scale
    global_min = cell_sgr_df['sgr'].min() - 0.02
    global_max = cell_sgr_df['sgr'].max() + 0.02

    # Generate violin plots for each drug
    drugs = cell_sgr_df['treatment'].unique()

    # Exclude control treatments from individual plots if treatment_conds_df is provided
    if treatment_conds_df is not None:
        drugs = [drug for drug in drugs if drug not in control_treatments]

    for drug in drugs:
        drug_data = cell_sgr_df[cell_sgr_df['treatment'] == drug].copy()
        drug_data['type'] = 'Drug'

        # Concatenate control data if available
        if not control_data.empty:
            plot_data = pd.concat([control_data, drug_data])
        else:
            plot_data = drug_data

        plt.figure(figsize=(12, 6))
        sns.violinplot(x='concentration', y='sgr', data=plot_data, hue='type', split=False, 
                       palette={'Control': 'blue', 'Drug': 'red'} if not control_data.empty else {'Drug': 'red'})
        plt.axhline(0, color='green', linestyle='--')  # Add a horizontal line at y=0
        plt.xlabel('Concentration')
        plt.ylabel('SGR')
        plt.title(f'Violin Plot of SGR by Concentration for Drug {drug}')
        plt.ylim(global_min, global_max)

        if not control_data.empty:
            plt.legend(title='Type', loc='best')

        if save_fig:
            # Define the directory to save the figure
            save_dir = os.path.join(exp_folder_path, 'response_data')
            os.makedirs(save_dir, exist_ok=True)

            plt.savefig(os.path.join(save_dir, f'Violin_drug_{drug}.svg'), format='svg')

        plt.show()


def hill_function(C, E_0, E_inf, EC50, H):
    """
    Hill function to model the concentration-response relationship.
    
    Parameters:
    - C: Concentration
    - E_0: Baseline effect (minimum effect)
    - E_inf: Maximum effect
    - EC50: Concentration of the drug that gives half-maximal response
    - H: Hill coefficient (slope)
    
    Returns:
    - Effect size as a function of concentration.
    """
    try:
        return E_0 + (E_inf - E_0) / (1 + (C / EC50)**(-H))
    except Exception as e:
        print(f"Error in hill_function: {e} with EC50={EC50}, H={H}, C={C}")
        raise

def constant_function(C, a):
    """
    Constant function y = a to model the concentration-response relationship.
    """
    return np.full_like(C, a)

def calculate_rss(y_true, y_pred):
    """
    Calculate the residual sum of squares (RSS).
    """
    return np.sum((y_true - y_pred) ** 2)

def sgr_hill_params(well_sgr_df):
    """
    Function to fit both the Hill function and a constant model (horizontal line)
    to growth rates for each drug and return a DataFrame with the fit parameters.
    
    Parameters:
    - well_sgr_df: DataFrame containing columns ['well', 'treatment', 'concentration', 'sgr']
    
    Returns:
    - params_df: DataFrame with columns ['Drug', 'Model', 'E_0', 'E_inf', 'EC50', 'H', 'DoR', 'RSS']
    """
    
    # Initialize a list to store the fit parameters
    fit_params = []

    # Get the unique treatments (drugs)
    treatments = well_sgr_df['treatment'].unique()

    for treatment in treatments:
        # Filter data for the current treatment
        drug_data = well_sgr_df[well_sgr_df['treatment'] == treatment]

        # Filter out zero concentrations for fitting purposes
        non_zero_data = drug_data[drug_data['concentration'] > 0]
        if non_zero_data.empty:
            print(f"Skipping drug {treatment} due to no non-zero concentrations.")
            continue

        concentrations = non_zero_data['concentration'].values
        mean_growth_rates = non_zero_data['sgr'].values
        
        # Fit the Hill function
        try:
            initial_guesses = [max(mean_growth_rates), min(mean_growth_rates), np.median(concentrations), 1]
            popt_hill, _ = curve_fit(hill_function, concentrations, mean_growth_rates, p0=initial_guesses, maxfev=10000)
            
            # Predict using the Hill function
            hill_pred = hill_function(concentrations, *popt_hill)
            hill_rss = calculate_rss(mean_growth_rates, hill_pred)
            
            # Calculate DoR for the Hill function
            DoR = (popt_hill[0] - popt_hill[1]) / popt_hill[0]  # (E_0 - E_inf) / E_0
            
        except RuntimeError:
            print(f"Could not fit Hill function for treatment {treatment}")
            popt_hill = [None, None, None, None]
            hill_rss = np.inf  # Assign a large RSS if the fit fails
            DoR = None

        # Fit the constant model (y = a)
        a_constant = np.mean(mean_growth_rates)
        constant_pred = constant_function(concentrations, a_constant)
        constant_rss = calculate_rss(mean_growth_rates, constant_pred)

        # Compare RSS of Hill function and constant model
        if constant_rss < hill_rss:
            # If the constant model is better, use it
            fit_params.append([treatment, 'Constant', a_constant, None, None, None, None, constant_rss])
        else:
            # Otherwise, use the Hill function
            fit_params.append([treatment, 'Hill', *popt_hill, DoR, hill_rss])
    
    # Create a DataFrame from the fit parameters
    params_df = pd.DataFrame(fit_params, columns=['Drug', 'Model', 'E_0', 'E_inf', 'EC50', 'H', 'DoR', 'RSS'])
    
    return params_df

def plot_response_curves(well_sgr_df, params_df):
    """
    Plot the SGR points from well_sgr_df along with the corresponding Hill function or flat fit from params_df.
    All treatments will be plotted on the same figure using different colors.
    
    Parameters:
    - well_sgr_df: DataFrame containing columns ['well', 'treatment', 'concentration', 'sgr'].
    - params_df: DataFrame containing columns ['Drug', 'Model', 'E_0', 'E_inf', 'EC50', 'H', 'DoR', 'RSS'].
    """
    
    # Set up color palette for multiple drugs
    palette = sns.color_palette("Set1", len(params_df['Drug'].unique()))

    # Create a figure for plotting
    plt.figure(figsize=(10, 8))

    # Iterate through each drug
    for i, treatment in enumerate(params_df['Drug'].unique()):
        # Get the SGR points for the current treatment
        drug_data = well_sgr_df[well_sgr_df['treatment'] == treatment]
        concentrations = drug_data['concentration'].values
        sgr_values = drug_data['sgr'].values
        
        # Get the corresponding model parameters from params_df
        drug_params = params_df[params_df['Drug'] == treatment].iloc[0]  # Get the first row for this drug

        # Plot the SGR data points
        plt.scatter(concentrations, sgr_values, label=f'{treatment} (data)', marker='o', color=palette[i])

        # If the model is a Hill function, plot the fitted Hill curve
        if drug_params['Model'] == 'Hill':
            E_0, E_inf, EC50, H = drug_params['E_0'], drug_params['E_inf'], drug_params['EC50'], drug_params['H']
            
            # Generate a range of concentrations for plotting the Hill curve
            conc_range = np.logspace(np.log10(min(concentrations)), np.log10(max(concentrations)), 100)
            
            # Calculate the Hill function values over the range
            hill_fit = hill_function(conc_range, E_0, E_inf, EC50, H)
            
            # Plot the Hill function curve
            plt.plot(conc_range, hill_fit, label=f'{treatment} (Hill fit)', linestyle='--', color=palette[i])

        # If the model is a flat fit, plot the flat line (constant value)
        elif drug_params['Model'] == 'Flat':
            a = drug_params['E_0']  # In flat fit, E_0 represents the constant value
            
            # Plot the flat fit as a horizontal line over the concentration range
            plt.plot([min(concentrations), max(concentrations)], [a, a], label=f'{treatment} (Flat fit)', linestyle='--', color=palette[i])

    # Add horizontal line at SGR = 0
    # plt.axhline(y=0, color='green', linestyle='--', linewidth=1, label='SGR = 0')
   
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add labels, title, and legend
    plt.xlabel('Concentration')
    plt.ylabel('SGR')
    plt.title('SGR Response Curves and Hill/Flat Fits for All Treatments')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show the plot
    plt.show()
