import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from scipy.optimize import curve_fit
from scipy.stats import linregress
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
        

def plot_violins(cell_sgr_df, treatment_conds_df=None, save_fig=False, exp_folder_path=None, data_folder=None):
    """
    Function to generate violin plots of SGRs by concentration for each drug,
    optionally including control data from specified controls. All plots will have the same y-axis scale.

    Parameters:
    - cell_sgr_df: DataFrame containing columns ['cell', 'location', 'well', 'treatment', 'concentration', 'sgr']
    - treatment_conds_df: DataFrame containing columns ['treatment', 'concentration', 'control']. If None, controls are excluded.
    - save_fig: Boolean indicating whether to save the figure. If True, exp_folder_path should not be None
    - exp_folder_path: Root directory to check for the response_data folder and save the figure. Should not be None if save_fig=True
    """

    if save_fig and exp_folder_path is None or data_folder is None:
        raise ValueError("exp_folder_path and data_folder must be specified if save_fig is True")

    set_matplotlib_formats('svg')

    control_data = pd.DataFrame()

    if treatment_conds_df is not None:
        control_treatments = treatment_conds_df[treatment_conds_df['control'] == 'Yes']['treatment'].unique()
        control_data = []
        for control in control_treatments:
            temp_data = cell_sgr_df[cell_sgr_df['treatment'] == control].copy()
            temp_data['concentration'] = f'Ctrl_{control}' 
            temp_data['type'] = 'Control'
            control_data.append(temp_data)
        if control_data:
            control_data = pd.concat(control_data)

    global_min = cell_sgr_df['sgr'].min() - 0.02
    global_max = cell_sgr_df['sgr'].max() + 0.02

    drugs = cell_sgr_df['treatment'].unique()

    if treatment_conds_df is not None:
        drugs = [drug for drug in drugs if drug not in control_treatments]

    for drug in drugs:
        drug_data = cell_sgr_df[cell_sgr_df['treatment'] == drug].copy()
        drug_data['type'] = 'Drug'

        if not control_data.empty:
            plot_data = pd.concat([control_data, drug_data])
        else:
            plot_data = drug_data

        plot_data['str_concentration'] = plot_data['concentration'].astype(str)

        # Define the custom order for the x-axis based on the unique concentration values
        concentration_order = plot_data['str_concentration'].unique()

        plt.figure(figsize=(12, 8))


        control_data_plot = plot_data[plot_data['type'] == 'Control']
        if not control_data_plot.empty:
            sns.violinplot(x='str_concentration', y='sgr', data=control_data_plot, color='blue', inner='box',
                           order=concentration_order, linewidth=1)

        drug_data_plot = plot_data[plot_data['type'] == 'Drug']
        sns.violinplot(x='str_concentration', y='sgr', data=drug_data_plot, color='red', inner='box',
                       order=concentration_order, linewidth=1)

        # Add horizontal line at y=0
        plt.axhline(0, color='green', linestyle='--')
        plt.xlabel('Concentration')
        plt.ylabel('SGR')
        plt.title(f'Drug {drug}')
        plt.ylim(global_min, global_max)


        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='blue', lw=4, label='Control'),
                           Line2D([0], [0], color='red', lw=4, label='Drug')]
        plt.legend(handles=legend_elements, title='Type')

        if save_fig:
            save_dir = os.path.join(exp_folder_path, 'data_folder')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'Violin_drug_{drug}.svg'), format='svg')

        plt.show()

def constant_function(C, a):
    """
    Constant function y = a to model the response.
    """
    return a + 0. * C

def hill_function(C, E_0, E_inf, EC50, H):
    """
    Hill function to model the concentration-response relationship.
    """
    try:
        return E_0 + (E_inf - E_0) / (1 + (C / EC50)**(-H))
    except Exception as e:
        print(f"Error in hill_function with parameters: E_0={E_0}, E_inf={E_inf}, EC50={EC50}, H={H}")
        print(f"Concentrations: {C}")
        raise e  # Re-raise the exception after logging details

def flat_fit(concentrations, y_data):
    """
    Fits a constant value to the data (a flat fit) using curve_fit.
    
    Parameters:
    - concentrations: Array of concentrations (independent variable).
    - y_data: Array of SGR values (dependent variable).
    
    Returns:
    - popt: Fitted parameter (constant value 'a').
    - pcov: Covariance of the fitted parameter.
    """
    initial_guess = np.mean(y_data)
    
    # Bounds for the constant value
    lower_bound = np.min(y_data) - 2 * np.ptp(y_data)  # min(y_data) - range * 2
    upper_bound = np.max(y_data) + 2 * np.ptp(y_data)  # max(y_data) + range * 2

    try:
        popt, pcov = curve_fit(constant_function, concentrations, y_data,
                               p0=[initial_guess], bounds=([lower_bound], [upper_bound]), maxfev=10000)
    except Exception as e:
        print(f"Error during flat_fit with concentrations: {concentrations}, y_data: {y_data}")
        raise e
    
    return popt, pcov

def hill_fit(concentrations, sgr_values):
    """
    Fits the Hill function to the data using curve_fit with parameter bounds.
    
    Parameters:
    - concentrations: Array of concentrations.
    - sgr_values: Array of SGR values.
    
    Returns:
    - popt: Optimized parameters for the Hill function.
    - pcov: Covariance of the optimized parameters.
    """
    # Initial guesses for E_0, E_inf, EC50, H
    initial_guesses = [max(sgr_values), min(sgr_values), np.median(concentrations), 1]
    
    # Define bounds for the parameters
    bounds = (
        [min(sgr_values), min(sgr_values), 1e-9, 0.1],  # Lower bounds: E_0, E_inf, EC50, H
        [max(sgr_values), max(sgr_values), max(concentrations), 10]  # Upper bounds: E_0, E_inf, EC50, H
    )

    try:
        popt, pcov = curve_fit(hill_function, concentrations, sgr_values, 
                               p0=initial_guesses, bounds=bounds, maxfev=10000)
    except Exception as e:
        print(f"Error during hill_fit with concentrations: {concentrations}, sgr_values: {sgr_values}")
        raise e
    
    return popt, pcov

def calculate_p_value(y_true, y_pred):
    """
    Calculate the p-value for goodness of fit using linregress.
    
    Parameters:
    - y_true: Actual SGR values.
    - y_pred: Predicted SGR values from the fit.
    
    Returns:
    - p_value: p-value indicating goodness of fit.
    """
    _, _, _, p_value, _ = linregress(y_true, y_pred)
    return p_value

def EC50_params(well_sgr_df, p_cutoff=0.05):
    """
    Function to fit the Hill function to growth rates for each drug and return a DataFrame with the fit parameters.
    If the Hill function fit has a p-value greater than the cutoff, it fits a flat model instead.

    Parameters:
    - well_sgr_df: DataFrame containing columns ['well', 'treatment', 'concentration', 'sgr']
    - p_cutoff: The cutoff for the p-value. If the Hill fit's p-value exceeds this, flat_fit is used.
    
    Returns:
    - params_df: DataFrame with columns ['Drug', 'Model', 'E_0', 'E_inf', 'EC50', 'H', 'DoR']
    """
    
    
    fit_params = []

    # Get the unique treatments (drugs)
    treatments = well_sgr_df['treatment'].unique()

    for treatment in treatments:
        # Filter data for the current treatment
        drug_data = well_sgr_df[well_sgr_df['treatment'] == treatment]
        concentrations = drug_data['concentration'].values
        sgr_values = drug_data['sgr'].values

        # Remove zero concentrations
        non_zero_indices = concentrations > 0
        concentrations = concentrations[non_zero_indices]
        sgr_values = sgr_values[non_zero_indices]

        if len(concentrations) == 0:
            print(f"Skipping drug {treatment} due to no non-zero concentrations.")
            continue

        # Fit the Hill function
        try:
            popt_hill, _ = hill_fit(concentrations, sgr_values)
            hill_pred = hill_function(concentrations, *popt_hill)
            hill_p_value = calculate_p_value(sgr_values, hill_pred)

        except Exception as e:
            print(f"Error in fitting Hill function for treatment {treatment}: {e}")
            popt_flat, _ = flat_fit(concentrations, sgr_values)
            fit_params.append([treatment, 'Flat', popt_flat[0], None, None, None, None])
            continue
        
        # If the Hill fit p-value exceeds the cutoff, use flat_fit instead
        if hill_p_value > p_cutoff:
            popt_flat, _ = flat_fit(concentrations, sgr_values)
            fit_params.append([treatment, 'Flat', popt_flat[0], None, None, None, None])
        else:
            E_0, E_inf, EC50, H = popt_hill
            DoR = (E_0 - E_inf) / E_0  # Degree of Response
            fit_params.append([treatment, 'Hill', E_0, E_inf, EC50, H, DoR])
    
    # Create a DataFrame from the fit parameters
    params_df = pd.DataFrame(fit_params, columns=['Drug', 'Model', 'E_0', 'E_inf', 'EC50', 'H', 'DoR'])
    
    return params_df

def plot_EC50_curves(conc_sgr_df, params_df, save_fig=False, exp_folder_path=None, data_folder=None):
    """
    Plot the SGR points from well_sgr_df along with the corresponding Hill function or flat fit from params_df.
    All treatments will be plotted on the same figure using different colors.
    
    Parameters:
    - conc_sgr_df: DataFrame containing columns ['treatment', 'concentration', 'sgr', 'std_dev'].
    - params_df: DataFrame containing columns ['Drug', 'Model', 'E_0', 'E_inf', 'EC50', 'H', 'DoR']. This is the output of EC50_params().
    """
    
    if save_fig and exp_folder_path is None or data_folder is None:
        raise ValueError("exp_folder_path and data_folder must be specified if save_fig is True")
    
    # Set the Matplotlib environment to SVG
    set_matplotlib_formats('svg')
    
    # Set up color palette for multiple drugs
    palette = sns.color_palette("Set1", len(params_df['Drug'].unique()))

    # Create a figure for plotting
    plt.figure(figsize=(10, 8))

    # Iterate through each drug
    for i, treatment in enumerate(params_df['Drug'].unique()):
        # Get the SGR points for the current treatment
        drug_data = conc_sgr_df[conc_sgr_df['treatment'] == treatment]
        concentrations = drug_data['concentration'].values
        sgr_values = drug_data['sgr'].values
        std_dev_values = drug_data['std_dev'].values  # Standard deviation for error bars
        
        # Get the corresponding model parameters from params_df
        drug_params = params_df[params_df['Drug'] == treatment].iloc[0]  # Get the first row for this drug

        # Plot the SGR data points with error bars
        plt.errorbar(concentrations, sgr_values, yerr=std_dev_values, label=f'{treatment} (data)', 
                     fmt='o', color=palette[i], capsize=5, linestyle='None')  # Add error bars with caps

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

    # Add horizontal line for control
    # plt.axhline(y=conc_sgr_df[conc_sgr_df['treatment'] == 'Untreated']['sgr'].item(), color='green', linestyle='--', linewidth=1, label='Control')
   
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add labels, title, and legend
    plt.xlabel('Concentration')
    plt.ylabel('SGR')
    plt.title('SGR Response Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_fig:
        save_dir = os.path.join(exp_folder_path, data_folder)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'EC50.svg'), format='svg')
        
    # Show the plot
    plt.show()
    
# def plot_EC50_curves(conc_sgr_df, params_df, save_fig=False, exp_folder_path=None, data_folder=None):
#     """
#     Plot the SGR points from well_sgr_df along with the corresponding Hill function or flat fit from params_df.
#     All treatments will be plotted on the same figure using different colors.
    
#     Parameters:
#     - conc_sgr_df: DataFrame containing columns ['treatment', 'concentration', 'sgr', 'std_dev'].
#     - params_df: DataFrame containing columns ['Drug', 'Model', 'E_0', 'E_inf', 'EC50', 'H', 'DoR']. This is the output of EC50_params().
#     """
    
#     if save_fig and exp_folder_path is None or data_folder is None:
#         raise ValueError("exp_folder_path and data_folder must be specified if save_fig is True")
#     # Set the Matplotlib environment to SVG
#     set_matplotlib_formats('svg')
    
#     # Set up color palette for multiple drugs
#     palette = sns.color_palette("Set1", len(params_df['Drug'].unique()))

#     # Create a figure for plotting
#     plt.figure(figsize=(10, 8))

#     # Iterate through each drug
#     for i, treatment in enumerate(params_df['Drug'].unique()):
#         # Get the SGR points for the current treatment
#         drug_data = conc_sgr_df[conc_sgr_df['treatment'] == treatment]
#         concentrations = drug_data['concentration'].values
#         sgr_values = drug_data['sgr'].values
        
#         # Get the corresponding model parameters from params_df
#         drug_params = params_df[params_df['Drug'] == treatment].iloc[0]  # Get the first row for this drug

#         # Plot the SGR data points
#         plt.scatter(concentrations, sgr_values, label=f'{treatment} (data)', marker='o', color=palette[i])

#         # If the model is a Hill function, plot the fitted Hill curve
#         if drug_params['Model'] == 'Hill':
#             E_0, E_inf, EC50, H = drug_params['E_0'], drug_params['E_inf'], drug_params['EC50'], drug_params['H']
            
#             # Generate a range of concentrations for plotting the Hill curve
#             conc_range = np.logspace(np.log10(min(concentrations)), np.log10(max(concentrations)), 100)
            
#             # Calculate the Hill function values over the range
#             hill_fit = hill_function(conc_range, E_0, E_inf, EC50, H)
            
#             # Plot the Hill function curve
#             plt.plot(conc_range, hill_fit, label=f'{treatment} (Hill fit)', linestyle='--', color=palette[i])

#         # If the model is a flat fit, plot the flat line (constant value)
#         elif drug_params['Model'] == 'Flat':
#             a = drug_params['E_0']  # In flat fit, E_0 represents the constant value
            
#             # Plot the flat fit as a horizontal line over the concentration range
#             plt.plot([min(concentrations), max(concentrations)], [a, a], label=f'{treatment} (Flat fit)', linestyle='--', color=palette[i])

#     # Add horizontal line
#     plt.axhline(y=conc_sgr_df[conc_sgr_df['treatment'] == 'Untreated']['sgr'].item(), color='green', linestyle='--', linewidth=1, label='Control')
   
#     # Set log scale for x-axis
#     plt.xscale('log')
    
#     # Add labels, title, and legend
#     plt.xlabel('Concentration')
#     plt.ylabel('SGR')
#     plt.title('SGR Response Curves and Hill/Flat Fits for All Treatments')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()

#     if save_fig:
#         save_dir = os.path.join(exp_folder_path, data_folder)
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, 'EC50.svg'), format='svg')
        
#     # Show the plot
#     plt.show()
