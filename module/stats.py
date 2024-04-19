
#my modular imports
from module.utils import *

from module.action_potential_functions import  pAD_detection
from module.getters import  getExpandedDfIfFilename, updateFPStats, updateAPPStats

#generic imports
import os, shutil, itertools, json, time, functools, pickle
import pandas as pd
import glob #new
import igor2 as igor
import numpy as np
import pandas as pd #new
from pathlib import Path #new

from matplotlib.backends.backend_pdf import PdfPages #new
import matplotlib.pyplot as plt 
import scipy
import scipy.signal as sg #new
from scipy import stats
import seaborn as sns
from statannotations.Annotator import Annotator
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from collections import Counter
#CONSTANTS   
from module.constants import CACHE_DIR, INPUT_DIR, OUTPUT_DIR, color_dict, n_minimum,  p_value_threshold



########## WORKING WITH EXPANDED DF ###########
def propagate_I_set(df):
    # Create a dictionary with 'cell_id' as keys and 'I_set' as values
    # for rows where 'data_type' is 'APP' and 'application_order' is 1
    ap_I_set = df[(df['data_type'] == 'APP') & (df['application_order'] == 1) & (~df['I_set'].isna())].set_index('cell_id')['I_set'].to_dict()
    # Apply the I_set value to rows where it's NaN, without affecting rows where 'data_type' is 'APP'
    df['I_set'] = df.apply(lambda row: ap_I_set.get(row['cell_id'], row['I_set']) if pd.isna(row['I_set']) else row['I_set'], axis=1)
    return df

def loop_stats(filename_or_df):

    df, filename = getExpandedDfIfFilename(filename_or_df)
    df = propagate_I_set(df) 
    df_row_order = df['folder_file'].tolist()  # save origional row order
    
    combinations = [
                    (["cell_type", "cell_id", "data_type"], _update_FP_stats), #APPLYING FP stats to AP files also and visa versa #TODO this is inefficient 
                    (["cell_type", "data_type", "drug"], _update_APP_stats), 
               
                    # (["cell_type", "drug", "data_type"], _plot_pAD)
    ]
    for groupby_cols, handlingFn in combinations:
        df = apply_group_by_funcs(filename, df, groupby_cols, handlingFn) #note that as each function is run the updated df is fed to the next function
    df = df.loc[df['folder_file'].isin(df_row_order)]# rearrange the DataFrame to match the original row order
    return df

def apply_group_by_funcs(filename, df, groupby_cols, handleFn): #creating a list of new values and adding them to the existign df
    res_dfs_li = [] #list of dfs
    for group_info, group_df in df.groupby(groupby_cols):
        res_df = handleFn(filename, group_info, group_df)
        res_dfs_li.append(res_df)
    new_df = pd.concat(res_dfs_li) # some functions expand expanded_df so all function must return df and only df
    return new_df

def _update_APP_stats(filename, celltype_datatype_drug, df):
    '''
    Creates a list of rows to be added to APP stats df and calles updateAPPStats()
    '''
    cell_type, data_type, drug = celltype_datatype_drug
    df=df[df['application_order'] <= 1] #remove second aplication data 
    update_rows=[]

    if data_type == 'APP':
        
        #measure == input_R / RMP
        columns_with_lists = ['inputR_PRE', 'RMP_PRE',
                            'inputR_APP', 'RMP_APP', 
                            'inputR_WASH', 'RMP_WASH']
        

        #measure == pAD_True_AP_count and AP_count 
        columns_to_count = ['PRE_AP_locs', 'PRE_pAD_locs', 
                            'APP_AP_locs', 'APP_pAD_locs', 
                            'WASH_AP_locs', 'WASH_pAD_locs'] 
        
        for index, row in df.iterrows():  # Looping over each cell_id
            cell_id = row['cell_id'] 
            ap_count_by_condition = {'PRE': 0, 'APP': 0, 'WASH': 0}  # Initialize AP counts for each condition

            for col_name in columns_to_count:  # Looping over PRE, APP, WASH
                pre_app_wash, AP_type = col_name.split('_')[0], "_".join(col_name.split('_')[1:3])
                value_len = len(row[col_name]) if isinstance(row[col_name], list) else 0
                update_row = {
                    "folder_file": row['folder_file'],
                    "cell_type": cell_type,
                    "cell_id": cell_id,
                    "measure": f"AP_count_{AP_type}",
                    "treatment": drug,
                    'protocol':row['I_set'],
                    "pre_app_wash": pre_app_wash,
                    "value": value_len,
                    "sem": np.nan
                }
                update_rows.append(update_row)
                ap_count_by_condition[pre_app_wash] += value_len  # Accumulate total AP count

            for list_col in columns_with_lists:
                measure, pre_app_wash = list_col.split('_')[0], list_col.split('_')[1]
                mean_row = {
                    "folder_file": row['folder_file'],
                    "cell_type": cell_type,
                    "cell_id": cell_id,
                    "measure": measure,
                    "treatment": drug,
                    'protocol':row['I_set'],
                    "pre_app_wash": pre_app_wash,
                    "value": safe_mean(row[list_col]),
                    "sem": safe_sem(row[list_col])
                }
                update_rows.append(mean_row)

        updateAPPStats(filename, update_rows) 
    else:
        pass
    return df

def safe_mean(arr):
    '''
    Calculate the mean of an array while handling NaN values and empty slices.
    Parameters:
        arr (array_like): Input array for which to calculate the mean.
    Returns:
        float: The mean of non-NaN values in the array, or NaN if the array is empty or contains only NaN values.
    '''
    if len(arr) <= 1 or np.all(np.isnan(arr)):
        return np.nan
    else:
        return np.nanmean(arr)
    
def safe_sem(arr):
    '''
    Calculate the standard error of the mean (SEM) of an array while handling NaN values and empty slices.
    Parameters:
        arr (array_like): Input array for which to calculate the SEM.

    Returns:
        float: The SEM of non-NaN values in the array, or NaN if the array is empty or contains only NaN values.
    '''
    if isinstance(arr, list) and len(arr) <= 1:
        return np.nan
    else:
        return stats.sem(arr, nan_policy='omit')
    

def get_common_val_PRE_DRUG(df, col_with_list, sort_by_list_idx):
    """
    Get the common 'I injected' value from two distinct groups in a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        col_with_list (str): The name of the column containing lists.
        sort_by_list_idx (int): The index of the list element to sort by.

    Returns:
        int: The most common 'I injected' value between the two groups, or np.nan if there are no overlapping values.
    """
    # Exclude rows with NaN values in the relevant column
    df_filtered = df.dropna(subset=[col_with_list])

    # Split DataFrame into two groups based on the 'drug' column
    pre_group = df_filtered[df_filtered['drug'] == 'PRE']
    other_group = df_filtered[df_filtered['drug'] != 'PRE']

    # Extract 'I injected' values for each group
    pre_injected_values = [row[col_with_list][sort_by_list_idx] for idx, row in pre_group.iterrows()]
    other_injected_values = [row[col_with_list][sort_by_list_idx] for idx, row in other_group.iterrows()]

    # Find intersection of 'I injected' values between the two groups
    common_injected_values = set(pre_injected_values).intersection(other_injected_values)

    if not common_injected_values:
        return np.nan

    # Count occurrences of each common value
    common_value_counts = Counter(common_injected_values)

    # Return the most common value as an integer
    return int(max(common_value_counts, key=common_value_counts.get))



def _update_FP_stats(filename, celltype_cellid_datatype, df):
    '''
    input: single cell FP expanded df
    output: updated FP_stats with a mean pre and post for each cell_id and measure per treatment_group (cell_type/drug)
    '''
    cell_type, cell_id, data_type = celltype_cellid_datatype
    df=df[df['application_order'] <= 1] #remove second aplication data 
    update_rows = []
    if data_type == 'FP':

        treatment = ', '.join(df[df['drug'] != 'PRE']['drug'].unique())
        for drug, pre_post_df in df.groupby('drug'):
            pre_post = 'PRE' if 'PRE' in pre_post_df['drug'].values else 'POST'
            I_setting =  pre_post_df['I_set'].iloc[0] if pre_post_df['I_set'].nunique() == 1 else "Not all values are the same."

            for measure in ['max_firing', 
                            'rheobased_threshold',
                            'FI_slope',
                            'voltage_threshold',
                            'AP_height',
                            'AP_slope',
                            'AP_width',
                            'AP_latency',
                            'AP_dvdt_max']:

                mean_value, file_values, folder_files = fetchMeans(pre_post_df, measure)
                update_row = {
                    "cell_type": cell_type,
                    "cell_id": cell_id,
                    "measure": measure,
                    "treatment": treatment,
                    'protocol': I_setting , #JJB doubt this will work as nan unless data_type=AP
                    "pre_post": pre_post,
                    "mean_value": mean_value,
                    "file_values": file_values, 
                    "folder_files": folder_files, 
                    "R_series": list(pre_post_df['R_series']),
                }
                # Append the dictionary to the list
                update_rows.append(update_row)

            for unnormalised_measure in ['tau_rc', 'sag']:
                normalised_current = get_common_val_PRE_DRUG(df, unnormalised_measure, 2)
                if pd.notna(normalised_current):
                    pre_post_df_clean = pre_post_df.dropna(subset=[unnormalised_measure])
                    idx_normalised =[idx for idx, row in pre_post_df_clean.iterrows() if row[unnormalised_measure][2] == normalised_current]
                    file_values =  list(df.loc[idx_normalised, unnormalised_measure].apply(lambda x: x[0]))
                    update_row = {
                        "cell_type": cell_type,
                        "cell_id": cell_id,
                        "measure": unnormalised_measure,
                        "treatment": treatment,
                        'protocol': I_setting , #JJB doubt this will work as nan unless data_type=AP
                        "pre_post": pre_post,
                        "mean_value": np.mean(file_values),
                        "file_values": file_values, 
                        "folder_files": list(df.loc[idx_normalised, 'folder_file']), 
                    }
                    # Append the dictionary to the list
                    update_rows.append(update_row)
                else:
                    print(f"{unnormalised_measure} for {cell_id} has no common FP current injected.")
                    

        updateFPStats(filename, update_rows)
    else:
        pass
    return df


def fetchMeans(pre_post_df, measure):
    column = pre_post_df[measure]
    folder_files = pre_post_df['folder_file']
    valid_file_values = [] # single value for each folder_file
    valid_folder_files = []
    
    for folder_file, value in zip(folder_files, column): # loop folder_files
               
        if (isinstance(value, list) and len(value) == 0) or (isinstance(value, (float, np.floating)) and np.isnan(value)):  # if the value is empty list or NaN   
            print(f"Missing {measure} data in folder_file: {folder_file}")
            valid_file_values.append(np.nan)
            valid_folder_files.append(np.nan)
       
        else:  #add mean or float to valid_file_values
            valid_file_values.extend([np.mean(value)] if isinstance(value, list) else [value])
            valid_folder_files.append(folder_file)
   
    mean_value = np.mean(valid_file_values)  # mean of all files
    return mean_value, valid_file_values, valid_folder_files


######## CALCULATE / PLOT STATS
def add_statistical_annotation_hues(ax, data, x, y, group, x_order, hue_order, test='paired_ttest', p_threshold=0.05):
    '''
    Performs paired t-tests between groups for each x value, annotates the plot with significance markers,
    and adds text to indicate which groups are being compared.
    '''
    if test == 'paired_ttest':
        # Ensure treatments and hues are in the same order as plotted
        treatments = [t for t in x_order if t in data[x].unique()]
        hues = [h for h in hue_order if h in data[group].unique()]

        # Determine behavior based on the number of groups
        if len(hues) > 2:
            comparison_pairs = [(hues[0], comp_group) for comp_group in hues[1:]]
        else:
            comparison_pairs = [(hues[0], hues[1])] if len(hues) > 1 else []

        # Perform comparisons and annotate
        for treatment in treatments:
            for base_group, comp_group in comparison_pairs:
                base_group_data = data[(data[group] == base_group) & (data[x] == treatment)][y]
                comp_data = data[(data[group] == comp_group) & (data[x] == treatment)][y]

                if len(base_group_data) > 0 and len(comp_data) > 0:
                    stat, p_value = stats.ttest_rel(base_group_data, comp_data)
                    # print (f"{treatment} {base_group} vs {comp_group} p value: {p_value}" )

                    sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < p_threshold else ''
                    if sig_marker:
                        # Find the correct x position for the treatment
                        x_pos = treatments.index(treatment)
                        y_pos = max(base_group_data.max(), comp_data.max()) + 0.1

                        ax.text(x_pos, y_pos, sig_marker, ha='center', va='bottom', fontsize=14, fontweight='bold')
                        group_annotation = f"{base_group} vs {comp_group} for {treatment}"
                        # print(f"annotating {group_annotation}")
                        ax.text(x_pos, y_pos - 0.05, group_annotation, ha='center', va='bottom', fontsize=10, color='gray')

# def add_statistical_annotation_hues(ax, data, x, y, group, test=None, p_threshold=0.05):
#     '''
#     will perform a test between hues of a graph i.e. group
#     input: axis to plot, df, x axis column name, y axis column name, group to compare column name
#     output: calculates and plots stars above x value where hue is significantly different 
    
#     '''
#     if test == 'paired_ttest':
#         treatments = data[x].unique()
#         for treatment in treatments:
#             pre_data = data[(data[group] == 'PRE') & (data[x] == treatment)][y]
#             post_data = data[(data[group] == 'POST') & (data[x] == treatment)][y]
#             if len(pre_data) > 0 and len(post_data) > 0:
#                 stat, p_value = stats.ttest_rel(pre_data, post_data)
                
#                 if p_value < 0.001:
#                     sig_marker = '***'
#                 elif p_value < 0.01:
#                     sig_marker = '**'
#                 elif p_value < p_threshold:
#                     sig_marker = '*'
#                 else:
#                     continue  # No significant difference
                
#                 x_pos = list(treatments).index(treatment)
#                 y_pos = max(pre_data.max(), post_data.max())
#                 ax.text(x_pos, y_pos, sig_marker, ha='center', va='bottom', fontsize=20)







####### IN THE PROCESS OF BEING REWRITTEN




def plotSwarmHistogram(name, data=None, order=None, color_dict=None, x='col_name_dv', y='col_name_iv', 
                       bar_hue = None, swarm_hue=None, x_label=None, y_label=None, marker='o', swarm_dodge= False): 
    fig, axs = plt.subplots(1,1, figsize = (20, 10))
    sns.barplot(data = data, x=x, y=y,  order=order, palette=color_dict, capsize=.1, 
                         alpha=0.8, errcolor=".2", edgecolor=".2" , hue = bar_hue, ci=68)
    sns.swarmplot(data = data,x=x, y=y, order=order, palette=color_dict,  hue= swarm_hue, linewidth=1, linestyle='-', dodge=swarm_dodge, marker=marker)  
    axs.set_xlabel( x_label, fontsize = 20)
    axs.set_ylabel( y_label, fontsize = 20)
    axs.set_title( name +' '+ y_label + '  (CI 95%)', fontsize = 30) 
    plt.tight_layout()
    return fig, axs





#### FOR future function

def _plot_pAD(celltype_drug_datatype, df, color_dict):
    cell_type, drug , data_type = celltype_drug_datatype
    if data_type == 'APP': 
        pAD_df = df[['cell_id','PRE_pAD_AP_locs', 'APP_pAD_AP_locs', 'WASH_pAD_AP_locs', 'PRE_Somatic_AP_locs', 'APP_Somatic_AP_locs', 'WASH_Somatic_AP_locs']]
        pAD_df = pAD_df.dropna() #removing traces with no APs at all
        if len(pAD_df['cell_id'].unique()) <= n_minimum:
            print(f'Insuficient data with APs for {cell_type} with {drug} application ')
            return df
        
        pAD_df_to_plot = pd.melt(pAD_df, id_vars=['cell_id'], var_name='col_name', value_name='AP_locs'  )
        pAD_df_to_plot[['drug', 'APP']] = pAD_df_to_plot['col_name'].str.split('_', n=1, expand=True).apply(lambda x: x.str.split('_').str[0])
        pAD_df_to_plot['count'] = pAD_df_to_plot['AP_locs'].apply(len)
        pAD_df_to_plot['drug'] = pAD_df_to_plot['drug'].str.replace('APP', drug)
        order = ['PRE', drug , 'WASH']
        
        fig, axs = plotSwarmHistogram(cell_type, data=pAD_df_to_plot, order=order, color_dict=color_dict, x='drug', y='count', 
                                        swarm_hue='APP', bar_hue='APP', x_label='', y_label='number of APs', marker = 'x',  swarm_dodge= True)
        axs.set_title( f'{cell_type}_pAD_vs_somatic_APs_{drug} (CI 95%)', fontsize = 30) 
        saveFP_HistogramFig(fig, f'{cell_type}_pAD_vs_somatic_APs_{drug}')
        plt.close('all')
        
    elif data_type == 'FP_APP': 
        return df
    elif data_type == 'pAD': 
        return df
    elif data_type == 'FP': 
        return df
    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP, pAD or FP
    return df 