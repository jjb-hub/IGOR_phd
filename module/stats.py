
#my modular imports
from module.utils import *

from module.action_potential_functions import  pAD_detection
from module.getters import  getExpandedDfIfFilename, updateFPStats

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


#CONSTANTS   
from module.constants import CACHE_DIR, INPUT_DIR, OUTPUT_DIR, color_dict, n_minimum,  p_value_threshold



########## WORKING WITH EXPANDED DF ###########


def loopFP_stats(filename_or_df):

    df, filename = getExpandedDfIfFilename(filename_or_df)
    df_row_order = df['folder_file'].tolist()  # save origional row order

    combinations = [(["cell_type", "cell_id", "data_type"], _update_FP_stats),
                    # (["cell_type", "drug", "data_type"], _colapse_to_file_value_FP), #AP charecteristics from first 2 sweeps with APs
                    # (["cell_id", "drug",  "data_type"], _colapse_to_cell_pre_post_FP),
                    # (["cell_type",  "data_type"], _colapse_to_cell_pre_post_tau_sag_FP), 
                    # (["cell_type",  "data_type"], _plotwithstats_FP), 
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

def _update_FP_stats(filename, celltype_cellid_datatype, df):
    '''
    input: single cell FP expanded df
    output: updates FP_stats
    '''
    cell_type, cell_id, data_type = celltype_cellid_datatype
    df=df[df['application_order'] <= 1] #remove second aplication data 
    update_rows = []
    if data_type == 'FP':
        # normalised tau and sag by v difference / I injected (pA) #TODO
        # df['tau_rc_v_diff'] = df['tau_rc'].apply(lambda x: [abs(item[1] - item[3]) for item in x])
        # df['sag_v_diff'] = df['sag'].apply(lambda x: [abs(item[1] - item[3]) for item in x])

        treatment = ', '.join(df[df['drug'] != 'PRE']['drug'].unique())
        for drug, pre_post_df in df.groupby('drug'):
            pre_post = 'PRE' if 'PRE' in pre_post_df['drug'].values else 'POST'

            for measure in ['max_firing', 
                            'voltage_threshold',
                            'AP_height',
                            'AP_slope',
                            'AP_width',
                            'AP_latency']:

                mean_value, file_values = fetchMeans(pre_post_df, measure)
                update_row = {
                    "cell_type": cell_type,
                    "cell_id": cell_id,
                    "measure": measure,
                    "treatment": treatment,
                    "pre_post": pre_post,
                    "mean_value": mean_value,
                    "file_values": file_values
                }
                # Append the dictionary to the list
                update_rows.append(update_row)
                print (update_row)
        updateFPStats(filename, update_rows)
    else:
        pass
    return df

def fetchMeans(pre_post_df, measure):
    column=pre_post_df[measure]
    if all(isinstance(value, list) for value in column): #list comprehansion 
        file_values = [np.mean(value) for value in column]
        mean_value = np.mean(file_values)
    else: #single value comprehension
        file_values = column.tolist()  
        mean_value = np.mean(file_values)
    return mean_value, file_values



######## CALCULATE / PLOT STATS
def add_statistical_annotation(ax, data, x, y, group, test=None, p_threshold=0.05):
    '''
    input: axis to plot, df, x axis column name, y axis column name, group to compare column name, '''
    if test == 'paired_ttest':
        treatments = data[x].unique()
        for treatment in treatments:
            pre_data = data[(data[group] == 'PRE') & (data[x] == treatment)][y]
            post_data = data[(data[group] == 'POST') & (data[x] == treatment)][y]
            if len(pre_data) > 0 and len(post_data) > 0:
                stat, p_value = stats.ttest_rel(pre_data, post_data)
                
                if p_value < 0.001:
                    sig_marker = '***'
                elif p_value < 0.01:
                    sig_marker = '**'
                elif p_value < p_threshold:
                    sig_marker = '*'
                else:
                    continue  # No significant difference
                
                x_pos = list(treatments).index(treatment)
                y_pos = max(pre_data.max(), post_data.max())
                ax.text(x_pos, y_pos, sig_marker, ha='center', va='bottom', fontsize=20)



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
    if data_type == 'AP': 
        pAD_df = df[['cell_id','PRE_pAD_AP_locs', 'APP_pAD_AP_locs', 'WASH_pAD_AP_locs', 'PRE_Somatic_AP_locs', 'APP_Somatic_AP_locs', 'WASH_Somatic_AP_locs']]
        pAD_df = pAD_df.dropna() #removing traces with no APs at all
        if len(pAD_df['cell_id'].unique()) <= n_minimum:
            print(f'Insuficient data with APs for {cell_type} with {drug} application ')
            return df
        
        pAD_df_to_plot = pd.melt(pAD_df, id_vars=['cell_id'], var_name='col_name', value_name='AP_locs'  )
        pAD_df_to_plot[['drug', 'AP']] = pAD_df_to_plot['col_name'].str.split('_', n=1, expand=True).apply(lambda x: x.str.split('_').str[0])
        pAD_df_to_plot['count'] = pAD_df_to_plot['AP_locs'].apply(len)
        pAD_df_to_plot['drug'] = pAD_df_to_plot['drug'].str.replace('APP', drug)
        order = ['PRE', drug , 'WASH']
        
        fig, axs = plotSwarmHistogram(cell_type, data=pAD_df_to_plot, order=order, color_dict=color_dict, x='drug', y='count', 
                                        swarm_hue='AP', bar_hue='AP', x_label='', y_label='number of APs', marker = 'x',  swarm_dodge= True)
        axs.set_title( f'{cell_type}_pAD_vs_somatic_APs_{drug} (CI 95%)', fontsize = 30) 
        saveFP_HistogramFig(fig, f'{cell_type}_pAD_vs_somatic_APs_{drug}')
        plt.close('all')
        
    elif data_type == 'FP_AP': 
        return df
    elif data_type == 'pAD': 
        return df
    elif data_type == 'FP': 
        return df
    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP, pAD or FP
    return df 