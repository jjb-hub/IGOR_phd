# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:43:53 2023

@author: Debapratim Jana, Jasmine Butler
"""

#my modular imports
from module.base_utils import *
from module.action_potential_functions import calculate_max_firing, extract_FI_x_y, extract_FI_slope_and_rheobased_threshold
from module.action_potential_functions import ap_characteristics_extractor_main, tau_analyser, sag_current_analyser , pAD_detection

#generic imports
import os, shutil, itertools, json, time, functools, pickle
import pandas as pd
import glob #new
import igor.packed
import igor.binarywave
import numpy as np
import pandas as pd #new
from pathlib import Path #new

from matplotlib.backends.backend_pdf import PdfPages #new
import matplotlib.pyplot as plt 
import scipy
import scipy.signal as sg #new
import seaborn as sns
from statannotations.Annotator import Annotator


#CONSTANTS   
ROOT = os.getcwd() #This gives terminal location (terminal working dir)
INPUT_DIR = f'{ROOT}/input'
OUTPUT_DIR = f'{ROOT}/output'
CACHE_DIR = f'{INPUT_DIR}/cache'

n_minimum = 3 
p_value_threshold=0.05 #FIX ME build these in with simple get or guild functions outside the meta loop also interrate with meta loop



#EXPAND FEATURE_DF
#takes a row of the df (a single file) and extractes values based on the data type  FP or AP then appends values to df

## New function DJ : 
    
def generate_V_pAD_df(folder_file): 
    '''
    Generates pAD_df, V_array  
    
    Input : 
           folder_file : str 
    
    Ouput : 
           pAD_df  : pAD dataframe built from pAD_detection
           V_array : v array 
           
           
    '''
    path_V, path_I = make_path(folder_file)
    V_list, V_df = igor_exporter(path_V)
    V_array      = np.array(V_df) 
    
    peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  =   pAD_detection(V_array)
    
    return pAD_df , V_array

def _handleFile(row): 
    row = row.copy()
    # Dataframe extraction 
    path_V, path_I = make_path(row.folder_file)
    V_list, V_df = igor_exporter(path_V)
    V_array      = np.array(V_df) 
    
    # handel missing I traces (Soma_outwave)
    try:
        I_list, I_df = igor_exporter(path_I)
        I_array      = np.array(I_df)
    except FileNotFoundError:
        print('I file not found, path:', path_I)


    if row.data_type in ["FP", "FP_AP"]:
        # pass 
        # print("FP type file")
        # print(row.folder_file)

        #data files to be fed into extraction funcs feature_df_extended
        row["max_firing"] = calculate_max_firing(V_array)
        
        x, y, v_rest = extract_FI_x_y (path_I, path_V) # x = current injected, y = number of APs, v_rest = the steady state voltage (no I injected)
        FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True) #FIX ME: handel pAD APs better slope fit
        row["rheobased_threshold"] = rheobase_threshold
        row["FI_slope"] = FI_slope

        peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(V_df)
        row["voltage_threshold"] = v_thresholds_all 
        row["AP_height"] = peak_heights_all
        row["AP_width"] = peak_fw_all
        row["AP_slope"] = peak_slope_all 
        row["AP_latency"] = peak_latencies_all

        #to DJ: to remove the named tuples I just mad them lists of lists i.e. [[value, steady_state, I_injected], [value, steady_state, I_injected] ]
        row["tau_rc"] = tau_analyser(V_array, I_array, x, plotting_viz= False, analysis_mode = 'max')
        row["sag"]    = sag_current_analyser(V_array, I_array, x)

        
    elif row.data_type == "AP":

        print("AP type file")
        print(row.folder_file)        
        
        # pAD classification
        peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  =   pAD_detection(V_array)

        if len(pAD_df["pAD_count"]) == 0:  #if there are no APs of any kind detected
            row["pAD_count"] =  np.nan
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()   # getlocations of ALL? APs : AP_lcos 
        else:
            row["pAD_count"] =  pAD_df["pAD_count"][0]     #  need to make count not series / ERROR HERE  
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()    

            #unsure weather to just do count but can use len()
            row['PRE_Somatic_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
            row['APP_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
            row['WASH_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()
            row['PRE_pAD_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
            row['APP_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
            row['WASH_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()

        #make points list for PRE , APP and WASH voltages #THIS IS TOO BIG EXTRACT RELEVANT DATA i.e. SD and add 
        #saving full points list is too long! #to float 16 instead of 32 
        #FIX ME 
        # varray[ rows to take   : cols to take ]
        # varray[:, :row.drug_in] #take all rows up to drug_in

        # row['PRE_V'] = V_array[:, :row.drug_in] 
        # row['APP_V'] = V_array[:, row.drug_in:row.drug_out]
        # row['WASH_V'] = V_array[:, row.drug_out:]
        pass
    
    elif row.data_type == "pAD":
        print('data_type pAD')

    else:
        raise NotADirectoryError(f"Didn't handle: {row.data_type}") # data_type can be AP, FP_AP or FP
    return row


def expandFeatureDF(filename):
    df = pd.read_excel (f'{INPUT_DIR}/{filename}', converters={'drug_in':int, 'drug_out':int})
    og_columns = df.columns.copy() #origional columns #for ordering columns
    df['mouseline'] = df.cell_ID.str[:3]
    df = df.apply(_handleFile, axis=1) #Apply a function along an axis (rows = 1) of the DataFrame

    #ORDERING DF internal: like this new columns added will appear at the end of the df in the order they were created in _handelfile()
    all_cur_columns = df.columns.copy()
    new_colums_set = set(all_cur_columns ) - set(og_columns) # Subtract mathematical sets of all cols - old columns to get a set of the new columns
    new_colums_li = list(new_colums_set)
    reorder_cols_li =  list(og_columns) + new_colums_li # ['mouseline'] + ordering
    df_reordered = df[reorder_cols_li]    
    
    return df_reordered


# GENERATE STATS AND PLOT EXPANDED_DF  

def apply_group_by_funcs(df, groupby_cols, handleFn, color_dict): #creating a list of new values and adding them to the existign df
    res_dfs_li = [] #list of dfs
    for group_info, group_df in df.groupby(groupby_cols):
        
        res_df = handleFn(group_info, group_df, color_dict)
        res_dfs_li.append(res_df)
        
    new_df = pd.concat(res_dfs_li)
    return new_df


def _colapse_to_file_value_FP(celltype_drug_datatype, df, color_dict):
    #colaps  metrics to a single value for each file/row i.e. lists become a mean ect.
    cell_type, drug, data_type = celltype_drug_datatype
    
    if data_type == 'AP': 
        return df
    elif data_type == "pAD":
        return df
    elif data_type in ['FP', 'FP_AP']:
        df = df.copy()
        #AP Charecteristics i.e. mean/SD e.c.t. for a single file 
        df['mean_voltage_threshold_file'] = df.voltage_threshold.apply(np.mean)  #FIX ME: RuntimeWarning: Mean of empty slice.
        df['mean_AP_height_file'] = df.AP_height.apply(np.mean)
        df['mean_AP_slope_file'] = df.AP_slope.apply(np.mean)
        df['mean_AP_width_file'] = df.AP_width.apply(np.mean)
        df['mean_AP_latency_file'] = df.AP_latency.apply(np.mean)
        
        
        #Tau and Sag colapse to lowest value #NO LOBGER TUPLES
        # extract_truple_data('sag', df) #creating columns 'tau_file' and 'sag_file'
        # extract_truple_data('tau_rc', df)
       
    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP or FP
    
    return df
    


def _colapse_to_cell_pre_post_FP(cellid_drug_datatype, df, color_dict):
    #for each cellid_drug_datatype colapses to a  single value 
    cell_id, drug , data_type = cellid_drug_datatype
    
    if data_type == 'AP': 
        return df
    elif data_type == 'FP_AP': 
        return df
    elif data_type == 'pAD': 
        return df
    
    elif data_type == 'FP':
        df = df.copy()
        df['max_firing_cell_drug'] = df['max_firing'].mean() #RuntimeWarning: Mean of empty slice.
        #AP Charecteristics 
        df['voltage_threshold_cell_drug'] = df['mean_voltage_threshold_file'].mean()
        df['AP_height_cell_drug'] = df['mean_AP_height_file'].mean()
        df['AP_slope_cell_drug'] = df['mean_AP_slope_file'].mean()
        df['AP_width_cell_drug'] = df['mean_AP_width_file'].mean()
        df['AP_latency_cell_drug'] = df['mean_AP_latency_file'].mean()
        
        #Tau and Sag #FIX ME tau and sag are in list now #FIX ME 
        #chose optimal paring for comparison
        # df['tau_cell_drug'] = df['tau_rc_file'].mean()
        # df['sag_cell_drug'] = df['sag_file'].mean()

    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP, pAD or FP
    return df 

def _colapse_to_cell_pre_post_tau_sag_FP(celltype_datatype, df, color_dict):
    cell_type, data_type = celltype_datatype
    if data_type in ['AP', 'pAD']: 
        return df
    elif data_type == 'FP_AP': 
        return df
    elif data_type == 'FP': 
        return df
         ### FIX ME 
        #List of Lists    len(list)=1/2     each item = [value, steady_state, I_injected]    if no data 'NaN'
        # for drug, df_drug in df.groupby('drug'):  #normalised for I injection
        #     df_drug['sag']
        # df['tau_cell_drug'] =  /
        # df['sag_file'] = 
    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP, pAD or FP
    return df 

def _plotwithstats_FP(celltype_datatype, df, color_dict):
    cell_type, data_type = celltype_datatype

    if data_type == 'AP': 
        return df
    elif data_type == 'FP_AP': 
        return df
    elif data_type == 'pAD': 
        return df
    
    elif data_type == 'FP':
        cell_id_list = list(df['cell_ID'].unique())
        build_first_drug_ap_column(df, cell_id_list) #builds df['first_drug_AP']
        
        df_only_first_app = df.loc[df['application_order'] <= 1] #only include first drug application data 
            
        plot_list = [['max_firing_cell_drug', 'Firing_(Hz)'], 
                    ['voltage_threshold_cell_drug','Voltage_Threshold_(mV)'], 
                    ['AP_height_cell_drug', ' AP_Height_(mV)'], 
                    ['AP_slope_cell_drug', 'AP_slope_(V_s^-1)'],
                    ['AP_width_cell_drug', 'AP_width_(s) '],
                    ['AP_latency_cell_drug', 'AP_latency_(ms)']
                    ] 
        
                    #  ['tau_cell_drug', 'Tau_RC_(ms)'],
                    #  ['sag_cell_drug', 'Percentage_sag_(%)']
        for col_name, name in plot_list:
            #FIX ME: only plot and do stats when n>=3
            statistical_df = build_FP_statistical_df(df_only_first_app, cell_id_list, col_name)
            drug_count_n = statistical_df['drug'].value_counts()
            drugs_to_remove = drug_count_n[drug_count_n < n_minimum].index.tolist()
            statistical_df_filtered = statistical_df[~statistical_df['cell_ID'].isin(statistical_df[statistical_df['drug'].isin(drugs_to_remove)]['cell_ID'])]
   
            #create df_to_plot with only one value per cell (not per file)
            df_to_plot = df_only_first_app[['cell_ID','drug',col_name, 'first_drug_AP']]
            df_to_plot = df_to_plot.drop_duplicates()
            #remove cells where the 'drug' occures <  n_minimum
            df_to_plot_filtered = df_to_plot[~df_to_plot['cell_ID'].isin(df_to_plot[df_to_plot['drug'].isin(drugs_to_remove)]['cell_ID'])]

            if len(df_to_plot_filtered) == 0:
                print(f'Insufficient data for {cell_type}for{name}')
                continue 
            
            student_t_df = build_student_t_df(statistical_df_filtered, cell_type) #calculate and create student t test df to be used to plot stars

            order_dict = list(color_dict.keys()) #ploting in order == dict keys
            order_me = list(df_to_plot_filtered['drug'].unique())
            order = [x for x in order_dict if x in order_me]

            fig, axs = plotSwarmHistogram(cell_type, data=df_to_plot_filtered, order=order, color_dict=color_dict, x='drug', y=col_name, 
                                        swarm_hue='first_drug_AP', x_label='Drug applied', y_label=name)
            put_significnce_stars(axs, student_t_df, data=df_to_plot, x='drug', y=col_name, order = order) #add stats from student t_test above
            saveFP_HistogramFig(fig, f'{cell_type}_{name}')
            plt.close('all')
    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP, pAD or FP
    return df 

def _plot_pAD(celltype_drug_datatype, df, color_dict):
    cell_type, drug , data_type = celltype_drug_datatype
    if data_type == 'AP': 
        pAD_df = df[['cell_ID','PRE_pAD_AP_locs', 'APP_pAD_AP_locs', 'WASH_pAD_AP_locs', 'PRE_Somatic_AP_locs', 'APP_Somatic_AP_locs', 'WASH_Somatic_AP_locs']]
        pAD_df = pAD_df.dropna() #removing traces with no APs at all
        if len(pAD_df['cell_ID'].unique()) <= n_minimum:
            print(f'Insuficient data with APs for {cell_type} with {drug} application ')
            return df
        
        pAD_df_to_plot = pd.melt(pAD_df, id_vars=['cell_ID'], var_name='col_name', value_name='AP_locs'  )
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


def loopCombinations_stats(filename):
    df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False) #load feature df
    color_dict = getColors(filename)

    #create a copy of file_folder column to use at end of looping to restore  origional row order !!! #FIX ME
    # df_row_order = df['folder_file'] / df_raw_col_order[]
    
    combinations = [
                    (["cell_type", "drug", "data_type"], _colapse_to_file_value_FP),
                    (["cell_ID", "drug",  "data_type"], _colapse_to_cell_pre_post_FP),
                    (["cell_type",  "data_type"], _colapse_to_cell_pre_post_tau_sag_FP), 
                    (["cell_type",  "data_type"], _plotwithstats_FP), 
                    (["cell_type", "drug", "data_type"], _plot_pAD)
           
    ]

    for col_names, handlingFn in combinations:
        df = apply_group_by_funcs(df, col_names, handlingFn, color_dict) #note that as each function is run the updated df is fed to the next function

    # df[ order(match(df['folder_file'], df_row_order)) ]  #FIX ME
    return df


## dependants of above #old

def build_first_drug_ap_column(df, cell_id_list):
    df['first_drug_AP'] = ''  #create a column for  specifying the first drug applied for each cell
    for cell_id in cell_id_list:
        cell_df = df.loc[df['cell_ID'] == cell_id] #slice df to cell only
        first_drug_series = cell_df.loc[df['application_order'] == 1, 'drug'] 
        if len(first_drug_series.unique()) > 1: 
            print ('ERROR IN DATA ENTERY: multiple drugs for first aplication on cell ', cell_id)
        first_drug_string = first_drug_series.unique()[0] 
        df.loc[df['cell_ID'] == cell_id, 'first_drug_AP'] = first_drug_string
    return

def fill_statistical_df_lists(cell_df, col_name, first_drug_string, lists_to_fill):
    '''
    Parameters
    ----------
    cell_df : df for a single cell
    col_name : string of column name for cell and drug  e.g. 'max_firing_cell_drug'
    lists_to_fill: list of THREE  lists to be filled e.g. [PRE_, POST_, first_drug_]
    '''   
    PRE = cell_df.loc[cell_df['drug'] == 'PRE', col_name] #HARD CODE
    POST = cell_df.loc[cell_df['application_order'] == 1, col_name ]
    PRE = PRE.unique()
    POST = POST.unique()
    
    if len(PRE) ==1 & len(POST) ==1 :
        lists_to_fill[0].append(float(PRE))
        lists_to_fill[1].append(float(POST))
        lists_to_fill[2].append(first_drug_string)
    else:
        print('ERROR IN DATA : values for single cell_drug not congruent' )
        print('PRE = ', PRE)
        print('POST = ', POST)
        lists_to_fill[0].append('NaN')   
        lists_to_fill[1].append('NaN')
        lists_to_fill[2].append(first_drug_string)
    return

def build_student_t_df(statistical_df, cell_type):
    '''
    Parameters
    ----------
    statistical_df : df with single cell value PRE POST
    cell_type : string indicating the cell type  e.g. L6b_DRD
    Returns
    -------
    student_t_df : df with columns ['cell_type', 'drug', 't_stat', 'p_val' ]
    '''
    student_t_df_list = []
    for drug, drug_df in statistical_df.groupby('drug'):
        # print(drug_df)
        T_stat, p_val = scipy.stats.ttest_rel(drug_df['PRE'], drug_df['POST']) #H0: two related or repeated samples have identical average (expected) values
        student_t_df_row = [cell_type, drug, T_stat, p_val]
        student_t_df_list.append(student_t_df_row)
        
    student_t_df = pd.DataFrame(student_t_df_list, columns = ['cell_type', 'drug', 't_stat', 'p_val' ])     
    # print (student_t_df)
    return student_t_df


def put_significnce_stars(axs, df, data=None,
                          x=None, y=None, order=None):  # , p_values):
    '''
     Parameters
    ----------
    axs : axis of figure to plot significance on
    df: data frame with columns ['cell_type', 'drug', 't_stat', 'p_val' ]
    data: data frame plotted (bar/hist or scatter)
    x: x plotted
    y: y plotted
    order : order of x_tics
    '''
    significant_drugs = df.loc[df.p_val <= 0.05, 'drug']
    if significant_drugs.empty:
        return
        
    p_values = df.loc[df.p_val <= 0.05, 'p_val']
    pairs = list(zip(['PRE'] * len(significant_drugs), significant_drugs,))
    annotator = Annotator(axs, pairs, data=data,
                          x=x, y=y, order=order)
    annotator.configure(text_format="star",
                        loc="inside", fontsize='xx-large')
    annotator.set_pvalues_and_annotate(p_values.values)
    return 

def build_FP_statistical_df(df_only_first_app, cell_id_list, col_name):  #col_name = 'max_firing_cell_drug'
    PRE_ = []
    POST_ = []
    first_drug_ = []
    lists_to_fill = [PRE_, POST_, first_drug_]
    
    for cell_id in cell_id_list:
        
        cell_df = df_only_first_app.loc[df_only_first_app['cell_ID'] == cell_id] #slice df to cell only 
        first_drug_string = cell_df['first_drug_AP'].unique()[0]

        fill_statistical_df_lists(cell_df, col_name, first_drug_string, lists_to_fill) #e.g col_name = 'max_firing_cell_drug'
    #create statistical df for celltype with all raw data for single value type e.g. max firing
    statistical_df = pd.DataFrame({'cell_ID': cell_id_list,
                                  'drug': first_drug_,
                                  'PRE': PRE_,
                                  'POST': POST_
                                  })
    return statistical_df


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


