# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:43:53 2023

@author: Debapratim Jana, Jasmine Butler
"""


import igor.packed
import igor.binarywave
import pandas as pd
import numpy as np
from ephys.ap_functions import calculate_max_firing, extract_FI_x_y, extract_FI_slope_and_rheobased_threshold
from ephys.ap_functions import ap_characteristics_extractor_main, tau_analyser, sag_current_analyser , pAD_detection
import matplotlib.pyplot as plt 
import seaborn as sns

#%% BASIC FUNCTIONS:  igor data extractor, path manager 

color_dict = {"PRE":"black", "CONTROL": 'grey', "TCB2":'green', "DMT":"teal", "PSIL":"orange", "LSD":"purple", "MDL":'blue'}
 
def make_path(base_path, folder_file):
    '''
    Parameters       
    ----------
    folder_file : TYPE
        DESCRIPTION.

    Returns
    -------
    path_V : string - path for V data 
    path_I : string - path for I data 
    '''
    extension_V = "Soma.ibw"
    extension_I = "Soma_outwave.ibw" 
    path_V = base_path + folder_file + extension_V
    path_I = base_path + folder_file + extension_I
    return path_V, path_I

def igor_exporter(path):
    
    ''' 
     Parameters
    ----------
    path: path of data, .ibw files
    
    Returns
    -------
    'point_list' : a continious points  (combining sweeps into continious wave)  
    'igor_df' : a df with each column corisponding to one sweep  

     '''
    igor_file = igor.binarywave.load(path)
    wave = igor_file["wave"]["wData"]
    igor_df = pd.DataFrame(wave)
    
    point_list = list()
    counter = len(igor_df.columns)
    
    for i in range(len(igor_df.columns)):
        temp_list = igor_df.iloc[:,i].tolist()
        point_list.extend(temp_list)
        counter = counter - 1
    
    return (point_list, igor_df)




#%% RUN PLOTS

#drug_aplication_visualisation(feature_df, color_dict) # generates PDF of drug aplications

#plot_all_FI_curves(feature_df,  color_dict)  # generates PDF with all FI curves for single cell labed with drug and aplication order #### MAKE HZ NOT APs per sweep also isnt it in pA not nA??

#plot_FI_AP_curves(feature_df) #generated PDF with FI-AP for each cell


#%%   EXPAND FEATURE_DF

def _handleFile(row, base_path = "/Users/debap/Desktop/PatchData/"): #takes a row of the df (a single file) and extractes values based on the data type  FP or AP then appends values to df
    # print(row.index)
    # print(row.folder_file)# "- num entries:", row)
    row = row.copy()
    # Dataframe extraction 
    path_V, path_I = make_path(base_path, row.folder_file)
    V_list, V_df = igor_exporter(path_V)
    I_list, I_df = igor_exporter(path_I)
    V_array , I_array = np.array(V_df) , np.array(I_df)
    

    if row.data_type in ["FP", "FP_AP"]:
        pass
        
    
        # print("FP type file")
        # print(row.folder_file)
        # # row["new_data_type"] = ["Boo", 0, 1 ,2] #creating a new column or filling a pre existing column with values or list of values
        
        # #data files to be fed into extraction funcs
                
        
        # row["max_firing"] = calculate_max_firing(V_array)
        
        # x, y, v_rest = extract_FI_x_y (path_I, path_V) # x = current injected, y = number of APs, v_rest = the steady state voltage
        # FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True) ## FIX ME some values are negative! this should not be so better to use slope or I step? Needs a pAD detector 
        # row["rheobased_threshold"] = rheobase_threshold
        # row["FI_slope"] = FI_slope
        
        

        # peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(V_df)
        # row["voltage_threshold"] = v_thresholds_all 
        # row["AP_height"] = peak_heights_all
        # row["AP_width"] = peak_fw_all
        # row["AP_slope"] = peak_slope_all 

        # #not yet working    #UnboundLocalError: local variable 'last_current_point' referenced before assignment
        # #extract_FI_x_y has been used differently by DJ check this is correct  # step_current_values  == x
        # tau_all         =  tau_analyser(V_array, I_array, x, plotting_viz= False, analysis_mode = 'max')
        # sag_current_all =  sag_current_analyser(V_array, I_array, x)

        # row["tau_rc"] = tau_all
        # row["sag"]    = sag_current_all
    
     
        
    elif row.data_type == "AP":
        print("AP type file")
        print(row.folder_file)        
        # row["new_data_type"] = ["Boo", 0, 1 ,2]
        # pAD classification
        peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df, X_pca   =   pAD_detection(V_array)
        if len(pAD_df["pAD_count"]) == 0: 
            row["pAD_count"] =  np.nan
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()   # get AP_lcos 
        else:
            row["pAD_count"] =  pAD_df["pAD_count"][0]     #  need to make count not series   
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()   # get AP_lcos 
        pass
    else:
        raise NotADirectoryError(f"Didn't handle: {row.data_type}")
    return row

def _handleCellID(cell_id_drug, cell_df):
    cell_id, drug = cell_id_drug
    # fp_idx = cell_df.data_type == "FP" #indexes of FP data in cell_df
    # cell_df.loc[fp_idx, "new_col"] = calcValue()

    return cell_df


def loopCombinations(df):
    og_columns = df.columns.copy() #origional columns #for ordering columns
    df['mouseline'] = df.cell_ID.str[:3]
    # print("Columns: ", df.columns[-40:])
    # df = pd.read_excel(r'E:\OneDrive - Floating Reality\analysis\feature_df_py.xlsx')
    df = df.apply(_handleFile, axis=1) #Apply a function along an axis (rows = 1) of the DataFrame
    # display(df)
    combinations = [
                    (["cell_ID", "drug"], _handleCellID), #finding all combination in df and applying a function to them #here could average and plot or add to new df for stats
                    #(["cell_type", "drug"], _poltbygroup) #same as df.apply as folder_file is unique for each row
    ]
    # for col_names, handlingFn in combinations:
    #     df = apply_group_by_funcs(df, col_names, handlingFn) #handel function is the generic function to be applied to different column group by
    
    #ORDERING DF internal: like this new columns added will appear at the end of the df in the order they were created in _handelfile()
    all_cur_columns = df.columns.copy()
    new_colums_set = set(all_cur_columns ) - set(og_columns) # Subtract mathematical sets of all cols - old columns to get a set of the new columns
    new_colums_li = list(new_colums_set)
    reorder_cols_li =  list(og_columns) + new_colums_li # ['mouseline'] + ordering
    df_reordered = df[reorder_cols_li]    
    
    return df_reordered


#%% stats/plotting 
def apply_group_by_funcs(df, groupby_cols, handleFn): #creating a list of new values and adding them to the existign df
    res_dfs_li = [] #list of dfs
    for group_info, group_df in df.groupby(groupby_cols):
        
        res_df = handleFn(group_info, group_df, color_dict)
        res_dfs_li.append(res_df)
        
    new_df = pd.concat(res_dfs_li)
    return new_df



def _getstats_FP(mouseline_drug_datatype, df, color_dict):
    
    mouse_line, drug, data_type = mouseline_drug_datatype
    
    if data_type == 'AP': #if data type is not firing properties (FP then return df)
            return df
        
    df = df.copy()
    #for entire mouseline_drug combination
    df['mean_max_firing'] = df['max_firing'].mean() #dealing with pd.series a single df column
    df['SD_max_firing'] = df['max_firing'].std()
    
    df['mean_FI_slope'] = df['FI_slope'].mean() #dealing with pd.series a single df column
    df['SD_FI_slope'] = df['FI_slope'].std()
    
    df['mean_rheobased_threshold'] =df['rheobased_threshold'].mean()
    
    #AP Charecteristics i.e. mean/SD e.c.t. for a single file (replications not combined consider STATISTICAL IMPLICATIONS FORPLOTTING)
    df['mean_voltage_threshold_file'] = df.voltage_threshold.apply(np.mean) #apply 'loops' through the slied df and applies the function np.mean
    df['SD_voltage_threshold_file'] = df.voltage_threshold.apply(np.std)
    
    df['mean_AP_height_file'] = df.AP_height.apply(np.mean)
    
    df['mean_AP_slope_file'] = df.AP_slope.apply(np.mean)
    
    df['mean_AP_width'] = df.AP_width.apply(np.mean)
    
    return df

def _plotwithstats_FP(mouseline_datatype, df, color_dict):
    
    
    mouse_line, data_type = mouseline_datatype
    order = list(color_dict.keys()) #ploting in order of dict keys
    
    if data_type == 'AP': #if data type is not firing properties (FP then return df)
        return df
    if data_type == 'FP_AP': #if data type is not firing properties (FP then return df)
        return df
    
    # generating plot by single file values e.g. max_firing or mean_voltage_threshold_file
    
    factors_to_plot = ['max_firing', 'rheobased_threshold', 'FI_slope', 'mean_voltage_threshold_file', 'mean_AP_height_file', 'mean_AP_slope_file', 'mean_AP_width']
    names_to_plot = ['_max_firing_Hz', 'rheobased_threshold_pA' , '_FI_slope_linear', '_voltage_threshold_mV', '_AP_height_mV', '_AP_slope', '_AP_width_ms']
    
    for _y, name in zip(factors_to_plot, names_to_plot):
    
        sns_barplot_swarmplot (df, order, mouse_line, _x='drug', _y=_y, name = name)
    

    plt.close("all") #close open figures
    
    return df 

def sns_barplot_swarmplot (df, order, mouse_line, _x='drug', _y='max_firing', name = '_max_firing_Hz'):
    '''
    Generates figure and plot and saves to patch_daddy_output/
    '''
    fig, axs = plt.subplots(1,1, figsize = (20, 10))
    sns.barplot(data = df, x=_x, y=_y,  order=order, palette=color_dict, capsize=.1, 
                             alpha=0.8, errcolor=".2", edgecolor=".2" )
    sns.swarmplot(data = df,x=_x, y=_y, order=order, palette=color_dict, linewidth=1, linestyle='-')
    axs.set_xlabel( "Drug applied", fontsize = 20)
    axs.set_ylabel( name, fontsize = 20)
    axs.set_title( mouse_line + name + '  (CI 95%)', fontsize = 30)
    
    fig.savefig('patch_daddy_output/' + mouse_line + name + '.pdf')
    
    return

def loopCombinations_stats(df):
    #create a copy of file_folder column to use at end of looping to restore  origional row order !!! NEEDS TO BE DONE
    
    #keepingin mind that the order is vital  as the df is passed through againeach one
    combinations = [
                    (["mouseline", "drug", "data_type"], _getstats_FP), #stats_df to be fed to next function mouseline 
                    (["mouseline",  "data_type"], _plotwithstats_FP) #finding all combination in df and applying a function to them #here could average and plot or add to new df for stats
                    # (["cell_ID", "drug", "data_type"], _plotstats_FP)
    ]
    
    # pdf = PdfPages('patch_daddy_output.pdf') # open pdf for plotsto be saved in
    # pdf.savefig(fig)
    # pdf.close()

    for col_names, handlingFn in combinations:

    
        df = apply_group_by_funcs(df, col_names, handlingFn) #note that as each function is run the updated df is fed to the next function
    
    
    
    return df