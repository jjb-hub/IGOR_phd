# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:43:53 2023

@author: Debapratim Jana, Jasmine Butler
"""

#my modular imports
from utils.igor_utils import igor_exporter, make_path
from ephys.ap_functions import calculate_max_firing, extract_FI_x_y, extract_FI_slope_and_rheobased_threshold
from ephys.ap_functions import ap_characteristics_extractor_main, tau_analyser, sag_current_analyser , pAD_detection

#generic imports
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



color_dict = {"PRE":"black", "CONTROL": 'grey', "TCB2":'green', "DMT":"teal", "PSIL":"orange", "LSD":"purple", "MDL":'blue'}
 
##nee to remove hard coded color_dict

#%%   EXPAND FEATURE_DF
#takes a row of the df (a single file) and extractes values based on the data type  FP or AP then appends values to df

def _handleFile(row): # , base_path = "/Users/debap/Desktop/PatchData/"  # DJ WHY WAS THIS HERE?
    # print(row.index)
    # print(row.folder_file)# "- num entries:", row)
    row = row.copy()
    # Dataframe extraction 
    path_V, path_I = make_path(row.folder_file)
    
    skip_I = False
    
    V_list, V_df = igor_exporter(path_V)
    V_array      = np.array(V_df) 
    
    # missing I traces (Soma_outwave)
    try:
        I_list, I_df = igor_exporter(path_I)
        I_array      = np.array(I_df)
    except FileNotFoundError:
        print('I file not found, path:', path_I)


    if row.data_type in ["FP", "FP_AP"]:
        # pass
        
    
        print("FP type file")
        print(row.folder_file)
        # row["new_data_type"] = ["Boo", 0, 1 ,2] #creating a new column or filling a pre existing column with values or list of values
        
        #data files to be fed into extraction funcs
                
        
        row["max_firing"] = calculate_max_firing(V_array)
        
        x, y, v_rest = extract_FI_x_y (path_I, path_V) # x = current injected, y = number of APs, v_rest = the steady state voltage
        FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True) ## FIX ME some values are negative! this should not be so better to use slope or I step? Needs a pAD detector 
        row["rheobased_threshold"] = rheobase_threshold
        row["FI_slope"] = FI_slope
        
        

        peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(V_df)
        row["voltage_threshold"] = v_thresholds_all 
        row["AP_height"] = peak_heights_all
        row["AP_width"] = peak_fw_all
        row["AP_slope"] = peak_slope_all 
        row["AP_latency"] = peak_latencies_all

        #not yet working    #UnboundLocalError: local variable 'last_current_point' referenced before assignment
        #extract_FI_x_y has been used differently by DJ check this is correct  # step_current_values  == x
        tau_all         =  tau_analyser(V_array, I_array, x, plotting_viz= False, analysis_mode = 'max')
        sag_current_all =  sag_current_analyser(V_array, I_array, x)

        row["tau_rc"] = tau_all
        row["sag"]    = sag_current_all
    
     
        
    elif row.data_type == "AP":
        
        
        print("AP type file")
        print(row.folder_file)        
        
        # pAD classification
        peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df, X_pca   =   pAD_detection(V_array)
        
        
        
     

        if len(pAD_df["pAD_count"]) == 0:  #if there are no APs of any kind detected
            row["pAD_count"] =  np.nan
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()   # getlocations of ALL? APs : AP_lcos 
        else:
            row["pAD_count"] =  pAD_df["pAD_count"][0]     #  need to make count not series / ERROR HERE  
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()    

            #unsure weather to just do count but can use len()
            row['PRE_Somatic_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
            row['AP_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
            row['WASH_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()
            row['PRE_pAD_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
            row['AP_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
            row['WASH_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()

        pass


    else:
        raise NotADirectoryError(f"Didn't handle: {row.data_type}") # data_type can be AP, FP_AP or FP
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


def _colapse_to_file_value_FP(celltype_drug_datatype, df, color_dict):
    #this functuon should colaps all metrics to a single value for each file i.e. lists become a mean ect.
    cell_type, drug, data_type = celltype_drug_datatype
    
    if data_type == 'AP': #if data type is not firing properties (FP then return df)
            return df
    elif data_type in ['FP', 'FP_AP']:
        df = df.copy()

        #AP Charecteristics i.e. mean/SD e.c.t. for a single file (replications not combined consider STATISTICAL IMPLICATIONS FORPLOTTING)
        df['mean_voltage_threshold_file'] = df.voltage_threshold.apply(np.mean) #apply 'loops' through the slied df and applies the function np.mean
        df['mean_AP_height_file'] = df.AP_height.apply(np.mean)
        df['mean_AP_slope_file'] = df.AP_slope.apply(np.mean)
        df['mean_AP_width_file'] = df.AP_width.apply(np.mean)
        df['mean_AP_latency_file'] = df.AP_latency.apply(np.mean)
        
        
        #Tau and Sag colapse to lowest value
        extract_truple_data('sag', df) #creating columns 'tau_file' and 'sag_file'
        extract_truple_data('tau_rc', df)
    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP or FP
    return df
    


def _colapse_to_cell_pre_post_FP(cellid_drug_datatype, df, color_dict):
    
    #for each cellid_drug_datatype I need a single value for a metric to do testing on 

    
    cell_id, drug , data_type = cellid_drug_datatype
    
    if data_type == 'AP': #if data type is not firing properties (FP then return df)
        return df
    
    elif data_type == 'FP_AP': #if data type is not firing properties (FP then return df) #later this will be filled with other plots
        return df
   
    elif data_type == 'FP':
        df['max_firing_cell_drug'] = df['max_firing'].mean()
        
        #AP Charecteristics 
        df['voltage_threshold_cell_drug'] = df['mean_voltage_threshold_file'].mean()
        df['AP_height_cell_drug'] = df['mean_AP_height_file'].mean()
        df['AP_slope_cell_drug'] = df['mean_AP_slope_file'].mean()
        df['AP_width_cell_drug'] = df['mean_AP_width_file'].mean()
        df['AP_latency_cell_drug'] = df['mean_AP_latency_file'].mean()
        
        #Tau and Sag 
        df['tau_cell_drug'] = df['tau_rc_file'].mean()
        df['sag_cell_drug'] = df['sag_file'].mean()

    else:
        raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP or FP
    return df 


def _prep_plotwithstats_FP(celltype_datatype, df, color_dict):
    global multi_page_pdf
    
    cell_type, data_type = celltype_datatype
    #if data type is not firing properties (FP then return df)
    if data_type == 'AP': 
        return df
    if data_type == 'FP_AP': 
        return df
    
    print(f'analising... {cell_type}')
    cell_id_list = list(df['cell_ID'].unique())
    
    #tobuild dummy df
    # subset = feature_df_expanded_stats.loc[feature_df_expanded_stats['cell_type']== 'L5a_TLX']
    # subset = subset.loc[subset['data_type']== 'FP']
    # df_only_first_app = subset.loc[subset['application_order'] <= 1]

    build_first_drug_ap_column(df, cell_id_list) #builds df['first_drug_AP']
    
    df_only_first_app = df.loc[df['application_order'] <= 1] #only include first drug application data 
        
    plot_list = [['max_firing_cell_drug', 'Firing (Hz)'], 
                 ['voltage_threshold_cell_drug','Voltage Threshold (mV)'], 
                 ['AP_height_cell_drug', ' AP Height (mV)'], 
                 ['AP_slope_cell_drug', 'AP slope (V/s)'],
                 ['AP_width_cell_drug', 'AP width (s) '],
                 ['AP_latency_cell_drug', 'AP latency ()'],
                 ['tau_cell_drug', 'Tau RC (ms)'],
                 ['sag_cell_drug', 'Percentage sag (%)']
                 ]
    for col_name, name in plot_list:
        
        statistical_df = build_statistical_df(df_only_first_app, cell_id_list, col_name)
            
        #generate order for plotting with only relevant data 
        order_dict = list(color_dict.keys()) #ploting in order of dict keys
        order_me = list(df_only_first_app['drug'].unique())
        order = [x for x in order_dict if x in order_me]
        
        #create df_to_plot with only one value per cell (not per file)
        df_to_plot = df_only_first_app[['cell_ID','drug',col_name, 'first_drug_AP']]
        df_to_plot = df_to_plot.drop_duplicates()
        
        student_t_df = build_student_t_df(statistical_df, cell_type) #calculate and create student t test df to be used to plot stars
        
        fig, axs = plot_sns_swarm_hist(cell_type, df_to_plot, order, color_dict, x='drug', y=col_name, 
                                       hue='first_drug_AP', x_label='Drug applied', y_label=name)
        
        put_significnce_stars(axs, student_t_df, data=df_to_plot, x='drug', y=col_name, order = order) #add stats from student t_test above
        
    
        
        multi_page_pdf.savefig() #save barplot
        plt.close("all")
        # save_df_to_pdf(student_t_df, multi_page_pdf) #save table of p_values to pdf with figures

    return df 



def loopCombinations_stats(df, OUTPUT_DIR):
    global multi_page_pdf
    multi_page_pdf = PdfPages(f'{OUTPUT_DIR}/FP_metrics_.pdf')
    #create a copy of file_folder column to use at end of looping to restore  origional row order !!! NEEDS TO BE DONE
    # df_row_order = df['folder_file']
    
    #finding all combination in df and applying a function to them 
    #keepingin mind that the order is vital  as the df is passed through againeach one
    combinations = [
                    (["cell_type", "drug", "data_type"], _colapse_to_file_value_FP),
                    (["cell_ID", "drug",  "data_type"], _colapse_to_cell_pre_post_FP), #stats_df to be fed to next function mouseline 
                    (["cell_type",  "data_type"], _prep_plotwithstats_FP)
                     
                    # (["cell_ID", "drug", "data_type"], _pAD_detector_AP)
    ]

    for col_names, handlingFn in combinations:
        df = apply_group_by_funcs(df, col_names, handlingFn) #note that as each function is run the updated df is fed to the next function

    multi_page_pdf.close()
    # df[ order(match(df['folder_file'], df_row_order)) ]
    return df


## dependants of above

def build_first_drug_ap_column(df, cell_id_list):
    '''
    Parameters
    ----------
    df : df to add column too
    cell_id_list : list of cell IDs to loop

    Returns
    -------
    None.
    '''
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

    Returns
    -------
    None.

    '''   
    PRE = cell_df.loc[cell_df['drug'] == 'PRE', col_name] #REMOVE HARD CODE
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

def save_df_to_pdf(df, pdf_name):
    '''
    Parameters
    ----------
    df : df to be saved
    pdf_name : name of OPEN multipage.pdf #https://matplotlib.org/stable/gallery/misc/multipage_pdf.html

    Returns
    -------
    None.

    '''
        #https://stackoverflow.com/questions/53828284/how-to-save-pandas-dataframe-into-existing-pdf-from-pdfpages #style
    fig, ax = plt.subplots()# save student_t_test
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.set_fontsize(24)
    table.scale(1,4)
    ax.axis('off')
    pdf_name.savefig()
    plt.close("all")
    return

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

    Returns
    -------
    None.
    '''
    significant_drugs = df.loc[df.p_val <= 0.05, 'drug']
    # print(significant_drugs)
    
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

def build_statistical_df(df_only_first_app, cell_id_list, col_name):  #col_name = 'max_firing_cell_drug'
    PRE_ = []
    POST_ = []
    first_drug_ = []
    lists_to_fill = [PRE_, POST_, first_drug_]
    
    for cell_id in cell_id_list:
        
        cell_df = df_only_first_app.loc[df_only_first_app['cell_ID'] == cell_id] #slice df to cell only 
        first_drug_string = cell_df['first_drug_AP'].unique()[0]

        fill_statistical_df_lists(cell_df, col_name, first_drug_string, lists_to_fill) #col_name = 'max_firing_cell_drug'

    #create statistical df for celltype with all raw data for single value type e.g. max firing
    statistical_df = pd.DataFrame({'cell_ID': cell_id_list,
                                  'drug': first_drug_,
                                  'PRE': PRE_,
                                  'POST': POST_
                                  })
    
    # print(statistical_df)
    return statistical_df

def plot_sns_swarm_hist(cell_type, df_to_plot, order, color_dict, x='drug', y='max_firing_cell_drug', hue='first_drug_AP', x_label='Drug applied', y_label='Firing (Hz)'): #y = col_name
    fig, axs = plt.subplots(1,1, figsize = (20, 10))
    sns.barplot(data = df_to_plot, x=x, y=y,  order=order, palette=color_dict, capsize=.1, 
                         alpha=0.8, errcolor=".2", edgecolor=".2" )
    sns.swarmplot(data = df_to_plot,x=x, y=y, order=order, palette=color_dict,  hue= hue, linewidth=1, linestyle='-')  #, legend=False #would like to remove legend 
    axs.set_xlabel( x_label, fontsize = 20)
    axs.set_ylabel( y_label, fontsize = 20)
    axs.set_title( cell_type +' '+ y_label + '  (CI 95%)', fontsize = 30) 
    return fig, axs


def extract_truple_data(col_name, df): #col_name = 'tau' / 'sag'
    for row_ind, row in df.iterrows(): #truple structure forces loop
        val_lst_file = []
    
        list_of_truples = row[col_name]
        for el in list_of_truples:
            val_lst_file.append( el.val) # truple has: val , steady_state and current_inj
        
        if col_name == 'sag':
            df.at[row_ind, col_name+'_file'] = 100 * np.mean(val_lst_file)
        else:
            df.at[row_ind, col_name+'_file'] = np.mean(val_lst_file)
        
    return




# ##OLD TO DELETE

# def _getstats_FP(mouseline_drug_datatype, df, color_dict):
    
#     mouse_line, drug, data_type = mouseline_drug_datatype
    
#     if data_type == 'AP': #if data type is not firing properties (FP then return df)
#             return df
#     elif data_type in ['FP', 'FP_AP']:    
#         df = df.copy()
#         #for entire mouseline_drug combination
#         df['mean_max_firing'] = df['max_firing'].mean() #dealing with pd.series a single df column
#         df['SD_max_firing'] = df['max_firing'].std()
        
#         df['mean_FI_slope'] = df['FI_slope'].mean() #dealing with pd.series a single df column
#         df['SD_FI_slope'] = df['FI_slope'].std()
        
#         df['mean_rheobased_threshold'] =df['rheobased_threshold'].mean()
        
#         #AP Charecteristics i.e. mean/SD e.c.t. for a single file (replications not combined consider STATISTICAL IMPLICATIONS FORPLOTTING)
#         df['mean_voltage_threshold_file'] = df.voltage_threshold.apply(np.mean) #apply 'loops' through the slied df and applies the function np.mean
#         df['SD_voltage_threshold_file'] = df.voltage_threshold.apply(np.std)
        
#         df['mean_AP_height_file'] = df.AP_height.apply(np.mean)
        
#         df['mean_AP_slope_file'] = df.AP_slope.apply(np.mean)
        
#         df['mean_AP_width'] = df.AP_width.apply(np.mean)
        
#     else:
#         raise NotADirectoryError(f"Didn't handle: {data_type}") # data_type can be AP, FP_AP or FP
#     return df



# def _plotwithstats_FP(mouseline_datatype, df, color_dict):
    
    
#     mouse_line, data_type = mouseline_datatype
#     order = list(color_dict.keys()) #ploting in order of dict keys
    
#     if data_type == 'AP': #if data type is not firing properties (FP then return df)
#         return df
#     if data_type == 'FP_AP': #if data type is not firing properties (FP then return df)
#         return df
    
#     # generating plot by single file values e.g. max_firing or mean_voltage_threshold_file
    
#     factors_to_plot = ['max_firing', 'rheobased_threshold', 'FI_slope', 'mean_voltage_threshold_file', 'mean_AP_height_file', 'mean_AP_slope_file', 'mean_AP_width']
#     names_to_plot = ['_max_firing_Hz', 'rheobased_threshold_pA' , '_FI_slope_linear', '_voltage_threshold_mV', '_AP_height_mV', '_AP_slope', '_AP_width_ms']
    
#     for _y, name in zip(factors_to_plot, names_to_plot):
    
#         sns_barplot_swarmplot (df, order, mouse_line, _x='drug', _y=_y, name = name)
    

#     plt.close("all") #close open figures
    
#     return df 

# def sns_barplot_swarmplot (df, order, mouse_line, _x='drug', _y='max_firing', name = '_max_firing_Hz'):
#     '''
#     Generates figure and plot and saves to patch_daddy_output/
#     '''
#     fig, axs = plt.subplots(1,1, figsize = (20, 10))
#     sns.barplot(data = df, x=_x, y=_y,  order=order, palette=color_dict, capsize=.1, 
#                              alpha=0.8, errcolor=".2", edgecolor=".2" )
#     sns.swarmplot(data = df,x=_x, y=_y, order=order, palette=color_dict, linewidth=1, linestyle='-')
#     axs.set_xlabel( "Drug applied", fontsize = 20)
#     axs.set_ylabel( name, fontsize = 20)
#     axs.set_title( mouse_line + name + '  (CI 95%)', fontsize = 30)
    
#     fig.savefig('patch_daddy_output/' + mouse_line + name + '.pdf')
    
#     return

# def loopCombinations_stats(df):
#     #create a copy of file_folder column to use at end of looping to restore  origional row order !!! NEEDS TO BE DONE
    
#     #keepingin mind that the order is vital  as the df is passed through againeach one
#     combinations = [
#                     (["mouseline", "drug", "data_type"], _getstats_FP), #stats_df to be fed to next function mouseline 
#                     (["mouseline",  "data_type"], _plotwithstats_FP) #finding all combination in df and applying a function to them #here could average and plot or add to new df for stats
#                     # (["cell_ID", "drug", "data_type"], _plotstats_FP)
#     ]
    
#     # pdf = PdfPages('patch_daddy_output.pdf') # open pdf for plotsto be saved in
#     # pdf.savefig(fig)
#     # pdf.close()

#     for col_names, handlingFn in combinations:

    
#         df = apply_group_by_funcs(df, col_names, handlingFn) #note that as each function is run the updated df is fed to the next function
    
    
    
#     return df