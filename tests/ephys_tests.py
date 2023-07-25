# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:16:00 2023

@author: Debapratim Jana, Jasmine Butler
"""
from utils.base_utils import *
from utils.metabuild_functions import  expandFeatureDF , generate_V_pAD_df #only works in this order for me DJ had lower idk how
from ephys import ap_functions
from utils.plotter import buildApplicationFig, buildAP_MeanFig 
import os, shutil, itertools, json, timeit, functools, pickle
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils

#%% TESTING FUNCTIONS
# single_drug_aplication_visualisation(feature_df,  color_dict, cell = 'DRD230104b', drug = 'TCB2')
# plot_single_df_I_or_V_by_col (df_to_plot, y_label = 'V (mV) or I injected (pA)')

def function_tester_all_files(feature_df, test_function = None):
    
    # all we need to do is just specify the function we wanna to test and this just iterates through entire feature xlsx.
    
    if test_function is None: 
        raise Exception("Test function not chosen")
        sys.exit(1)
    
    for nrow in range(feature_df.shape[0]):
        
        folder_file	 , cell_ID	, data_type	  , drug 	 , replication_no	,aplication_order	, drug_in	,drug_out	, I_setting ,	R_series_Mohm , _  = feature_df.loc[nrow + 1]
        
        # most test functions look at AP files?? 
        
        if data_type == 'AP':
            print("Analysing %s " % folder_file)
            
            if test_function == 'AP_figure': 
            
                buildApplicationFig(color_dict, cell_ID = cell_ID , folder_file = folder_file	 , drug_in = drug_in, drug_out= drug_out ,I_set = I_setting )
            elif  test_function ==  'PCA' : 
                pAD_df, V_array = generate_V_pAD_df(folder_file)
            elif  test_function == 'Phase':
                pAD_df, V_array = generate_V_pAD_df(folder_file)
            elif  test_function == 'Histogram':
                pAD_df, V_array = generate_V_pAD_df(folder_file)
            elif  test_function ==  'Mean_AP':
                pAD_df, V_array = generate_V_pAD_df(folder_file)
                buildAP_MeanFig(cell_ID, pAD_df, V_array)
    return None 

filename = "feature_df_py.xlsx"  # df of files and factors
expanded_df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
function_tester_all_files(expanded_df, 'Mean_AP')

#  test buildMeanAPFig

#### JJB TESTERS 
def plot_single_df_I_or_V_by_col (df_to_plot, y_label = 'V (mV) or I injected (pA)'): 
    '''
    Opens a single figure and plots each column on top of eachother (nice for I FP data)
    Parameters
    ----------
    df_to_plot : df where sweeps are columns
    y_label : string -   'V (mV)' or 'I injected (pA)'

    Returns
    -------
    plot

    '''
    fig, ax = plt.subplots(1,1, figsize = (10, 8))
    for i in df_to_plot.columns:
    # for i in [ 14]:
        print('I is = ', i)
        x_ = df_to_plot.iloc[:,i].tolist() #create list of values for each col 
        #quick plot all I steps free of APs
        y_to_fit = np.arange(len(x_)) * 0.0001 #sampeling at 10KHz will give time in seconds
        ax.plot(y_to_fit, x_) 
        ax.set_xlabel( "Time (s)", fontsize = 15)
        ax.set_ylabel( y_label, fontsize = 15)
        
    return

def single_drug_aplication_visualisation(feature_df,  color_dict, cell = 'DRD230104b', drug = 'TCB2'):
    '''
    Generates cell + '_' + drug + '_sweeps.pdf' with each sweep from IGOR displayed on a single page, colour change at drug aplication (voltage data )

    Parameters
    ----------
    feature_df : df including all factors needed to distinguish data 
    color_dict : dict with a colour for each drug to be plotted

    Returns
    -------
    None.

    '''
    
    start = timeit.default_timer()
    with PdfPages(cell + '_' + drug + '_sweeps.pdf') as pdf:
        
        aplication_df = feature_df[feature_df.data_type == 'AP'] #create sub dataframe of aplications
        
        for row_ind, row in aplication_df.iterrows():  #row is a series that can be called row['colname']
            if row['cell_ID'] == cell and row['drug'] == drug:
                path_V, path_I = make_path(row['folder_file'])
                print (path_V)
                _ , y_df = igor_exporter(path_V)
                
                drug_in = row['drug_in']
                drug_out = row['drug_out']
                
                for col in  y_df.columns:
                    fig, axs = plt.subplots(1,1, figsize = (10, 5))
                    y = y_df[col]
                    x = np.arange(len(y)) * 0.0001 #sampeling at 10KHz frequency if igor?
                    sweep = col+1
                    
                    if drug_in <= sweep <= drug_out:
                        axs.plot(x,y, c = color_dict[row['drug']], lw=1)
                    else:
                        axs.plot(x,y, c = color_dict['PRE'], lw=1)
    
                    axs.set_xlabel( "Time (s)", fontsize = 15)
                    axs.set_ylabel( "Membrane Potential (mV)", fontsize = 15)
                    axs.set_title(row['cell_ID'] + ' '+ row['drug'] +' '+ " Aplication" + " (" + str(row['aplication_order']) + ")", fontsize = 25)
                    pdf.savefig(fig)
                    plt.close("all") 

    stop = timeit.default_timer()
    print('Time: ', stop - start)      
    return

#%% LOAD FEATURE_DF + START INPUTS
ROOT = os.getcwd() #This gives terminal location (terminal working dir)
INPUT_DIR = f'{ROOT}/input'
OUTPUT_DIR = f'{ROOT}/output'
CACHE_DIR = f'{INPUT_DIR}/cache'

color_dict = {"pAD":"orange","Somatic":"blue","WASH":"lightsteelblue", "PRE":"black", "CONTROL": 'grey', "TCB2":'green', "DMT":"teal", "PSIL":"orange", "LSD":"purple", "MDL":'blue', 'I_display':'cornflowerblue'} 

 
 

dj_xlsx = False

if dj_xlsx:
    feature_df = pd.read_excel (f'{INPUT_DIR}/feature_df_py.xlsx')
    data_path = f'{INPUT_DIR}/PatchData/' #THIS IS HARD CODED INTO make_path(file_folder)
else:
    feature_df = openpyxl.load_workbook (f'{INPUT_DIR}/feature_df_py.xlsx') 
    data = feature_df['Sheet1'].values
    feature_df = pd.DataFrame(data)
    header = feature_df.iloc[0]
    feature_df = feature_df[1:]


#%%TEST PATHS / FUNCS

# #FP tester paths 
# path_I =  f'{INPUT_DIR}/PatchData/JJB230509/t14Soma_outwave.ibw'
# path_V =  f'{INPUT_DIR}/PatchData/JJB230509/t14Soma.ibw'


#AP tester paths
path_I =  f'{INPUT_DIR}/PatchData/JJB230411/t10Soma_outwave.ibw'
path_V = f'{INPUT_DIR}/PatchData/JJB230411/t10Soma.ibw'


#to plot I steps or V responce to steps all in 1 
listV, dfV = igor_exporter(path_V)
listI, dfI = igor_exporter(path_I)
V = np.array(dfV)
I = np.array(dfI)

peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df   = ap_functions.pAD_detection(V)

utils.plotter.buildPhasePlotFig('TLX230411a', pAD_df, V)
utils.plotter.buildpADHistogram('TLX230411a', pAD_df, V)



# #ap_functions.ap_characteristics_extractor_main(V, 0)
# for idx in range(V.shape[-1]):
#     print("sweep %s " % idx)
#     ap_functions.ap_characteristics_extractor_subroutine_derivative(V, idx)

# idx = 38
# ap_functions.ap_characteristics_extractor_subroutine_derivative(V, idx)
# plt.plot(V[:,idx])
# plt.show()

x, y, v_rest = ap_functions.extract_FI_x_y (path_I, path_V)




