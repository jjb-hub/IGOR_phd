# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:16:00 2023

@author: Debapratim Jana, Jasmine Butler
"""
from utils.base_utils import *
from utils.mettabuild_functions import loopCombinations, expandFeatureDF #only works in this order for me DJ had lower idk how
from ephys import ap_functions

import os, shutil, itertools, json, timeit, functools, pickle
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

   

#%% LOAD FEATURE_DF + START INPUTS
ROOT = os.getcwd() #This gives terminal location (terminal working dir)
INPUT_DIR = f'{ROOT}/input'
OUTPUT_DIR = f'{ROOT}/output'
CACHE_DIR = f'{INPUT_DIR}/cache'

if True:
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
path_I =  f'{INPUT_DIR}/PatchData/JJB230110/t6Soma_outwave.ibw'
path_V = f'{INPUT_DIR}/PatchData/JJB230110/t6Soma.ibw'


#to plot I steps or V responce to steps all in 1 
listV, dfV = igor_exporter(path_V)
listI, dfI = igor_exporter(path_I)
V = np.array(dfV)
I = np.array(dfI)



# #ap_functions.ap_characteristics_extractor_main(V, 0)
# for idx in range(V.shape[-1]):
#     print("sweep %s " % idx)
#     ap_functions.ap_characteristics_extractor_subroutine_derivative(V, idx)

# idx = 38
# ap_functions.ap_characteristics_extractor_subroutine_derivative(V, idx)
# plt.plot(V[:,idx])
# plt.show()

x, y, v_rest = ap_functions.extract_FI_x_y (path_I, path_V)



#%% TESTING FUNCTIONS
# single_drug_aplication_visualisation(feature_df,  color_dict, cell = 'DRD230104b', drug = 'TCB2')
# plot_single_df_I_or_V_by_col (df_to_plot, y_label = 'V (mV) or I injected (pA)')


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

