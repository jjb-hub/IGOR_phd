# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:16:00 2023

@author: Debapratim Jana, Jasmine Butler
"""
from utils.base_utils import *
from utils.mettabuild_functions import  expandFeatureDF , generate_V_pAD_df #only works in this order for me DJ had lower idk how
from ephys import ap_functions
from utils.plotter import buildApplicationFig, buildMeanAPFig 
import os, shutil, itertools, json, timeit, functools, pickle
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils


def function_tester_all_files(feature_df, test_function = None):
    
    # all we need to do is just specify the function we wanna to test and this just iterates through entire feature xlsx.
    
    if test_function is None: 
        raise Exception("Test function not chosen")
        sys.exit(1)
    
    for nrow in range(feature_df.shape[0]):
        
        folder_file	 , cell_ID	, data_type	  , drug 	 , replication_no	,application_order	, drug_in	,drug_out	, I_setting ,	R_series_Mohm , _  = feature_df.loc[nrow + 1]
        
        # most test functions look at AP files?? 
        
        if data_type == 'AP':
            print("Analysing %s " % folder_file)
            
            if test_function == 'depol': 
                if  type(drug_in) is int :
                    pAD_df, V_array = generate_V_pAD_df(folder_file) 
                    depol_val = ap_functions.cell_membrane_polarisation_detector_new(folder_file =  folder_file, cell_ID = cell_ID,   drug = drug ,  drug_in = drug_in  , drug_out = drug_out ,  application_order=  application_order,  pAD_locs= pAD_df[pAD_df['pAD'] ==  'pAD']['AP_loc'] , I_set = I_setting ) 
                    
            
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
                buildMeanAPFig(cell_ID, pAD_df, V_array)
            elif test_function == 'beta_pAD': 
                pAD_df, V_array = generate_V_pAD_df(folder_file)
                ap_functions.beta_pAD_detection(V_array)
                
    return None 

def function_tester_single_file(folder_file, cell_ID, test_function = None):
    if test_function is None: 
        raise Exception("Test function not chosen")
        sys.exit(1)
    
    
    # most test functions look at AP files?? 
    
    # if data_type == 'AP':
    #     print("Analysing %s " % folder_file)
        
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
        buildMeanAPFig(cell_ID, pAD_df, V_array)
    
    return None 
        
    
def quick_debugger(folder_file): 
    pAD_df, V_array = generate_V_pAD_df(folder_file)
    for idx in range(V_array.shape[-1]):
        plt.plot(V_array[:, idx])
    plt.show()
    print('Number of APs are %s' % ap_functions.num_ap_finder(V_array))
    return None 



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



def path_to_array_wrapper(input_directory, filename_trace):
    '''
    
    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.
    filename_trace : TYPE
        DESCRIPTION.

    Returns
    -------
    V : TYPE
        DESCRIPTION.
    I : TYPE
        DESCRIPTION.

    '''
    path_V, path_I  =  f'{INPUT_DIR}/PatchData/' + filename_trace + 'Soma.ibw' ,   f'{INPUT_DIR}/PatchData/' + filename_trace  + 'Soma_outwave.ibw'
    listV, dfV = igor_exporter(path_V)
    listI, dfI = igor_exporter(path_I)
    V = np.array(dfV)
    I = np.array(dfI)
    
    return V, I 
 
