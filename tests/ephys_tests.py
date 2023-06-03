# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:16:00 2023

@author: Debapratim Jana, Jasmine Butler
"""
from utils.helper_functions import loopCombinations, loopCombinations_stats #only works in this order for me DJ had lower idk how
from ephys import ap_functions
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
   

#%% LOAD FEATURE_DF + START INPUTS



ROOT = os.getcwd() #This gives terminal location (terminal working dir)
INPUT_DIR = f'{ROOT}/input'
OUTPUT_DIR = f'{ROOT}/output'

feature_df = pd.read_excel (f'{INPUT_DIR}/feature_df_py.xlsx') 

data_path = f'{INPUT_DIR}/PatchData/' #THIS IS HARD CODED INTO make_path(file_folder)

feature_df_ex = feature_df.copy()

#feature_df_expanded_raw = loopCombinations(feature_df_ex)
#feature_df_expanded_stats = loopCombinations_stats(feature_df_expanded_raw)
#feature_df_ex_tau = loopCombinations(feature_df_ex)


#%%TEST PATHS / FUNCS

# #FP tester paths 
# path_I =  '/Users/jasminebutler/Desktop/IGOR_phd/input/PatchData/JJB221230/t15Soma.ibw'

# path_V =  '/Users/jasminebutler/Desktop/IGOR_phd/input/PatchData/JJB221230/t15Soma.ibw'


#AP tester paths
path_I =  f'{INPUT_DIR}/PatchData/JJB230110/t6Soma_outwave.ibw'
path_V = f'{INPUT_DIR}/PatchData/JJB230110/t6Soma.ibw'


#to plot I steps or V responce to steps all in 1 
_, dfV = igor_exporter(path_V)
_, dfI = igor_exporter(path_I)


V = np.array(dfV)



# #ap_functions.ap_characteristics_extractor_main(V, 0)
# for idx in range(V.shape[-1]):
#     print("sweep %s " % idx)
#     ap_functions.ap_characteristics_extractor_subroutine_derivative(V, idx)

# idx = 38
# ap_functions.ap_characteristics_extractor_subroutine_derivative(V, idx)
# plt.plot(V[:,idx])
# plt.show()



# ap_functions.ap_characteristics_extractor_main(V, all_sweeps=True)


peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df, X_pca   =  ap_functions.pAD_detection(V, pca_plotting = False , kmeans_plotting = False , histogram_plotting=False)

drug_in = 16
drug_out = 28

PRE_Somatic_AP_locs = pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] < drug_in), 'AP_loc'].tolist()
AP_Somatic_AP_locs = pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] >= drug_in) & (pAD_df['AP_sweep_num'] <= drug_out), 'AP_loc'].tolist()
WASH_Somatic_AP_locs = pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] > drug_out), 'AP_loc'].tolist()


pAD_df

# #%% TESTING FUNCTIONS


# def plot_single_df_I_or_V_by_col (df_to_plot, y_label = 'V (mV) or I injected (pA)'): 
#     '''
#     Opens a single figure and plots each column on top of eachother (nice for I FP data)
#     Parameters
#     ----------
#     df_to_plot : df where sweeps are columns
#     y_label : string -   'V (mV)' or 'I injected (pA)'

#     Returns
#     -------
#     plot

#     '''
#     fig, ax = plt.subplots(1,1, figsize = (10, 8))
#     for i in df_to_plot.columns:
#     # for i in [ 14]:
#         print('I is = ', i)
#         x_ = df_to_plot.iloc[:,i].tolist() #create list of values for each col 
#         #quick plot all I steps free of APs
#         y_to_fit = np.arange(len(x_)) * 0.0001 #sampeling at 10KHz will give time in seconds
#         ax.plot(y_to_fit, x_) 
#         ax.set_xlabel( "Time (s)", fontsize = 15)
#         ax.set_ylabel( y_label, fontsize = 15)
        
#     return

# def single_drug_aplication_visualisation(feature_df,  color_dict, cell = 'DRD230104b', drug = 'TCB2'):
#     '''
#     Generates cell + '_' + drug + '_sweeps.pdf' with each sweep from IGOR displayed on a single page, colour change at drug aplication (voltage data )

#     Parameters
#     ----------
#     feature_df : df including all factors needed to distinguish data 
#     color_dict : dict with a colour for each drug to be plotted

#     Returns
#     -------
#     None.

#     '''
    
#     start = timeit.default_timer()
#     with PdfPages(cell + '_' + drug + '_sweeps.pdf') as pdf:
        
#         aplication_df = feature_df[feature_df.data_type == 'AP'] #create sub dataframe of aplications
        
#         for row_ind, row in aplication_df.iterrows():  #row is a series that can be called row['colname']
#             if row['cell_ID'] == cell and row['drug'] == drug:
#                 path_V, path_I = make_path(row['folder_file'])
#                 print (path_V)
#                 _ , y_df = igor_exporter(path_V)
                
#                 drug_in = row['drug_in']
#                 drug_out = row['drug_out']
                
#                 for col in  y_df.columns:
#                     fig, axs = plt.subplots(1,1, figsize = (10, 5))
#                     y = y_df[col]
#                     x = np.arange(len(y)) * 0.0001 #sampeling at 10KHz frequency if igor?
#                     sweep = col+1
                    
#                     if drug_in <= sweep <= drug_out:
#                         axs.plot(x,y, c = color_dict[row['drug']], lw=1)
#                     else:
#                         axs.plot(x,y, c = color_dict['PRE'], lw=1)
    
#                     axs.set_xlabel( "Time (s)", fontsize = 15)
#                     axs.set_ylabel( "Membrane Potential (mV)", fontsize = 15)
#                     axs.set_title(row['cell_ID'] + ' '+ row['drug'] +' '+ " Aplication" + " (" + str(row['aplication_order']) + ")", fontsize = 25)
#                     pdf.savefig(fig)
#                     plt.close("all") 

#     stop = timeit.default_timer()
#     print('Time: ', stop - start)      
#     return

