# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:58:27 2023

@author: Debapratim Jana, Jasmine Butler
"""


from utils.helper_functions import loopCombinations, loopCombinations_stats, igor_exporter
from ephys import ap_functions
import openpyxl
import pandas as pd
import numpy as np
import os



#%% LOAD FEATURE_DF + START INPUTS


ROOT = os.getcwd() #This gives terminal location (terminal working dir)
INPUT_DIR = f'{ROOT}/input'
OUTPUT_DIR = f'{ROOT}/output'

feature_df = pd.read_excel (f'{INPUT_DIR}/feature_df_py.xlsx') 

data_path = f'{INPUT_DIR}/PatchData/' #THIS IS HARD CODED INTO make_path(file_folder)

#REDUNDANT
# base_path = '/Users/debap/Desktop/PatchData/'
# path_to_data = '/Users/debap/Desktop/PatchData/'  # change to local
# feature_df = openpyxl.load_workbook ('/Users/debap/Desktop/PatchData/feature_df_py.xlsx') 
# data = feature_df['Sheet1'].values
# feature_df = pd.DataFrame(data)
# header = feature_df.iloc[0]
# feature_df = feature_df[1:]
# feature_df.columns = header

#### RUN PLOTS

#drug_aplication_visualisation(feature_df, color_dict) # generates PDF of drug aplications

#plot_all_FI_curves(feature_df,  color_dict)  # generates PDF with all FI curves for single cell labed with drug and aplication order #### MAKE HZ NOT APs per sweep also isnt it in pA not nA??

#plot_FI_AP_curves(feature_df) #generated PDF with FI-AP for each cell


### Extrapolate data from files
feature_df_ex = feature_df.copy()

feature_df_expanded_raw = loopCombinations(feature_df_ex)  #in helper functions
#feature_df_expanded_stats = loopCombinations_stats(feature_df_expanded_raw)
#feature_df_ex_tau = loopCombinations(feature_df_ex)


