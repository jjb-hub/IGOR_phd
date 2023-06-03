    # -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:58:27 2023

@author: Debapratim Jana, Jasmine Butler
"""
#%%

from utils.helper_functions import loopCombinations, loopCombinations_stats
from utils import igor_utils 
from ephys import ap_functions
import openpyxl
import pandas as pd
import numpy as np
import os
import warnings

os.environ["OMP_NUM_THREADS"] ='3'
warnings.filterwarnings('ignore')
   
#setters 

# color_dict = {"PRE":"black", "CONTROL": 'grey', "TCB2":'green', "DMT":"teal", "PSIL":"orange", "LSD":"purple", "MDL":'blue'}  
#hard coded in helper funftions need to fix!
 

#%% LOAD FEATURE_DF + START INPUTS


ROOT = os.getcwd() #This gives terminal location (terminal working dir)
INPUT_DIR = f'{ROOT}/input'
OUTPUT_DIR = f'{ROOT}/output'

feature_df = pd.read_excel (f'{INPUT_DIR}/feature_df_py.xlsx') 

data_path = f'{INPUT_DIR}/PatchData/' #THIS IS HARD CODED INTO make_path(file_folder)


#%% RUN PLOTS
#drug_aplication_visualisation(feature_df, color_dict) # generates PDF of drug aplications # in plotters in utils

#plot_all_FI_curves(feature_df,  color_dict)  # generates PDF with all FI curves for single cell labed with drug and aplication order #### MAKE HZ NOT APs per sweep also isnt it in pA not nA??

#plot_FI_AP_curves(feature_df) #generated PDF with FI-AP for each cell


#%% Extrapolate data from files
feature_df_ex = feature_df.copy()
feature_df_expanded_raw = loopCombinations(feature_df_ex)  #in helper functions #check dif in make_path and passing of directory

#%%

#Do statistical anlaysis of FP data and plot
multi_page_pdf = None #https://matplotlib.org/stable/gallery/misc/multipage_pdf.html
feature_df_expanded_stats = loopCombinations_stats(feature_df_expanded_raw, OUTPUT_DIR)


print(feature_df_expanded_raw)
