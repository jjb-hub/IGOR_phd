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

   

#%% LOAD FEATURE_DF + START INPUTS

base_path = '/Users/debap/Desktop/PatchData/'
path_to_data = '/Users/debap/Desktop/PatchData/'  # change to local



#feature_df = pd.read_excel ('/Users/debap/Desktop/PatchData/feature_df_py.xlsx') 
feature_df = openpyxl.load_workbook ('/Users/debap/Desktop/PatchData/feature_df_py.xlsx') 
data = feature_df['Sheet1'].values
feature_df = pd.DataFrame(data)
header = feature_df.iloc[0]
feature_df = feature_df[1:]
feature_df.columns = header


feature_df_ex = feature_df.copy()

feature_df_expanded_raw = loopCombinations(feature_df_ex)
#feature_df_expanded_stats = loopCombinations_stats(feature_df_expanded_raw)
#feature_df_ex_tau = loopCombinations(feature_df_ex)


