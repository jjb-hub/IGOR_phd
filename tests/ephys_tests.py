# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:16:00 2023

@author: Debapratim Jana, Jasmine Butler
"""

from ephys import ap_functions
from utils.helper_functions import loopCombinations, loopCombinations_stats, igor_exporter
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

   

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

#feature_df_expanded_raw = loopCombinations(feature_df_ex)
#feature_df_expanded_stats = loopCombinations_stats(feature_df_expanded_raw)
#feature_df_ex_tau = loopCombinations(feature_df_ex)


#%%TEST PATHS / FUNCS

#FP tester paths 
path_I =  '/Users/debap/Desktop/PatchData/JJB221230/t15Soma_outwave.ibw'
path_V =  '/Users/debap/Desktop/PatchData/JJB221230/t15Soma.ibw'





#AP tester paths
path_I =  '/Users/debap/Desktop/PatchData/JJB210209/t14Soma_outwave.ibw'
path_V = '/Users/debap/Desktop/PatchData/JJB210209/t14Soma.ibw'

#to plot I steps or V responce to steps all in 1 
_, dfV = igor_exporter(path_V)
_, dfI = igor_exporter(path_I)


V = np.array(dfV)



#ap_functions.ap_characteristics_extractor_main(V, 0)
for idx in range(V.shape[-1]):
    print("sweep %s " % idx)
    ap_functions.ap_characteristics_extractor_subroutine_derivative(V, idx)
    #plt.plot(V[:,idx])



# ap_functions.ap_characteristics_extractor_main(V, all_sweeps=True)


peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df, X_pca   =  ap_functions.pAD_detection(V)
#pAD_df
