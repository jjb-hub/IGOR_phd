########## GETTERS and BUILDERS of raw data ##########

from module.constants import INPUT_DIR
from module.utils import *
from module.action_potential_functions import (
    calculate_max_firing, 
    extract_FI_x_y, 
    extract_FI_slope_and_rheobased_threshold, 
    ap_characteristics_extractor_main,
    tau_analyser,
    pAD_detection,
    sag_current_analyser 
    )
from module.metadata import build_cell_type_dict

import pandas as pd
import numpy as np 






 ########## DICT ORGANISATION ##########

def getorbuild_cell_type_dict(filename, from_scratch=None): #may be redundant dont know if will use or just to check input file
    filename_no_extension = filename.split(".")[0]
    identifier = 'cell_type_dict'
    from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, identifier):
        print(f'BUILDING "{identifier}"') 
        cell_type_dict = build_cell_type_dict(filename)
        subcache_dir = f"{CACHE_DIR}/{filename.split('.')[0]}"
        saveJSON(f"{subcache_dir}/cell_type_dict.json", cell_type_dict)
        print (f"CELL TYPE DICT {cell_type_dict} SAVED TO {subcache_dir}")
    else:
        getJSON(f"{subcache_dir}/cell_type_dict.json")
    return cell_type_dict


############ BUILDERS ##########


#feature df (row = single file) and extractes values based on the data type  FP or AP then appends values to df
def _handleFile(row): #THIS FUNCTION IS TOO BIG AND TOO SLOW TODO 
    row = row.copy() #isnt this redundant? JJB
    
    path_V, path_I = make_path(row.folder_file) 
    V_list, V_df = igor_exporter(path_V)
    V_array      = np.array(V_df) 
    try: # handel missing I traces (Soma_outwave)
        I_list, I_df = igor_exporter(path_I)
        I_array      = np.array(I_df)
    except FileNotFoundError:
        print('I file not found, path:', path_I)

# FP data handeling
    if row.data_type in ["FP", "FP_AP"]: 
        # pass 
        print("FP type file")
        print(row.folder_file)

        #data files to be fed into extraction funcs feature_df_extended
        row["max_firing"] = calculate_max_firing(V_array)
        
        x, y, v_rest = extract_FI_x_y (path_I, path_V) # x = current injected, y = number of APs, v_rest = the steady state voltage (no I injected)
        FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True) #FIX ME: handel pAD APs better slope fit
        row["rheobased_threshold"] = rheobase_threshold
        row["FI_slope"] = FI_slope

        peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(V_df)
        row["voltage_threshold"] = v_thresholds_all 
        row["AP_height"] = peak_heights_all
        row["AP_width"] = peak_fw_all
        row["AP_slope"] = peak_slope_all 
        row["AP_latency"] = peak_latencies_all

        #to DJ: to remove the named tuples I just mad them lists of lists i.e. [[value, steady_state, I_injected], [value, steady_state, I_injected] ]
        row["tau_rc"] = tau_analyser(V_array, I_array, x, plotting_viz= False, analysis_mode = 'max')
        row["sag"]    = sag_current_analyser(V_array, I_array, x)

# APP handeling
    elif row.data_type == "AP":

        print("AP type file")
        print(row.folder_file)        
        
        # pAD classification
        peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  =   pAD_detection(V_array)
        print('pAD detection complete .... ')

        if len(pAD_df["pAD_count"]) == 0:  #if there are no APs of any kind detected
            row["pAD_count"] =  np.nan
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()   # getlocations of ALL? APs : AP_lcos 
        else:
            row["pAD_count"] =  pAD_df["pAD_count"][0]     #  need to make count not series / ERROR HERE  
            row["AP_locs"]   =  pAD_df["AP_loc"].ravel()    

            #unsure weather to just do count but can use len()
            row['PRE_Somatic_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
            row['APP_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
            row['WASH_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()
            row['PRE_pAD_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
            row['APP_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
            row['WASH_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()

        #make points list for PRE , APP and WASH voltages #THIS IS TOO BIG EXTRACT RELEVANT DATA i.e. SD and add 
        #saving full points list is too long! #to float 16 instead of 32 
        #FIX ME 
        # varray[ rows to take   : cols to take ]
        # varray[:, :row.drug_in] #take all rows up to drug_in

        # row['PRE_V'] = V_array[:, :row.drug_in] 
        # row['APP_V'] = V_array[:, row.drug_in:row.drug_out]
        # row['WASH_V'] = V_array[:, row.drug_out:]
        pass
    
    elif row.data_type == "pAD":
        print('data_type pAD')

    else:
        raise NotADirectoryError(f"Didn't handle: {row.data_type}") # data_type can be AP, FP_AP or FP
    return row


def expandFeatureDF(filename_or_df): #if passed in get or build will be filename therefor cant subset to try #modularising now
    if not isinstance(filename_or_df, pd.DataFrame):
        print('fetchig raw df')
        df = getRawDF(filename_or_df)
    else:
        df=filename_or_df

    og_columns = df.columns.copy() #origional columns #for ordering columns
    df['mouseline'] = df.cell_id.str[:3]
    df = df.apply(_handleFile, axis=1) #Apply a function along an axis (rows = 1) of the DataFrame

    #ORDERING DF internal: like this new columns added will appear at the end of the df in the order they were created in _handelfile()
    all_cur_columns = df.columns.copy()
    new_colums_set = set(all_cur_columns ) - set(og_columns) # Subtract mathematical sets of all cols - old columns to get a set of the new columns
    new_colums_li = list(new_colums_set)
    reorder_cols_li =  list(og_columns) + new_colums_li # ['mouseline'] + ordering
    df_reordered = df[reorder_cols_li]    
    return df_reordered


def build_cell_type_expandedFeatureDF(filename, cell_type):
    df = getRawDF(filename)
    cell_type_df = subselectDf(df, {'cell_type':[cell_type]})
    cell_type_df_expanded = expandFeatureDF(cell_type_df)
    return cell_type_df_expanded


########## GETTERS ##########

#takes filename or df,  the cell id and optional data type returns subset of df
def getCellDF(filename_or_df, cell_id, data_type = None):
    if not isinstance(filename_or_df, pd.DataFrame):
        df=getRawDF(filename_or_df)
    else:
        df=filename_or_df

    cell_df = df[df['cell_id']==cell_id]
    if data_type is not None:
        cell_df = cell_df[cell_df['data_type']== data_type]
    return cell_df

#expands featureDF into expandedDF if specififed for only a single cell_type 
def getorbuildExpandedDF(filename, identifier, builder_cb, cell_type=None, from_scratch=None):
    filename_no_extension = filename.split(".")[0]
    from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, identifier):
        print(f'BUILDING "{identifier}"')    
        if cell_type: 
            df = builder_cb(filename, cell_type) #build_cell_type_expandedFeatureDF will be the builder_cb
        else:
            df = builder_cb(filename) # expandFeatureDF will be the builder_cb
        cache(filename_no_extension, identifier, df)
    else : df = getCache(filename, identifier)
    return df

