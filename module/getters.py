

from module.constants import INPUT_DIR
from module.utils import *
from module.action_potential_functions import (
    calculate_max_firing, 
    extract_FI_x_y, 
    extract_FI_slope_and_rheobased_threshold, 
    ap_characteristics_extractor_main,
    tau_analyser,
    pAD_detection,
    sag_current_analyser,
    mean_inputR_APP_calculator,
    mean_RMP_APP_calculator, 
    )

import pandas as pd
import numpy as np 
import traceback







########## GETTERS ##########


def getRawDf(filename):
    return getOrBuildDf(filename, "feature_df", buildRawDf)

def getExpandedDf(filename):
    return getOrBuildDf(filename, "expanded_df", buildExpandedDF)

def getExpandedSubsetDf(filename, cell_type, from_scratch=None):
    return getorbuildSubselectExpandedDF(filename, f"{cell_type}_expanded_df", buildExpandedDF, cell_type, from_scratch)

def getFPStats(filename):
    return getOrBuildDf(filename, "FP_stats", buildFPStatsDf)

def getAPPStats(filename):
    return getOrBuildDf(filename, "APP_stats", buildAPPStatsDf)


def getCellDf(filename_or_df, cell_id, data_type = None):
    df, filename=getExpandedDfIfFilename(filename_or_df)
    cell_df = df[df['cell_id']==cell_id]
    if data_type is not None:
        cell_df = cell_df[cell_df['data_type']== data_type]
    return cell_df





def getOrBuildDf(filename, df_identifier, builder_cb):
    filename_no_extension = filename.split(".")[0]
    # Check cache to avoid recalcuating from scratch if alreasy done
    if isCached(filename_no_extension, df_identifier):
        return getCache(filename_no_extension, df_identifier)
    # Build useing callback otherwise and cache result
    print(f'BUILDING "{df_identifier}"')
    df = builder_cb(filename)
    cache(filename_no_extension, df_identifier, df)
    return df

    
#expands featureDF into expandedDF if specififed for only a single cell_type 
def getorbuildSubselectExpandedDF(filename, identifier, builder_cb, cell_type, from_scratch):
    filename_no_extension = filename.split(".")[0]
    from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, identifier):
        print(f'BUILDING "{identifier}"')    

        df_raw = getRawDf(filename)
        df_cell_type = df_raw[df_raw['cell_type']== cell_type]
        
        df = builder_cb(df_cell_type) # buildExpandedDF will be the builder_cb
        cache(filename_no_extension, identifier, df)
    else : df = getCache(filename, identifier)
    return df

def getExpandedDfIfFilename(filename_or_df):
    '''
    input: filename or df s
    output: df
    '''
    if not isinstance(filename_or_df,  pd.DataFrame):
        filename=filename_or_df
        df = getExpandedDf(filename_or_df)
    else:
        df = filename_or_df
        filename= "feature_df_py.xlsx" #HARD CODED
        print ('using suplied expanded df')
    return df, filename



########## SETTERS ##########

def updateFPStats(filename, rows):
    FP_stats_df = getFPStats(filename)
    
    # Create a unique identifier in the existing DataFrame
    FP_stats_df['unique_id'] = FP_stats_df.apply(lambda row: '_'.join([str(row[col]) for col in ["cell_type", "cell_id", "measure", "treatment", "pre_post"]]), axis=1)
    
    for row in rows:
        unique_id = '_'.join([str(row[col]) for col in ["cell_type", "cell_id", "measure", "treatment", "pre_post"]])
        data_row = pd.DataFrame([row])
        data_row['unique_id'] = unique_id

        # Check if this unique_id already exists
        if unique_id in FP_stats_df['unique_id'].values:
            # Update the existing row
            match_index = FP_stats_df[FP_stats_df['unique_id'] == unique_id].index[0]
            FP_stats_df.at[match_index, 'mean_value'] = data_row.at[0, 'mean_value']
            FP_stats_df.at[match_index, 'file_values'] = data_row.at[0, 'file_values']
        else:
            # Append the new row
            FP_stats_df = pd.concat([FP_stats_df, data_row], ignore_index=True)

    # Drop the unique identifier column
    FP_stats_df.drop('unique_id', axis=1, inplace=True)

    cache(filename, "FP_stats", FP_stats_df)

def updateAPPStats(filename, rows):
    APP_stats_df = getAPPStats(filename)
    
    # Create a unique identifier 
    APP_stats_df['unique_id'] = APP_stats_df.apply(lambda row: '_'.join([str(row[col]) for col in ["cell_type", "cell_id", "measure", "treatment", "pre_app_wash"]]), axis=1)
    
    for row in rows:
        unique_id = '_'.join([str(row[col]) for col in ["cell_type", "cell_id", "measure", "treatment", "pre_app_wash"]])
        data_row = pd.DataFrame([row])
        data_row['unique_id'] = unique_id

        if unique_id in APP_stats_df['unique_id'].values: #update row
            match_index = APP_stats_df[APP_stats_df['unique_id'] == unique_id].index[0]
            APP_stats_df.at[match_index, 'value'] = data_row.at[0, 'value']
            APP_stats_df.at[match_index, 'sem'] = data_row.at[0, 'sem']
        else: #add row
            APP_stats_df = pd.concat([APP_stats_df, data_row], ignore_index=True)

    APP_stats_df.drop('unique_id', axis=1, inplace=True)# Drop unique identifier 
    cache(filename, "APP_stats", APP_stats_df)


############ BUILDERS ##########

#row handeler for expandDF
def buildExpandedDF(filename_or_df): 
    '''
    input: filename or df (to allow subsetting)
    output: expanded df with aditional columns 'mouseline', 
    '''
    if not isinstance(filename_or_df, pd.DataFrame):
        print('fetchig raw df')
        df = getRawDf(filename_or_df)
    else:
        df=filename_or_df
        print ('expanding on provided df')

    og_columns = df.columns.copy() #origional column order
    df['mouseline'] = df.cell_id.str[:3]

    df = df.apply(lambda row: _handleFile(row), axis=1) #chat gpt but dont get why ?

    #ORDERING DF internal: like this new columns added will appear at the end of the df in the order they were created in _handelfile()
    all_cur_columns = df.columns.copy()
    new_colums_set = set(all_cur_columns ) - set(og_columns) # Subtract mathematical sets of all cols - old columns to get a set of the new columns
    new_colums_li = list(new_colums_set)
    reorder_cols_li =  list(og_columns) + new_colums_li # ['mouseline'] + ordering
    df_reordered = df[reorder_cols_li]    

    return df_reordered

def _handleFile(row):
    '''
    input: single row from feature_df corrisponding to a single file
    output: updated row with aditonal columns
    '''
    row = row.copy() #conserve order
    error_msg = None
    error_traceback = None

    def log_error(msg, tb):
        nonlocal error_msg 
        error_msg=msg
        error_traceback = tb

    #open file
    path_V, path_I = make_path(row.folder_file) 
    V_list, V_df = igor_exporter(path_V)
    V_array      = np.array(V_df) 
    try: # handel missing I traces (Soma_outwave)
        I_list, I_df = igor_exporter(path_I)
        I_array      = np.array(I_df)
    except FileNotFoundError:
        I_list, I_df = None, None
        I_array      = None
        print('I file not found, path:', path_I)

    try:
        # FP data handeling
        if row.data_type in ["FP", "FP_AP"]: 
            
            print("FP type file")
            print(row.folder_file)

            #extraction funcs 
            row["max_firing"] = calculate_max_firing(V_array)
            
            x, y, v_rest = extract_FI_x_y (path_I, path_V) # x = current injected, y = number of APs, v_rest = the steady state voltage (no I injected)
            FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True) #TODO handel pAD APs better slope fit
            row["rheobased_threshold"] = rheobase_threshold
            row["FI_slope"] = FI_slope

            peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(V_df)
            row["voltage_threshold"] = v_thresholds_all 
            row["AP_height"] = peak_heights_all
            row["AP_width"] = peak_fw_all
            row["AP_slope"] = peak_slope_all 
            row["AP_latency"] = peak_latencies_all

            # lists of lists i.e. [[value, steady_state_I, I_injected, RMP], [...] ]
            row["tau_rc"] = tau_analyser(V_array, I_array, x, plotting_viz= False, analysis_mode = 'max')
            row["sag"]    = sag_current_analyser(V_array, I_array, x)

        # APP handeling
        elif row.data_type == "AP":

            print("AP type file")
            print(row.folder_file)        
            
            #input_R 
            if I_array is not None: #if I exists
                if (I_array[:, 0] != 0).any(): #all columns are the same so check column 1 for non 0 value
                    print(f"I injected, calculating inputR ...")
                    input_R_PRE, input_R_APP, input_R_WASH = mean_inputR_APP_calculator(V_array, I_array, row.drug_in, row.drug_out)
                    row['inputR_PRE'] = input_R_PRE
                    row['inputR_APP'] = input_R_APP
                    row['inputR_WASH'] = input_R_WASH
                    pass_I_array = I_array
                else:
                    print("No I injected, cannot calculate input_R.")
                    pass_I_array = None
            else:
                print("I_df does not exist, cannot calculate input_R.")
                pass_I_array = None
                
            mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH = mean_RMP_APP_calculator(V_array, row.drug_in, row.drug_out, I_array=pass_I_array) #with current (I) data 
            row['RMP_PRE'] = mean_RMP_PRE
            row['RMP_APP'] = mean_RMP_APP
            row['RMP_WASH'] = mean_RMP_WASH


            # pAD classification
            peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  =   pAD_detection(V_array)
            print('pAD detection complete .... ')

            if len(pAD_df["pAD_count"]) == 0:  #if there are no APs of any kind detected
                row["pAD_count"] =  np.nan
                row["AP_locs"]   =  pAD_df["AP_loc"].ravel()   # getlocations of ALL? APs : AP_lcos 
            else:
                row["pAD_count"] =  pAD_df["pAD_count"][0]     #  need to make count not series / ERROR HERE  
                row["AP_locs"]   =  pAD_df["AP_loc"].ravel()    

                #lists of positions of action potential 
                row['PRE_Somatic_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
                row['APP_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
                row['WASH_Somatic_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'Somatic') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()
                row['PRE_pAD_AP_locs'] = pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] < row.drug_in), 'AP_loc'].tolist()
                row['APP_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] >= row.drug_in) & (pAD_df['AP_sweep_num'] <= row.drug_out), 'AP_loc'].tolist()
                row['WASH_pAD_AP_locs'] =pAD_df.loc[(pAD_df['pAD'] == 'pAD') & (pAD_df['AP_sweep_num'] > row.drug_out), 'AP_loc'].tolist()
            pass
        
        elif row.data_type == "pAD":
            print(f'{row.folder_file} data_type pAD')

        else:
            raise NotADirectoryError(f"Didn't handle data type: {row.data_type}") # data_type can be AP, FP_AP or FP 
    
    # docuument errors
    except Exception as e:
        error_type = type(e).__name__
        error_tb = traceback.format_exc()
        lines = error_tb.split('\n')
        relevant_tb = []

        for idx, line in enumerate(lines):
            if f'{error_type}: {str(e)}' in line: #take the line before the error message
                relevant_tb = lines[idx - 1].strip()
                break

        log_error(f'{error_type}: {str(e)}', relevant_tb)
        error_traceback = relevant_tb  # Capture traceback within the except block #check redundancey after DJ #28 merge with gpt addition in log_error() def
    if error_msg:
        row['error'] = error_msg
        row['traceback'] = error_traceback
        print(f'{row.cell_id} error message logged: {error_msg}')
        print(f'{row.cell_id} traceback: {error_traceback}')
    else:
        row['error'] = 'ran'

    return row


def buildRawDf(filename):
    file_name, file_type = filename.split(".")
    if file_type.lower() != "xlsx":  # Update the file type check for Excel files
        raise Exception(f'METHOD TO DEAL WITH FILE TYPE "{file_type}" ABSENT')
    if not os.path.isfile(f"{INPUT_DIR}/{filename}"):
        raise Exception(f'FILE {filename} IS ABSENT IN "input/" DIRECTORY')

    df = pd.read_excel (f'{INPUT_DIR}/{filename}', converters={'drug_in':int, 'drug_out':int})
    df['cell_subtype'].fillna('None', inplace=True) #for consistency in lack of subtype specification
    return (df)


def buildFPStatsDf(filename):
    return pd.DataFrame(
        columns=[
            "cell_type",
            "cell_id",
            "measure",
            "treatment",
            "pre_post",
            "mean_value",
            "file_values",
        ]
    )


def buildAPPStatsDf(filename):
    return pd.DataFrame(
        columns=[
            "cell_type",
            "cell_id",
            "measure",
            "treatment",
            "pre_app_wash",
            "value",
            "sem",
        ]
    )

